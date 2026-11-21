import http.client
import io
import json
import traceback
import urllib.error
import urllib.request
from http.client import HTTPMessage
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from portfolio_tracker.contracts import StockData, StockEvaluation, Verdict
from portfolio_tracker.evaluator import (
    DEFAULT_GEMINI_API_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    ENV_PATH_ENV_VAR,
    EvaluationError,
    GeminiStockEvaluator,
    http_post_json,
    resolve_gemini_api_key,
)


def _sample_stock() -> StockData:
    return StockData(
        symbol="NVDA",
        name="NVIDIA Corporation",
        currency="USD",
        price=140.5,
        fifty_two_week_low=60.0,
        fifty_two_week_high=150.0,
        market_cap=3_400_000_000_000.0,
        trailing_pe=52.3,
        forward_pe=35.1,
        debt_to_equity=17.2,
        profit_margins=0.55,
        revenue_growth=0.94,
        recommendation_key="buy",
    )


def _valid_evaluation_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "verdict": "ACHAT",
        "score": 8,
        "summary": "Croissance exceptionnelle et marges très élevées.",
        "strengths": ["Forte croissance du chiffre d'affaires", "Marges nettes élevées"],
        "risks": ["Valorisation exigeante", "Cycle semi-conducteurs"],
    }
    payload.update(overrides)
    return payload


def _gemini_response_from_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": parts,
                }
            }
        ]
    }


def _gemini_response_from_text(text: str) -> dict[str, Any]:
    return _gemini_response_from_parts([{"text": text}])


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


# --- 1. Succès de l'évaluation ---


def test_evaluate_success_checks_request_and_returns_stock_evaluation() -> None:
    captured: dict[str, Any] = {}
    stock = _sample_stock()
    api_response = _gemini_response_from_text(json.dumps(_valid_evaluation_payload()))

    def fake_http_client(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return api_response

    evaluator = GeminiStockEvaluator(
        api_key="test-secret-key",
        model="gemini-4-neon",
        http_client=fake_http_client,
        api_base_url="https://api.example.test/v1beta/",
        timeout=30,
    )

    result = evaluator.evaluate(stock)

    assert (
        captured["url"]
        == "https://api.example.test/v1beta/models/gemini-4-neon:generateContent"
    )
    assert captured["headers"] == {"x-goog-api-key": "test-secret-key"}
    assert captured["timeout"] == 30

    sent_payload = captured["payload"]
    prompt_text = sent_payload["contents"][0]["parts"][0]["text"]
    assert "NVDA" in prompt_text
    assert "NVIDIA Corporation" in prompt_text
    assert "140.5" in prompt_text
    assert "35.1" in prompt_text

    gen_config = sent_payload["generationConfig"]
    assert gen_config["responseMimeType"] == "application/json"
    schema = gen_config["responseSchema"]
    assert schema["type"] == "OBJECT"
    assert schema["properties"]["verdict"]["enum"] == ["ACHAT", "CONSERVER", "VENDRE"]
    assert schema["properties"]["score"]["minimum"] == 1
    assert schema["properties"]["score"]["maximum"] == 10
    assert set(schema["required"]) == {"verdict", "score", "summary", "strengths", "risks"}

    assert isinstance(result, StockEvaluation)
    assert result.symbol == "NVDA"
    assert result.verdict == "ACHAT"
    assert result.score == 8
    assert result.summary == "Croissance exceptionnelle et marges très élevées."
    assert result.strengths == ["Forte croissance du chiffre d'affaires", "Marges nettes élevées"]
    assert result.risks == ["Valorisation exigeante", "Cycle semi-conducteurs"]
    assert result.model == "gemini-4-neon"

    with pytest.raises(ValidationError):
        setattr(result, "score", 5)


def test_evaluate_ignores_thought_parts_and_uses_final_json() -> None:
    response = _gemini_response_from_parts(
        [
            {"thought": True, "text": "Réflexion interne qui n'est pas du JSON valide {["},
            {"text": json.dumps(_valid_evaluation_payload(verdict="CONSERVER", score=6))},
        ]
    )

    evaluator = GeminiStockEvaluator(
        api_key="test-key",
        http_client=lambda url, payload, headers, timeout: response,
    )

    result = evaluator.evaluate(_sample_stock())

    assert result.verdict == "CONSERVER"
    assert result.score == 6
    assert result.model == DEFAULT_GEMINI_MODEL


def test_evaluate_uses_default_base_url_and_overrides_symbol_and_model() -> None:
    captured_url: list[str] = []
    # Même si Gemini renvoie symbol/model dans son JSON, l'évaluateur impose les siens.
    raw_payload = _valid_evaluation_payload(symbol="WRONG", model="wrong-model")
    response = _gemini_response_from_text(json.dumps(raw_payload))

    def fake_http_client(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        captured_url.append(url)
        return response

    evaluator = GeminiStockEvaluator(api_key="test-key", http_client=fake_http_client)
    result = evaluator.evaluate(_sample_stock())

    assert captured_url == [
        f"{DEFAULT_GEMINI_API_BASE_URL}/models/{DEFAULT_GEMINI_MODEL}:generateContent"
    ]
    assert result.symbol == "NVDA"
    assert result.model == DEFAULT_GEMINI_MODEL


# --- 2. Résolution de la clé API ---


def test_resolve_api_key_prefers_explicit_argument_over_env_and_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    resolved = resolve_gemini_api_key(api_key="  from-arg  ", env_path=env_file)

    assert resolved == "from-arg"


def test_resolve_api_key_from_environment_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "  from-env-key  ")

    resolved = resolve_gemini_api_key(env_path=env_file)

    assert resolved == "from-env-key"


@pytest.mark.parametrize(
    ("env_content", "expected"),
    [
        (
            "\n# Commentaire\nOTHER_KEY=foo\nGEMINI_API_KEY=plain-key\n",
            "plain-key",
        ),
        (
            "export GEMINI_API_KEY=\"double-quoted-key\"\n",
            "double-quoted-key",
        ),
        (
            "export\tGEMINI_API_KEY='single-quoted-key'\n",
            "single-quoted-key",
        ),
        (
            "# Ligne sans égal ignorée\nINVALID_LINE\nGEMINI_API_KEY=first\nGEMINI_API_KEY=last\n",
            "last",
        ),
    ],
)
def test_resolve_api_key_from_dotenv_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_content: str,
    expected: str,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(env_content, encoding="utf-8")

    resolved = resolve_gemini_api_key(env_path=env_file)

    assert resolved == expected


def test_resolve_api_key_falls_back_to_dotenv_when_env_var_is_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    env_file = tmp_path / ".env"
    env_file.write_text('GEMINI_API_KEY="fallback-key"\n', encoding="utf-8")

    assert resolve_gemini_api_key(env_path=env_file) == "fallback-key"


@pytest.mark.parametrize(
    ("explicit_key", "env_var", "dotenv_content"),
    [
        (None, None, None),
        (None, "", None),
        (None, "   ", "# seulement un commentaire\nOTHER=123\n"),
        (None, None, 'GEMINI_API_KEY=""\n'),
        ("   ", "from-env-should-not-be-used", "GEMINI_API_KEY=from-dotenv\n"),
    ],
)
def test_resolve_api_key_missing_or_empty_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_key: str | None,
    env_var: str | None,
    dotenv_content: str | None,
) -> None:
    if env_var is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", env_var)

    env_file = tmp_path / ".env"
    if dotenv_content is not None:
        env_file.write_text(dotenv_content, encoding="utf-8")

    with pytest.raises(EvaluationError):
        resolve_gemini_api_key(api_key=explicit_key, env_path=env_file)

    with pytest.raises(EvaluationError):
        GeminiStockEvaluator(api_key=explicit_key, env_path=env_file)


# --- 3. JSON invalide et structure de réponse invalide ---


@pytest.mark.parametrize(
    "raw_text",
    [
        "{invalid json",
        "not json at all",
        "[1, 2, 3]",
        "null",
        '"just a string"',
        "42",
    ],
)
def test_evaluate_raises_on_invalid_or_non_object_json(raw_text: str) -> None:
    response = _gemini_response_from_text(raw_text)
    evaluator = GeminiStockEvaluator(
        api_key="test-key",
        http_client=lambda url, payload, headers, timeout: response,
    )

    with pytest.raises(EvaluationError):
        evaluator.evaluate(_sample_stock())


@pytest.mark.parametrize(
    "api_response",
    [
        {},
        {"candidates": []},
        {"candidates": ["not-a-dict"]},
        {"candidates": [{}]},
        {"candidates": [{"content": {}}]},
        {"candidates": [{"content": {"parts": "not-a-list"}}]},
        {"candidates": [{"content": {"parts": []}}]},
        {"candidates": [{"content": {"parts": [{"text": "   "}]}}]},
        {"candidates": [{"content": {"parts": [{"thought": True, "text": "{}"}]}}]},
    ],
)
def test_evaluate_raises_on_invalid_response_structure(api_response: dict[str, Any]) -> None:
    evaluator = GeminiStockEvaluator(
        api_key="test-key",
        http_client=lambda url, payload, headers, timeout: api_response,
    )

    with pytest.raises(EvaluationError):
        evaluator.evaluate(_sample_stock())


def test_evaluate_raises_when_http_client_returns_non_dict() -> None:
    def bad_client(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> Any:
        return ["unexpected", "list"]

    evaluator = GeminiStockEvaluator(api_key="test-key", http_client=bad_client)

    with pytest.raises(EvaluationError):
        evaluator.evaluate(_sample_stock())


# --- 4. Erreur API / réseau / HTTP ---


def test_evaluate_wraps_http_client_exception_and_redacts_api_key() -> None:
    secret_key = "AIzaSySecretKey12345"

    def failing_client(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        raise RuntimeError(f"network down for key {secret_key}")

    evaluator = GeminiStockEvaluator(api_key=secret_key, http_client=failing_client)

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate(_sample_stock())

    message = str(exc_info.value)
    formatted_tb = "".join(traceback.format_exception(exc_info.value))
    assert secret_key not in message
    assert secret_key not in formatted_tb
    assert "***" in message
    assert "network down" in message
    assert exc_info.value.__cause__ is None


def test_evaluate_preserves_cause_when_exception_has_no_secret() -> None:
    def failing_client(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        raise RuntimeError("plain network error")

    evaluator = GeminiStockEvaluator(api_key="test-secret-key", http_client=failing_client)

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate(_sample_stock())

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_evaluate_propagates_evaluation_error_from_http_client() -> None:
    original_error = EvaluationError("erreur déjà formatée")

    def failing_client(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        raise original_error

    evaluator = GeminiStockEvaluator(api_key="test-key", http_client=failing_client)

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate(_sample_stock())

    assert exc_info.value is original_error


@pytest.mark.parametrize(
    ("error_response", "expected_fragment"),
    [
        ({"error": {"code": 429, "message": "Quota exceeded"}}, "Quota exceeded"),
        ({"error": "Service temporarily unavailable"}, "Service temporarily unavailable"),
    ],
)
def test_evaluate_raises_on_api_error_payload(
    error_response: dict[str, Any], expected_fragment: str
) -> None:
    evaluator = GeminiStockEvaluator(
        api_key="test-key",
        http_client=lambda url, payload, headers, timeout: error_response,
    )

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate(_sample_stock())

    assert expected_fragment in str(exc_info.value)


def test_http_post_json_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["data"] = request.data
        captured["timeout"] = timeout
        return _FakeHTTPResponse(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = http_post_json(
        url="https://api.example.test/generate",
        payload={"hello": "world"},
        headers={"x-goog-api-key": "secret-token"},
        timeout=15,
    )

    assert result == {"ok": True}
    assert captured["url"] == "https://api.example.test/generate"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 15
    assert json.loads(captured["data"].decode("utf-8")) == {"hello": "world"}
    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers_lower["content-type"] == "application/json"
    assert headers_lower["x-goog-api-key"] == "secret-token"


@pytest.mark.parametrize(
    "error_body_builder",
    [
        lambda secret: json.dumps({"error": {"message": f"Invalid API key {secret}"}}).encode(
            "utf-8"
        ),
        lambda secret: f"Forbidden for token {secret}".encode(),
    ],
)
def test_http_post_json_http_error_sanitizes_api_key(
    monkeypatch: pytest.MonkeyPatch,
    error_body_builder: Any,
) -> None:
    secret = "super-secret-gemini-key"
    error_body: bytes = error_body_builder(secret)

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=403,
            msg="Forbidden",
            hdrs=HTTPMessage(),
            fp=io.BytesIO(error_body),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(EvaluationError) as exc_info:
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={"x-goog-api-key": secret},
            timeout=10,
        )

    message = str(exc_info.value)
    assert "403" in message
    assert secret not in message
    assert "***" in message
    assert isinstance(exc_info.value.__cause__, urllib.error.HTTPError)


def test_http_post_json_url_error_sanitizes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "secret-url-error-key"

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        raise urllib.error.URLError(f"DNS failure with {secret}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(EvaluationError) as exc_info:
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={"x-goog-api-key": secret},
            timeout=10,
            max_retries=0,
        )

    message = str(exc_info.value)
    formatted_tb = "".join(traceback.format_exception(exc_info.value))
    assert secret not in message
    assert secret not in formatted_tb
    assert "***" in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("invalid_timeout", [0, -1, True, False])
def test_http_post_json_rejects_invalid_timeout(invalid_timeout: int) -> None:
    with pytest.raises(EvaluationError, match="timeout"):
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={},
            timeout=invalid_timeout,
        )


def test_http_post_json_redacts_bearer_authorization_and_hides_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "tok-secret-bearer-999"

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        raise urllib.error.URLError(f"proxy rejected token {token}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(EvaluationError) as exc_info:
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            max_retries=0,
        )

    formatted_tb = "".join(traceback.format_exception(exc_info.value))
    assert token not in str(exc_info.value)
    assert token not in formatted_tb
    assert "***" in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_http_post_json_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(EvaluationError) as exc_info:
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={},
            timeout=12,
            max_retries=0,
        )

    assert "12s" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, TimeoutError)


@pytest.mark.parametrize("raw_body", [b"<html>Bad Gateway</html>", b"[1, 2, 3]"])
def test_http_post_json_invalid_or_non_object_http_response(
    monkeypatch: pytest.MonkeyPatch, raw_body: bytes
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(raw_body),
    )

    with pytest.raises(EvaluationError):
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={},
            timeout=10,
        )


# --- 5. Validation Pydantic ---


@pytest.mark.parametrize("verdict", ["ACHAT", "CONSERVER", "VENDRE"])
@pytest.mark.parametrize("score", [1, 5, 10])
def test_stock_evaluation_accepts_valid_verdicts_and_scores(
    verdict: Verdict, score: int
) -> None:
    evaluation = StockEvaluation(
        symbol="AAPL",
        verdict=verdict,
        score=score,
        summary="Synthèse factuelle.",
        strengths=["Point fort"],
        risks=["Risque"],
        model=DEFAULT_GEMINI_MODEL,
    )

    assert evaluation.verdict == verdict
    assert evaluation.score == score
    with pytest.raises(ValidationError):
        setattr(evaluation, "verdict", "VENDRE")


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"verdict": "HOLD"},
        {"score": 0},
        {"score": 11},
        {"score": True},
        {"score": False},
        {"score": 7.5},
        {"score": "haut"},
        {"strengths": "pas une liste"},
    ],
)
def test_stock_evaluation_direct_validation_rejects_invalid_values(
    invalid_fields: dict[str, Any],
) -> None:
    data: dict[str, Any] = {
        **_valid_evaluation_payload(),
        "symbol": "AAPL",
        "model": DEFAULT_GEMINI_MODEL,
        **invalid_fields,
    }

    with pytest.raises(ValidationError):
        StockEvaluation.model_validate(data)


@pytest.mark.parametrize(
    "missing_field",
    ["symbol", "verdict", "score", "summary", "strengths", "risks", "model"],
)
def test_stock_evaluation_direct_validation_rejects_missing_fields(missing_field: str) -> None:
    data: dict[str, Any] = {
        **_valid_evaluation_payload(),
        "symbol": "AAPL",
        "model": DEFAULT_GEMINI_MODEL,
    }
    del data[missing_field]

    with pytest.raises(ValidationError):
        StockEvaluation.model_validate(data)


@pytest.mark.parametrize(
    "bad_gemini_payload",
    [
        _valid_evaluation_payload(verdict="HOLD"),
        _valid_evaluation_payload(score=0),
        _valid_evaluation_payload(score=11),
        _valid_evaluation_payload(score=True),
        _valid_evaluation_payload(strengths="Croissance forte"),
        {
            "verdict": "ACHAT",
            "score": 8,
            "strengths": ["Point fort"],
            "risks": ["Risque"],
        },
    ],
)
def test_evaluate_wraps_pydantic_validation_error(bad_gemini_payload: dict[str, Any]) -> None:
    response = _gemini_response_from_text(json.dumps(bad_gemini_payload))
    evaluator = GeminiStockEvaluator(
        api_key="test-key",
        http_client=lambda url, payload, headers, timeout: response,
    )

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate(_sample_stock())

    assert isinstance(exc_info.value.__cause__, ValidationError)


# --- 6. Validation des paramètres de configuration ---


@pytest.mark.parametrize("invalid_model", ["", "   "])
def test_evaluator_rejects_empty_model(invalid_model: str) -> None:
    with pytest.raises(EvaluationError):
        GeminiStockEvaluator(api_key="test-key", model=invalid_model)


@pytest.mark.parametrize("invalid_base_url", ["", "   ", "/"])
def test_evaluator_rejects_empty_api_base_url(invalid_base_url: str) -> None:
    with pytest.raises(EvaluationError):
        GeminiStockEvaluator(api_key="test-key", api_base_url=invalid_base_url)


@pytest.mark.parametrize("invalid_timeout", [0, -1, -30, True, False])
def test_evaluator_rejects_invalid_timeout(invalid_timeout: int) -> None:
    with pytest.raises(EvaluationError):
        GeminiStockEvaluator(api_key="test-key", timeout=invalid_timeout)


def test_resolve_api_key_uses_env_path_override_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    custom_env = tmp_path / "custom.env"
    custom_env.write_text("GEMINI_API_KEY=from-custom-env-path\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PATH_ENV_VAR, str(custom_env))

    assert resolve_gemini_api_key() == "from-custom-env-path"


@pytest.mark.parametrize("status_code", [429, 503])
def test_http_post_json_retries_transient_http_errors_with_backoff(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    attempts = 0
    delays: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise urllib.error.HTTPError(
                url=request.full_url,
                code=status_code,
                msg="Transient",
                hdrs=HTTPMessage(),
                fp=io.BytesIO(b'{"error": {"message": "Try again"}}'),
            )
        return _FakeHTTPResponse(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = http_post_json(
        url="https://api.example.test/generate",
        payload={"a": 1},
        headers={},
        timeout=10,
        max_retries=2,
        backoff_seconds=0.5,
        sleep=delays.append,
    )

    assert result == {"ok": True}
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_http_post_json_retries_timeout_and_fails_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(EvaluationError, match="10s"):
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={},
            timeout=10,
            max_retries=2,
            backoff_seconds=0.25,
            sleep=delays.append,
        )

    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_http_post_json_does_not_retry_non_retryable_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=400,
            msg="Bad Request",
            hdrs=HTTPMessage(),
            fp=io.BytesIO(b'{"error": {"message": "Invalid argument"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(EvaluationError, match="400"):
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={},
            timeout=10,
            max_retries=2,
            backoff_seconds=0.5,
            sleep=delays.append,
        )

    assert attempts == 1
    assert delays == []


def test_http_post_json_retries_http_protocol_exception_and_fails_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        nonlocal attempts
        attempts += 1
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(EvaluationError, match="protocole HTTP"):
        http_post_json(
            url="https://api.example.test/generate",
            payload={"a": 1},
            headers={},
            timeout=10,
            max_retries=2,
            backoff_seconds=0.25,
            sleep=delays.append,
        )

    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_http_post_json_respects_retry_after_and_closes_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    streams: list[io.BytesIO] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            hdrs = HTTPMessage()
            hdrs["Retry-After"] = "7"
            stream = io.BytesIO(b'{"error": {"message": "Rate limited"}}')
            streams.append(stream)
            raise urllib.error.HTTPError(
                url=request.full_url,
                code=429,
                msg="Too Many Requests",
                hdrs=hdrs,
                fp=stream,
            )
        return _FakeHTTPResponse(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = http_post_json(
        url="https://api.example.test/generate",
        payload={"a": 1},
        headers={},
        timeout=10,
        max_retries=2,
        backoff_seconds=1.0,
        sleep=delays.append,
    )

    assert result == {"ok": True}
    assert attempts == 2
    assert delays == [7.0]
    assert streams[0].closed


@pytest.mark.parametrize(
    ("api_response", "expected_fragment"),
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "blockReason=SAFETY"),
        (
            {
                "candidates": [
                    {
                        "finishReason": "MAX_TOKENS",
                        "content": {"parts": [{"text": '{"verdict": "ACHAT"'}]},
                    }
                ]
            },
            "finishReason=MAX_TOKENS",
        ),
        (
            {"candidates": [{"finishReason": "SAFETY"}]},
            "finishReason=SAFETY",
        ),
    ],
)
def test_evaluate_reports_block_reason_and_non_stop_finish_reason(
    api_response: dict[str, Any], expected_fragment: str
) -> None:
    evaluator = GeminiStockEvaluator(
        api_key="test-key",
        http_client=lambda url, payload, headers, timeout: api_response,
    )

    with pytest.raises(EvaluationError, match=expected_fragment):
        evaluator.evaluate(_sample_stock())
