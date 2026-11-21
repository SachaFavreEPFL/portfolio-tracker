"""Tests unitaires pour harness.providers (sans appel réseau réel)."""

import http.client
import io
import json
import urllib.error
import urllib.request
from http.client import HTTPMessage
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from harness.providers import (
    GeminiProvider,
    ProviderError,
    ToolCall,
    _read_env_file_key,
    create_provider,
    http_json,
)

SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lit un fichier.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def test_gemini_session_handles_tool_call_and_ignores_thoughts() -> None:
    http_client = MagicMock(
        side_effect=[
            {
                "steps": [
                    {"type": "thought", "text": "réflexion interne"},
                    {
                        "type": "function_call",
                        "id": "call-1",
                        "name": "read_file",
                        "arguments": {"path": "AGENTS.md"},
                    },
                ]
            },
            {
                "steps": [
                    {"type": "thought", "text": "autre réflexion"},
                    {"type": "model_output", "content": [{"type": "text", "text": "Réponse finale."}]},
                ]
            },
        ]
    )
    provider = GeminiProvider("gemini-4-neon", api_key="fake-key", http_client=http_client)
    session = provider.start_session("sys", "user", SAMPLE_TOOLS)

    first = session.next_turn()
    assert first.tool_calls == [ToolCall(id="call-1", name="read_file", arguments={"path": "AGENTS.md"})]
    assert first.text == ""

    session.add_tool_result(first.tool_calls[0], "# AGENTS")
    second = session.next_turn()
    assert second.text == "Réponse finale."
    assert second.tool_calls == []

    # Vérifie le format Gemini des outils et l'en-tête API.
    first_call = http_client.call_args_list[0]
    assert first_call.args[1]["tools"] == [
        {"type": "function", **SAMPLE_TOOLS[0]["function"]}
    ]
    assert first_call.args[2] == {"x-goog-api-key": "fake-key"}


def test_gemini_provider_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        GeminiProvider("gemini-4-hydrogen", env_path=tmp_path / ".env")


def test_create_provider_enforces_gemini_models() -> None:
    http_client = MagicMock()

    explicit = create_provider("gemini-4-hydrogen", api_key="k", http_client=http_client)
    assert isinstance(explicit, GeminiProvider)
    assert explicit.model == "gemini-4-hydrogen"

    default_provider = create_provider(None, api_key="k", http_client=http_client)
    assert isinstance(default_provider, GeminiProvider)
    assert default_provider.model.startswith("gemini-")

    with pytest.raises(ProviderError, match="Modèle refusé"):
        create_provider("qwen3.8", api_key="k", http_client=http_client)


def test_gemini_session_handles_non_dict_tool_arguments() -> None:
    http_client = MagicMock(
        return_value={
            "steps": [
                {
                    "type": "function_call",
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": ["unexpected", "list"],
                },
                {
                    "type": "function_call",
                    "id": "call-2",
                    "name": "list_files",
                    "arguments": None,
                },
                {
                    "type": "function_call",
                    "id": "call-3",
                    "name": "list_files",
                    "arguments": "{invalid json",
                },
            ]
        }
    )
    provider = GeminiProvider("gemini-4-neon", api_key="fake-key", http_client=http_client)
    session = provider.start_session("sys", "user", SAMPLE_TOOLS)

    turn = session.next_turn()
    assert [call.arguments for call in turn.tool_calls] == [{}, {}, {}]


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def test_http_json_get_and_post_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str, Any]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _FakeHTTPResponse:
        captured.append((request.full_url, request.get_method(), request.data))
        return _FakeHTTPResponse(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert http_json("https://api.example.test/tags") == {"ok": True}
    assert http_json("https://api.example.test/chat", {"a": 1}) == {"ok": True}
    assert captured[0][1] == "GET"
    assert captured[0][2] is None
    assert captured[1][1] == "POST"
    assert captured[1][2] == b'{"a": 1}'


def test_http_json_http_error_includes_body_and_redacts_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret-harness-key-999"
    error_body = json.dumps({"error": {"message": f"Invalid token {secret}"}}).encode("utf-8")

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=HTTPMessage(),
            fp=io.BytesIO(error_body),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ProviderError) as exc_info:
        http_json(
            "https://api.example.test/interactions",
            {"model": "gemini-4-neon"},
            {"x-goog-api-key": secret},
        )

    message = str(exc_info.value)
    assert "401" in message
    assert "Invalid token ***" in message
    assert secret not in message


@pytest.mark.parametrize("raw_body", [b"<html>502</html>", b"[1, 2, 3]"])
def test_http_json_rejects_non_json_or_non_dict_response(
    monkeypatch: pytest.MonkeyPatch, raw_body: bytes
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(raw_body),
    )

    with pytest.raises(ProviderError):
        http_json("https://api.example.test/interactions", {"a": 1})


def test_http_json_wraps_url_error_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "secret-net-key"
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            urllib.error.URLError(f"failed with {secret}")
        ),
    )
    with pytest.raises(ProviderError) as exc_info:
        http_json(
            "https://api.example.test/interactions",
            {"a": 1},
            {"x-goog-api-key": secret},
            max_retries=0,
        )
    assert secret not in str(exc_info.value)
    assert "***" in str(exc_info.value)

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    with pytest.raises(ProviderError, match="15s"):
        http_json("https://api.example.test/interactions", {"a": 1}, timeout=15, max_retries=0)


@pytest.mark.parametrize("status_code", [429, 503])
def test_http_json_retries_transient_http_errors_with_backoff(
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

    result = http_json(
        "https://api.example.test/interactions",
        {"a": 1},
        max_retries=2,
        backoff_seconds=0.5,
        sleep=delays.append,
    )

    assert result == {"ok": True}
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_http_json_retries_http_protocol_exception_and_fails_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Any:
        nonlocal attempts
        attempts += 1
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ProviderError, match="protocole HTTP"):
        http_json(
            "https://api.example.test/interactions",
            {"a": 1},
            max_retries=2,
            backoff_seconds=0.25,
            sleep=delays.append,
        )

    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_http_json_does_not_retry_non_retryable_http_error(
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
            fp=io.BytesIO(b'{"error": {"message": "Bad request"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ProviderError, match="400"):
        http_json(
            "https://api.example.test/interactions",
            {"a": 1},
            max_retries=2,
            backoff_seconds=0.5,
            sleep=delays.append,
        )

    assert attempts == 1
    assert delays == []


def test_read_env_file_key_handles_formats_and_invalid_utf8(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"
    assert _read_env_file_key(missing, "GEMINI_API_KEY") == ""

    env_file = tmp_path / ".env"
    env_file.write_text('export GEMINI_API_KEY="my-secret"\n', encoding="utf-8")
    assert _read_env_file_key(env_file, "GEMINI_API_KEY") == "my-secret"

    bad_utf8 = tmp_path / "bad.env"
    bad_utf8.write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(ProviderError, match="Impossible de lire le fichier d'environnement"):
        _read_env_file_key(bad_utf8, "GEMINI_API_KEY")


def test_sessions_handle_null_or_malformed_payloads() -> None:
    gemini_null = GeminiProvider(
        "gemini-4-neon",
        api_key="secret-k",
        http_client=MagicMock(return_value={"steps": None}),
    )
    turn = gemini_null.start_session("sys", "user", SAMPLE_TOOLS).next_turn()
    assert turn.text == ""
    assert turn.tool_calls == []

    gemini_bad = GeminiProvider(
        "gemini-4-neon",
        api_key="secret-k",
        http_client=MagicMock(return_value={"steps": "not-a-list"}),
    )
    with pytest.raises(ProviderError, match="steps"):
        gemini_bad.start_session("sys", "user", SAMPLE_TOOLS).next_turn()

    gemini_err = GeminiProvider(
        "gemini-4-neon",
        api_key="secret-k",
        http_client=MagicMock(return_value={"error": {"message": "Quota exceeded for secret-k"}}),
    )
    with pytest.raises(ProviderError) as exc_info:
        gemini_err.start_session("sys", "user", SAMPLE_TOOLS).next_turn()
    assert "secret-k" not in str(exc_info.value)
    assert "Quota exceeded" in str(exc_info.value)

    gemini_non_str_text = GeminiProvider(
        "gemini-4-neon",
        api_key="secret-k",
        http_client=MagicMock(
            return_value={
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": 123},
                            {"type": "text", "text": "Valide"},
                        ],
                    }
                ]
            }
        ),
    )
    turn_ok = gemini_non_str_text.start_session("sys", "user", SAMPLE_TOOLS).next_turn()
    assert turn_ok.text == "Valide"


def test_harness_modules_do_not_import_portfolio_tracker() -> None:
    import ast

    harness_dir = Path(__file__).resolve().parents[1]
    for py_file in harness_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("portfolio_tracker"), (
                        f"Import interdit '{alias.name}' dans {py_file.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("portfolio_tracker"), (
                    f"Import interdit '{node.module}' dans {py_file.name}"
                )


def test_http_modules_stay_in_sync_between_harness_and_portfolio_tracker() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    harness_http = (repo_root / "harness" / "_http.py").read_text(encoding="utf-8").splitlines()
    app_http = (
        (repo_root / "src" / "portfolio_tracker" / "_http.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    # Ignore uniquement la docstring de module en ligne 1.
    assert harness_http[1:] == app_http[1:]


def test_http_json_redacts_authorization_header_and_suppresses_leaking_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import traceback

    token = "harness-bearer-secret-777"
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            urllib.error.URLError(f"upstream rejected {token}")
        ),
    )

    with pytest.raises(ProviderError) as exc_info:
        http_json(
            "https://api.example.test/interactions",
            {"a": 1},
            {"Authorization": f"Bearer {token}"},
            max_retries=0,
        )

    formatted_tb = "".join(traceback.format_exception(exc_info.value))
    assert token not in str(exc_info.value)
    assert token not in formatted_tb
    assert exc_info.value.__cause__ is None
