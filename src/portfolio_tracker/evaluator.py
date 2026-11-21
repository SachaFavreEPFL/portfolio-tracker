import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from portfolio_tracker._http import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    read_env_file_key,
    request_json,
    sanitize_error_message,
)
from portfolio_tracker.contracts import StockData, StockEvaluation, StockEvaluator

logger = logging.getLogger(__name__)

ENV_PATH_ENV_VAR = "PORTFOLIO_TRACKER_ENV_PATH"
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
DEFAULT_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT_SECONDS = 60


def _resolve_default_env_path() -> Path:
    """Résout le chemin du fichier .env depuis l'environnement, le dépôt source ou le dossier courant."""
    env_override = os.environ.get(ENV_PATH_ENV_VAR, "").strip()
    if env_override:
        return Path(env_override).expanduser()

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        return repo_root / ".env"
    return Path.cwd() / ".env"


DEFAULT_ENV_PATH = _resolve_default_env_path()

HttpJsonClient = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]

_EVALUATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdict": {
            "type": "STRING",
            "enum": ["ACHAT", "CONSERVER", "VENDRE"],
        },
        "score": {
            "type": "INTEGER",
            "minimum": 1,
            "maximum": 10,
        },
        "summary": {
            "type": "STRING",
        },
        "strengths": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "risks": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["verdict", "score", "summary", "strengths", "risks"],
}


class EvaluationError(Exception):
    """Erreur de configuration, d'appel API, de décodage ou de validation lors d'une évaluation."""


def http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Envoie une requête HTTP POST JSON via la bibliothèque standard avec retry sur erreurs transitoires."""
    return request_json(
        url,
        payload=payload,
        extra_headers=headers,
        timeout=timeout,
        error_cls=EvaluationError,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        sleep=sleep,
        logger_instance=logger,
    )


def resolve_gemini_api_key(
    api_key: str | None = None,
    env_path: Path | None = None,
) -> str:
    """Résout GEMINI_API_KEY depuis l'argument explicite, l'environnement ou un fichier .env."""
    resolved_env_path = env_path if env_path is not None else _resolve_default_env_path()
    candidate: str | None
    if api_key is not None:
        candidate = api_key
    else:
        env_value = os.environ.get("GEMINI_API_KEY")
        if env_value is not None and env_value.strip():
            candidate = env_value
        else:
            candidate = read_env_file_key(
                resolved_env_path, "GEMINI_API_KEY", error_cls=EvaluationError
            )

    resolved = candidate.strip() if candidate is not None else ""
    if not resolved:
        raise EvaluationError(
            "Clé API Gemini introuvable ou vide : fournissez api_key, définissez "
            "GEMINI_API_KEY ou renseignez le fichier .env."
        )
    return resolved


def _build_evaluation_prompt(stock: StockData) -> str:
    stock_json = json.dumps(stock.model_dump(mode="json"), indent=2, ensure_ascii=False)
    return (
        "Tu es un analyste financier rigoureux. Évalue l'action ci-dessous à partir "
        "uniquement des données fournies et réponds en français au format JSON structuré.\n"
        "- verdict : exactement une valeur parmi ACHAT, CONSERVER ou VENDRE.\n"
        "- score : entier compris entre 1 (très défavorable) et 10 (très favorable).\n"
        "- summary : synthèse concise et factuelle de la situation de l'action.\n"
        "- strengths : liste des principaux points forts observables.\n"
        "- risks : liste des principaux risques ou points de vigilance.\n\n"
        f"Données de l'action :\n{stock_json}"
    )


def _extract_candidate_text(
    response: dict[str, Any],
    symbol: str,
    secret: str | None = None,
) -> str:
    if "error" in response and response["error"] is not None:
        error_payload = response["error"]
        if isinstance(error_payload, dict) and error_payload.get("message"):
            message = str(error_payload["message"])
        else:
            message = str(error_payload)
        sanitized = sanitize_error_message(message, secret)
        raise EvaluationError(f"Erreur renvoyée par l'API Gemini pour {symbol} : {sanitized}")

    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        prompt_feedback = response.get("promptFeedback")
        block_reason = (
            prompt_feedback.get("blockReason")
            if isinstance(prompt_feedback, dict)
            else None
        )
        if isinstance(block_reason, str) and block_reason:
            raise EvaluationError(
                f"Requête bloquée par l'API Gemini pour {symbol} (blockReason={block_reason})."
            )
        raise EvaluationError(f"Aucun candidat valide renvoyé par l'API Gemini pour {symbol}.")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    if isinstance(finish_reason, str) and finish_reason and finish_reason != "STOP":
        raise EvaluationError(
            f"Réponse Gemini incomplète ou bloquée pour {symbol} (finishReason={finish_reason})."
        )

    content = candidate.get("content")
    if not isinstance(content, dict):
        raise EvaluationError(f"Contenu absent dans la réponse Gemini pour {symbol}.")

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise EvaluationError(f"Parties de contenu absentes dans la réponse Gemini pour {symbol}.")

    text_chunks: list[str] = []
    for part in parts:
        # Ignore les blocs de réflexion interne éventuellement émis par les modèles Gemini 2.5.
        if not isinstance(part, dict) or part.get("thought") is True:
            continue
        text = part.get("text")
        if isinstance(text, str):
            text_chunks.append(text)

    combined = "".join(text_chunks).strip()
    if not combined:
        raise EvaluationError(f"Texte de réponse vide renvoyé par l'API Gemini pour {symbol}.")
    return combined


class GeminiStockEvaluator(StockEvaluator):
    """Évalue une action en interrogeant l'API Gemini generateContent."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        http_client: HttpJsonClient = http_post_json,
        env_path: Path | None = None,
        api_base_url: str = DEFAULT_GEMINI_API_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = resolve_gemini_api_key(api_key=api_key, env_path=env_path)
        cleaned_model = model.strip()
        if not cleaned_model:
            raise EvaluationError("Le nom du modèle Gemini ne peut pas être vide.")
        cleaned_base_url = api_base_url.strip().rstrip("/")
        if not cleaned_base_url:
            raise EvaluationError("L'URL de base de l'API Gemini ne peut pas être vide.")
        if isinstance(timeout, bool) or timeout <= 0:
            raise EvaluationError("Le timeout doit être un entier strictement positif.")

        self._model = cleaned_model
        self._http_client = http_client
        self._api_base_url = cleaned_base_url
        self._timeout = timeout

    def evaluate(self, stock: StockData) -> StockEvaluation:
        url = f"{self._api_base_url}/models/{self._model}:generateContent"
        headers = {"x-goog-api-key": self._api_key}
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": _build_evaluation_prompt(stock)}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _EVALUATION_RESPONSE_SCHEMA,
            },
        }

        try:
            response = self._http_client(url, payload, headers, self._timeout)
        except EvaluationError:
            raise
        except Exception as exc:
            raw_detail = str(exc)
            detail = sanitize_error_message(raw_detail, self._api_key)
            cause = None if self._api_key in raw_detail else exc
            raise EvaluationError(
                f"Erreur lors de l'appel à l'API Gemini pour {stock.symbol} : {detail}"
            ) from cause

        if not isinstance(response, dict):
            raise EvaluationError(
                f"Format de réponse inattendu pour {stock.symbol} : {type(response).__name__}."
            )

        raw_text = _extract_candidate_text(response, stock.symbol, secret=self._api_key)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"Réponse JSON invalide reçue de Gemini pour {stock.symbol} : {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise EvaluationError(
                f"La réponse JSON de Gemini pour {stock.symbol} doit être un objet."
            )

        evaluation_data: dict[str, Any] = {
            **parsed,
            "symbol": stock.symbol,
            "model": self._model,
        }
        try:
            return StockEvaluation.model_validate(evaluation_data)
        except ValidationError as exc:
            raise EvaluationError(
                f"Évaluation invalide reçue de Gemini pour {stock.symbol} : {exc}"
            ) from exc
