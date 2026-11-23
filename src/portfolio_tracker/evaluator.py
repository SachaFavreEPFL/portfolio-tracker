import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from portfolio_tracker._http import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    MAX_RESPONSE_BYTES,
    read_env_file_key,
    request_json,
    sanitize_error_message,
)
from portfolio_tracker.contracts import (
    MAX_COMMENT_LENGTH,
    StockData,
    StockEvaluation,
    StockEvaluator,
)

logger = logging.getLogger(__name__)

ENV_PATH_ENV_VAR = "PORTFOLIO_TRACKER_ENV_PATH"
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
DEFAULT_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_VALIDATION_RETRIES = 1


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
            "minimum": 0,
            "maximum": 10,
        },
        "comment": {
            "type": "STRING",
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
    "required": ["verdict", "score", "comment", "summary", "strengths", "risks"],
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
    max_response_bytes: int = MAX_RESPONSE_BYTES,
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
        max_response_bytes=max_response_bytes,
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _build_evaluation_prompt(
    stock: StockData,
    *,
    analysis_date: str | None = None,
) -> str:
    stock_json = json.dumps(stock.model_dump(mode="json"), indent=2, ensure_ascii=False)
    date_line = f"Date d'analyse : {analysis_date}\n" if analysis_date else ""
    return (
        "Tu es un analyste financier rigoureux. Évalue l'action ci-dessous en croisant "
        "les données financières fournies avec une recherche Google sur son actualité récente "
        "(derniers résultats trimestriels, guidance, annonces stratégiques ou produits, "
        "risques réglementaires ou géopolitiques, mouvements récents du cours) "
        "et réponds en français au format JSON structuré.\n"
        "Les données financières fournies peuvent refléter des résultats passés avec du retard : "
        "pondère ton verdict et ton score si l'actualité récente remet en cause la dynamique historique.\n"
        f"{date_line}"
        "- verdict : exactement une valeur parmi ACHAT, CONSERVER ou VENDRE.\n"
        "- score : entier compris entre 0 (très défavorable) et 10 (très favorable).\n"
        "- Cohérence stricte entre score et verdict : 0-3 => VENDRE, 4-6 => CONSERVER, 7-10 => ACHAT.\n"
        f"- comment : avis concis en français ({MAX_COMMENT_LENGTH} caractères maximum) "
        "résumant la thèse et la décision pour un tableur.\n"
        "- summary : synthèse concise et factuelle intégrant fondamentaux et actualité récente.\n"
        "- strengths : liste des principaux points forts observables.\n"
        "- risks : liste des principaux risques ou points de vigilance.\n\n"
        f"Données de l'action :\n{stock_json}"
    )


def _extract_grounding_sources(response: dict[str, Any]) -> list[str]:
    """Extrait et déduplique les sources web citées dans groundingMetadata."""
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return []
    metadata = candidates[0].get("groundingMetadata")
    if not isinstance(metadata, dict):
        return []
    chunks = metadata.get("groundingChunks")
    if not isinstance(chunks, list):
        return []

    sources: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        raw_uri = web.get("uri")
        if not isinstance(raw_uri, str) or not raw_uri.strip():
            continue
        uri = raw_uri.strip()
        raw_title = web.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else None
        entry = f"{title} ({uri})" if title else uri
        if entry not in seen:
            seen.add(entry)
            sources.append(entry)
    return sources


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
        clock: Callable[[], datetime] = _utc_now,
        max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES,
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
        if isinstance(max_validation_retries, bool) or max_validation_retries < 0:
            raise EvaluationError(
                "max_validation_retries doit être un entier supérieur ou égal à 0."
            )

        self._model = cleaned_model
        self._http_client = http_client
        self._api_base_url = cleaned_base_url
        self._timeout = timeout
        self._clock = clock
        self._max_validation_retries = max_validation_retries

    def evaluate(self, stock: StockData) -> StockEvaluation:
        url = f"{self._api_base_url}/models/{self._model}:generateContent"
        headers = {"x-goog-api-key": self._api_key}
        analysis_date = self._clock().date().isoformat()
        base_prompt = _build_evaluation_prompt(stock, analysis_date=analysis_date)

        last_validation_note: str | None = None
        for attempt in range(self._max_validation_retries + 1):
            prompt = base_prompt
            if last_validation_note is not None:
                prompt = (
                    f"{base_prompt}\n\n"
                    "ATTENTION : ta réponse précédente a échoué à la validation "
                    f"({last_validation_note}). Respecte strictement le schéma JSON et "
                    "la correspondance exacte 0-3 => VENDRE, 4-6 => CONSERVER, 7-10 => ACHAT."
                )

            payload: dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "tools": [{"google_search": {}}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": _EVALUATION_RESPONSE_SCHEMA,
                },
            }

            logger.info(
                "Évaluation Gemini de %s via %s (tentative %d/%d).",
                stock.symbol,
                self._model,
                attempt + 1,
                self._max_validation_retries + 1,
            )
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
            sources = _extract_grounding_sources(response)

            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                if attempt < self._max_validation_retries:
                    last_validation_note = f"JSON invalide : {exc}"
                    logger.warning(
                        "Réponse JSON Gemini invalide pour %s (tentative %d/%d), nouvelle tentative.",
                        stock.symbol,
                        attempt + 1,
                        self._max_validation_retries + 1,
                    )
                    continue
                raise EvaluationError(
                    f"Réponse JSON invalide reçue de Gemini pour {stock.symbol} : {exc}"
                ) from exc

            if not isinstance(parsed, dict):
                if attempt < self._max_validation_retries:
                    last_validation_note = f"objet JSON attendu, reçu {type(parsed).__name__}"
                    logger.warning(
                        "Réponse JSON Gemini non-objet pour %s (tentative %d/%d), nouvelle tentative.",
                        stock.symbol,
                        attempt + 1,
                        self._max_validation_retries + 1,
                    )
                    continue
                raise EvaluationError(
                    f"La réponse JSON de Gemini pour {stock.symbol} doit être un objet."
                )

            evaluation_data: dict[str, Any] = {
                **parsed,
                "symbol": stock.symbol,
                "model": self._model,
                "sources": sources,
            }
            try:
                return StockEvaluation.model_validate(evaluation_data)
            except ValidationError as exc:
                if attempt < self._max_validation_retries:
                    first_err = exc.errors()[0].get("msg", str(exc)) if exc.errors() else str(exc)
                    last_validation_note = first_err
                    logger.warning(
                        "Évaluation Gemini invalide/incohérente pour %s (tentative %d/%d) : %s — nouvelle tentative.",
                        stock.symbol,
                        attempt + 1,
                        self._max_validation_retries + 1,
                        first_err,
                    )
                    continue
                raise EvaluationError(
                    f"Évaluation invalide reçue de Gemini pour {stock.symbol} : {exc}"
                ) from exc

        raise EvaluationError(f"Échec inattendu lors de l'évaluation Gemini pour {stock.symbol}.")
