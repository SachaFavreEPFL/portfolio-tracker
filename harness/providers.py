"""Connecteur LLM du harness via l'API Gemini Interactions."""

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness._http import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    read_env_file_key,
    request_json,
    sanitize_error_message,
)

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-neon")
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

HttpJsonFn = Callable[
    [str, dict[str, Any] | None, dict[str, str] | None, int],
    dict[str, Any],
]


class ProviderError(Exception):
    """Erreur de configuration ou de communication avec un fournisseur LLM."""


@dataclass(frozen=True)
class ToolCall:
    """Représentation unifiée d'un appel d'outil demandé par un modèle."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ProviderResponse:
    """Réponse unifiée d'un tour de conversation LLM."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatSession(Protocol):
    """Session multi-tours agnostique du fournisseur LLM."""

    def next_turn(self) -> ProviderResponse: ...

    def add_tool_result(self, call: ToolCall, result: str) -> None: ...


class LLMProvider(Protocol):
    """Contrat d'interface pour tout fournisseur LLM."""

    model: str

    def start_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
    ) -> ChatSession: ...


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 300,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Envoie une requête HTTP GET ou POST JSON avec retry sur erreurs transitoires."""
    return request_json(
        url,
        payload=payload,
        extra_headers=extra_headers,
        timeout=timeout,
        error_cls=ProviderError,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        sleep=sleep,
        logger_instance=logger,
    )


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    """Normalise les arguments d'appel d'outil en dictionnaire sans planter sur du JSON invalide."""
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return raw_args if isinstance(raw_args, dict) else {}


def _to_gemini_tools(tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convertit les schémas d'outils au format Gemini Interactions."""
    return [{"type": "function", **tool["function"]} for tool in tools_schema]


def _extract_gemini_text(steps: list[dict[str, Any]]) -> str:
    """Extrait le texte final des étapes Gemini en ignorant les blocs de réflexion."""
    parts: list[str] = []
    for step in steps:
        if step.get("type") == "thought":
            continue
        if isinstance(step.get("text"), str):
            parts.append(step["text"])
            continue
        content = step.get("content")
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    and item["text"]
                ):
                    parts.append(item["text"])
    return "\n".join(parts).strip()


class _GeminiSession:
    """Session multi-tours pour l'API Gemini Interactions."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_url: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        http_client: HttpJsonFn,
    ) -> None:
        self._model = model
        self._headers = {"x-goog-api-key": api_key}
        self._api_url = api_url
        self._tools = _to_gemini_tools(tools)
        self._http = http_client
        initial_text = f"[INSTRUCTIONS SYSTÈME]\n{system_prompt}\n\n[DEMANDE]\n{user_prompt}"
        self._history: list[dict[str, Any]] = [
            {"type": "user_input", "content": [{"type": "text", "text": initial_text}]}
        ]

    def next_turn(self) -> ProviderResponse:
        response = self._http(
            self._api_url,
            {
                "model": self._model,
                "store": False,
                "input": self._history,
                "tools": self._tools,
            },
            self._headers,
            300,
        )
        if not isinstance(response, dict):
            raise ProviderError(
                f"Format de réponse Gemini inattendu : {type(response).__name__}."
            )
        if "error" in response and response["error"] is not None:
            error_payload = response["error"]
            if isinstance(error_payload, dict) and error_payload.get("message"):
                message = str(error_payload["message"])
            else:
                message = str(error_payload)
            sanitized = sanitize_error_message(message, self._headers.get("x-goog-api-key"))
            raise ProviderError(f"Erreur renvoyée par l'API Gemini : {sanitized}")

        raw_steps = response.get("steps")
        if raw_steps is None:
            steps: list[dict[str, Any]] = []
        elif isinstance(raw_steps, list):
            steps = [step for step in raw_steps if isinstance(step, dict)]
        else:
            raise ProviderError(
                f"Champ 'steps' Gemini invalide : {type(raw_steps).__name__}."
            )
        self._history.extend(steps)

        tool_calls = [
            ToolCall(
                id=str(step.get("id", "")),
                name=str(step.get("name", "")),
                arguments=_parse_tool_arguments(step.get("arguments")),
            )
            for step in steps
            if step.get("type") == "function_call"
        ]
        return ProviderResponse(
            text="" if tool_calls else _extract_gemini_text(steps),
            tool_calls=tool_calls,
        )

    def add_tool_result(self, call: ToolCall, result: str) -> None:
        self._history.append(
            {
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text", "text": result}],
            }
        )


def _read_env_file_key(env_path: Path, key_name: str) -> str:
    """Lit une variable dans un fichier .env sans modifier os.environ."""
    return read_env_file_key(env_path, key_name, error_cls=ProviderError) or ""


class GeminiProvider:
    """Fournisseur LLM cloud via l'API Gemini Interactions."""

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        api_key: str | None = None,
        api_url: str = DEFAULT_GEMINI_URL,
        http_client: HttpJsonFn = http_json,
        env_path: Path = DEFAULT_ENV_PATH,
    ) -> None:
        if api_key is not None:
            resolved_key = api_key.strip()
        else:
            env_key = os.environ.get("GEMINI_API_KEY", "").strip()
            resolved_key = env_key or _read_env_file_key(env_path, "GEMINI_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "Variable d'environnement GEMINI_API_KEY manquante pour utiliser un modèle Gemini."
            )
        self.model = model
        self._api_key = resolved_key
        self._api_url = api_url
        self._http = http_client

    def start_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
    ) -> ChatSession:
        return _GeminiSession(
            model=self.model,
            api_key=self._api_key,
            api_url=self._api_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            http_client=self._http,
        )


def create_provider(
    model: str | None = None,
    *,
    api_key: str | None = None,
    http_client: HttpJsonFn = http_json,
) -> LLMProvider:
    """Instancie un GeminiProvider en vérifiant que le modèle est bien un modèle Gemini."""
    selected = (model or os.environ.get("GEMINI_MODEL", "") or DEFAULT_GEMINI_MODEL).strip()
    if not selected.startswith("gemini-"):
        raise ProviderError(
            f"Modèle refusé : '{selected}'. Seuls les modèles Gemini (préfixe 'gemini-') "
            "sont autorisés."
        )
    return GeminiProvider(model=selected, api_key=api_key, http_client=http_client)
