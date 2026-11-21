"""Connecteurs LLM unifiés (Ollama local et Gemini Interactions API)."""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable, Protocol
import urllib.request

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen")

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
    """Contrat d'interface pour tout fournisseur LLM (Ollama, Gemini, etc.)."""

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
) -> dict[str, Any]:
    """Envoie une requête HTTP GET ou POST JSON et décode la réponse."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    """Normalise les arguments d'appel d'outil en dictionnaire sans planter sur du JSON invalide."""
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return raw_args if isinstance(raw_args, dict) else {}


def detect_default_ollama_model(
    base_url: str = DEFAULT_OLLAMA_URL,
    http_client: HttpJsonFn = http_json,
) -> str:
    """Retourne qwen3.8 si présent dans Ollama, sinon le premier modèle installé."""
    tags = http_client(f"{base_url.rstrip('/')}/api/tags", None, None, 10)
    models = [m["name"] for m in tags.get("models", []) if isinstance(m, dict) and "name" in m]
    if not models:
        raise ProviderError(
            "Aucun modèle Ollama installé. Lancez par exemple : ollama pull qwen3.8"
        )
    for preferred in ("qwen3.8:latest", "qwen3.8"):
        if preferred in models:
            return preferred
    return models[0]


class _OllamaSession:
    """Session multi-tours pour l'API /api/chat d'Ollama."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        think: bool,
        http_client: HttpJsonFn,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._tools = tools
        self._think = think
        self._http = http_client
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def next_turn(self) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "tools": self._tools,
            "stream": False,
        }
        if not self._think:
            payload["think"] = False

        response = self._http(f"{self._base_url}/api/chat", payload, None, 300)
        message = response.get("message", {})
        self._messages.append(message)

        tool_calls: list[ToolCall] = []
        for idx, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function", {})
            # Certains modèles Ollama renvoient les arguments sous forme de chaîne JSON.
            args = _parse_tool_arguments(fn.get("arguments"))
            tool_calls.append(
                ToolCall(
                    id=str(call.get("id") or f"ollama-{idx}"),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        return ProviderResponse(
            text=str(message.get("content", "")).strip(),
            tool_calls=tool_calls,
        )

    def add_tool_result(self, call: ToolCall, result: str) -> None:
        self._messages.append({"role": "tool", "tool_name": call.name, "content": result})


class OllamaProvider:
    """Fournisseur LLM local via Ollama."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        think: bool = True,
        http_client: HttpJsonFn = http_json,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._think = think
        self._http = http_client

    def start_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
    ) -> ChatSession:
        return _OllamaSession(
            model=self.model,
            base_url=self._base_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            think=self._think,
            http_client=self._http,
        )


def _to_gemini_tools(tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convertit les schémas d'outils OpenAI/Ollama au format Gemini Interactions."""
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
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
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
        steps = response.get("steps", [])
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


class GeminiProvider:
    """Fournisseur LLM cloud via l'API Gemini Interactions."""

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        api_key: str | None = None,
        api_url: str = DEFAULT_GEMINI_URL,
        http_client: HttpJsonFn = http_json,
    ) -> None:
        resolved_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
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
    ollama_url: str = DEFAULT_OLLAMA_URL,
    api_key: str | None = None,
    think: bool = True,
    http_client: HttpJsonFn = http_json,
) -> LLMProvider:
    """Instancie automatiquement OllamaProvider ou GeminiProvider selon le nom du modèle."""
    selected = (model or os.environ.get("OLLAMA_MODEL", "")).strip()
    if selected.startswith("gemini-"):
        return GeminiProvider(model=selected, api_key=api_key, http_client=http_client)
    ollama_model = selected or detect_default_ollama_model(ollama_url, http_client=http_client)
    return OllamaProvider(
        model=ollama_model,
        base_url=ollama_url,
        think=think,
        http_client=http_client,
    )
