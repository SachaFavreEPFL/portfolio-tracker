"""Tests unitaires pour harness.providers (sans appel réseau réel)."""

from unittest.mock import MagicMock

import pytest

from harness.providers import (
    GeminiProvider,
    OllamaProvider,
    ProviderError,
    ToolCall,
    create_provider,
    detect_default_ollama_model,
)

SAMPLE_TOOLS = [
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


def test_ollama_session_handles_tool_call_and_final_answer() -> None:
    http_client = MagicMock(
        side_effect=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "AGENTS.md"}',
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Tout est lu."}},
        ]
    )
    provider = OllamaProvider("qwen3.8", think=False, http_client=http_client)
    session = provider.start_session("sys", "user", SAMPLE_TOOLS)

    first = session.next_turn()
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].name == "read_file"
    assert first.tool_calls[0].arguments == {"path": "AGENTS.md"}

    session.add_tool_result(first.tool_calls[0], "# AGENTS")
    second = session.next_turn()
    assert second.text == "Tout est lu."
    assert second.tool_calls == []

    # Vérifie que think=False et tool_name sont bien transmis dans le payload Ollama.
    first_payload = http_client.call_args_list[0].args[1]
    assert first_payload["think"] is False
    second_payload = http_client.call_args_list[1].args[1]
    assert {
        "role": "tool",
        "tool_name": "read_file",
        "content": "# AGENTS",
    } in second_payload["messages"]


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


def test_gemini_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        GeminiProvider("gemini-4-hydrogen")


def test_create_provider_routes_by_model_prefix() -> None:
    http_client = MagicMock(return_value={"models": [{"name": "qwen3.8:latest"}]})

    cloud = create_provider("gemini-4-neon", api_key="k", http_client=http_client)
    assert isinstance(cloud, GeminiProvider)
    assert cloud.model == "gemini-4-neon"

    local = create_provider(None, http_client=http_client)
    assert isinstance(local, OllamaProvider)
    assert local.model == "qwen3.8:latest"


def test_detect_default_ollama_model_raises_when_empty() -> None:
    http_client = MagicMock(return_value={"models": []})
    with pytest.raises(ProviderError):
        detect_default_ollama_model(http_client=http_client)


def test_ollama_session_handles_malformed_json_tool_arguments() -> None:
    http_client = MagicMock(
        return_value={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": "{invalid json"}},
                    {"function": {"name": "list_files", "arguments": '["not", "a", "dict"]'}},
                ],
            }
        }
    )
    provider = OllamaProvider("qwen3.8", http_client=http_client)
    session = provider.start_session("sys", "user", SAMPLE_TOOLS)

    turn = session.next_turn()
    assert len(turn.tool_calls) == 2
    assert turn.tool_calls[0].arguments == {}
    assert turn.tool_calls[1].arguments == {}


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
