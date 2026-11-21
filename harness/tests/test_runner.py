"""Tests unitaires pour harness.runner (boucle, garde-fou anti-boucle, délégation, logs)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from harness.providers import ProviderError, ProviderResponse, ToolCall
from harness.runner import run_agent


class _FakeSession:
    def __init__(self, turns: list[ProviderResponse]) -> None:
        self._turns = list(turns)
        self.tool_results: list[tuple[ToolCall, str]] = []

    def next_turn(self) -> ProviderResponse:
        return self._turns.pop(0)

    def add_tool_result(self, call: ToolCall, result: str) -> None:
        self.tool_results.append((call, result))


class _FakeProvider:
    def __init__(self, sessions_turns: list[list[ProviderResponse]], model: str = "fake-model") -> None:
        self.model = model
        self._sessions_turns = list(sessions_turns)
        self.sessions: list[_FakeSession] = []

    def start_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
    ) -> _FakeSession:
        session = _FakeSession(self._sessions_turns.pop(0))
        self.sessions.append(session)
        return session


def test_run_agent_completes_and_writes_jsonl_log(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("bonjour\n", encoding="utf-8")
    log_file = tmp_path / "logs" / "runs.jsonl"
    provider = _FakeProvider(
        [
            [
                ProviderResponse(
                    text="",
                    tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "notes.txt"})],
                ),
                ProviderResponse(text="VALIDÉ : fichier lu."),
            ]
        ]
    )

    result = run_agent(
        "Audite notes.txt",
        role_name="audit",
        provider=provider,
        root=tmp_path,
        log_path=log_file,
        verbose=False,
    )

    assert result.status == "completed"
    assert result.output == "VALIDÉ : fichier lu."
    assert result.steps == 2
    assert result.tool_calls == ["read_file"]
    assert provider.sessions[0].tool_results[0][1] == "1: bonjour"

    logged = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert logged["role"] == "audit"
    assert logged["status"] == "completed"
    assert logged["tool_calls"] == ["read_file"]


def test_run_agent_stops_on_repeated_identical_tool_call(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"
    repeated_call = ToolCall(id="1", name="read_file", arguments={"path": "missing.txt"})
    provider = _FakeProvider(
        [
            [
                ProviderResponse(text="", tool_calls=[repeated_call]),
                ProviderResponse(text="", tool_calls=[repeated_call]),
                ProviderResponse(text="", tool_calls=[repeated_call]),
            ]
        ]
    )

    result = run_agent(
        "Lis missing.txt",
        role_name="audit",
        provider=provider,
        root=tmp_path,
        log_path=log_file,
        verbose=False,
    )

    assert result.status == "loop_detected"
    assert "boucle détectée" in result.output


def test_run_agent_stops_on_alternating_loop(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"
    call_a = ToolCall(id="a", name="read_file", arguments={"path": "missing_a.txt"})
    call_b = ToolCall(id="b", name="read_file", arguments={"path": "missing_b.txt"})
    provider = _FakeProvider(
        [
            [
                ProviderResponse(text="", tool_calls=[call_a]),
                ProviderResponse(text="", tool_calls=[call_b]),
                ProviderResponse(text="", tool_calls=[call_a]),
                ProviderResponse(text="", tool_calls=[call_b]),
                ProviderResponse(text="", tool_calls=[call_a]),
            ]
        ]
    )

    result = run_agent(
        "Alterne deux lectures en erreur",
        role_name="audit",
        provider=provider,
        root=tmp_path,
        log_path=log_file,
        verbose=False,
    )

    assert result.status == "loop_detected"
    assert "boucle détectée" in result.output
    assert result.steps == 5


def test_orchestrator_delegates_to_sub_agent(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"
    provider = _FakeProvider(
        [
            # Session 1 : orchestrator
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="d1",
                            name="delegate_to_agent",
                            arguments={"agent_name": "audit", "task": "Vérifie le projet"},
                        )
                    ],
                ),
                ProviderResponse(text="Orchestration terminée."),
            ],
            # Session 2 : sous-agent audit
            [
                ProviderResponse(text="VALIDÉ"),
            ],
        ]
    )

    result = run_agent(
        "Audite tout",
        role_name="orchestrator",
        provider=provider,
        root=tmp_path,
        log_path=log_file,
        verbose=False,
    )

    assert result.status == "completed"
    assert result.output == "Orchestration terminée."
    # Vérifie que le sous-agent et l'orchestrateur ont tous les deux été journalisés.
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["role"] == "audit"
    assert json.loads(lines[1])["role"] == "orchestrator"


def test_orchestrator_uses_role_specific_provider_when_configured(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"
    orchestrator_provider = _FakeProvider(
        [
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="d1",
                            name="delegate_to_agent",
                            arguments={"agent_name": "doc", "task": "Documente le code"},
                        )
                    ],
                ),
                ProviderResponse(text="Fait."),
            ]
        ],
        model="gemini-4-neon",
    )
    doc_provider = _FakeProvider(
        [[ProviderResponse(text="Doc ajoutée.")]],
        model="qwen3.8",
    )

    result = run_agent(
        "Documente",
        role_name="orchestrator",
        provider=orchestrator_provider,
        root=tmp_path,
        log_path=log_file,
        verbose=False,
        role_providers={"doc": doc_provider},
    )

    assert result.status == "completed"
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
    assert lines[0]["role"] == "doc"
    assert lines[0]["model"] == "qwen3.8"
    assert lines[1]["role"] == "orchestrator"
    assert lines[1]["model"] == "gemini-4-neon"


def test_run_agent_detects_run_checks_loop_with_variable_output(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"
    check_call = ToolCall(id="c1", name="run_checks", arguments={})
    provider = _FakeProvider(
        [
            [
                ProviderResponse(text="", tool_calls=[check_call]),
                ProviderResponse(text="", tool_calls=[check_call]),
                ProviderResponse(text="", tool_calls=[check_call]),
            ]
        ]
    )

    with patch(
        "harness.tools.ToolBox._run_checks",
        side_effect=["code=1 (in 0.14s)", "code=1 (in 0.15s)", "code=1 (in 0.16s)"],
    ):
        result = run_agent(
            "Vérifie",
            role_name="coder",
            provider=provider,
            root=tmp_path,
            log_path=log_file,
            verbose=False,
        )

    assert result.status == "loop_detected"
    assert result.steps == 3


def test_run_agent_allows_run_checks_after_successful_edits(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("v = 0\n", encoding="utf-8")
    log_file = tmp_path / "runs.jsonl"
    provider = _FakeProvider(
        [
            [
                ProviderResponse(text="", tool_calls=[ToolCall(id="1", name="run_checks", arguments={})]),
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="2",
                            name="edit_file",
                            arguments={"path": "src/app.py", "target": "v = 0", "replacement": "v = 1"},
                        )
                    ],
                ),
                ProviderResponse(text="", tool_calls=[ToolCall(id="3", name="run_checks", arguments={})]),
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="4",
                            name="edit_file",
                            arguments={"path": "src/app.py", "target": "v = 1", "replacement": "v = 2"},
                        )
                    ],
                ),
                ProviderResponse(text="", tool_calls=[ToolCall(id="5", name="run_checks", arguments={})]),
                ProviderResponse(text="Corrigé."),
            ]
        ]
    )

    with patch("harness.tools.ToolBox._run_checks", return_value="code=0"):
        result = run_agent(
            "Corrige app.py",
            role_name="coder",
            provider=provider,
            root=tmp_path,
            log_path=log_file,
            verbose=False,
        )

    assert result.status == "completed"
    assert result.output == "Corrigé."


def test_run_agent_detects_delegate_loop_with_variable_sub_output(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"
    delegate_call = ToolCall(
        id="d1",
        name="delegate_to_agent",
        arguments={"agent_name": "audit", "task": "Audite"},
    )
    provider = _FakeProvider(
        [
            [
                ProviderResponse(text="", tool_calls=[delegate_call]),
                ProviderResponse(text="", tool_calls=[delegate_call]),
                ProviderResponse(text="", tool_calls=[delegate_call]),
            ],
            [ProviderResponse(text="Réponse 1")],
            [ProviderResponse(text="Réponse 2")],
            [ProviderResponse(text="Réponse 3")],
        ]
    )

    result = run_agent(
        "Audite en boucle",
        role_name="orchestrator",
        provider=provider,
        root=tmp_path,
        log_path=log_file,
        verbose=False,
    )

    assert result.status == "loop_detected"
    assert result.steps == 3


def test_delegation_propagates_provider_error_and_logs_failure(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"

    class _FailingSession:
        def next_turn(self) -> ProviderResponse:
            raise ProviderError("HTTP 503 indisponible")

        def add_tool_result(self, call: ToolCall, result: str) -> None:
            pass

    class _FailingProvider:
        model = "gemini-4-neon"

        def start_session(
            self,
            system_prompt: str,
            user_prompt: str,
            tools: list[dict[str, Any]],
        ) -> _FailingSession:
            return _FailingSession()

    orchestrator_provider = _FakeProvider(
        [
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="d1",
                            name="delegate_to_agent",
                            arguments={"agent_name": "audit", "task": "Audite"},
                        )
                    ],
                ),
            ]
        ],
        model="qwen3.8",
    )

    with pytest.raises(ProviderError, match="HTTP 503"):
        run_agent(
            "Lance audit",
            role_name="orchestrator",
            provider=orchestrator_provider,
            root=tmp_path,
            log_path=log_file,
            verbose=False,
            role_providers={"audit": _FailingProvider()},
        )

    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
    assert [entry["role"] for entry in lines] == ["audit", "orchestrator"]
    assert [entry["status"] for entry in lines] == ["error", "error"]


def test_orchestrator_enforces_delegation_and_global_step_budgets(tmp_path: Path) -> None:
    log_file = tmp_path / "runs.jsonl"
    provider = _FakeProvider(
        [
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="d1",
                            name="delegate_to_agent",
                            arguments={"agent_name": "audit", "task": "Tâche 1"},
                        )
                    ],
                ),
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="d2",
                            name="delegate_to_agent",
                            arguments={"agent_name": "audit", "task": "Tâche 2"},
                        )
                    ],
                ),
                ProviderResponse(text="Terminé malgré limite."),
            ],
            [ProviderResponse(text="Sous-agent 1 OK")],
        ]
    )

    result = run_agent(
        "Deux délégations",
        role_name="orchestrator",
        provider=provider,
        root=tmp_path,
        max_delegations=1,
        max_total_steps=5,
        log_path=log_file,
        verbose=False,
    )

    assert result.status == "completed"
    assert "nombre maximal de délégations atteint" in provider.sessions[0].tool_results[1][1]

    # Vérifie maintenant l'épuisement du budget global d'étapes.
    budget_provider = _FakeProvider(
        [
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="d1",
                            name="delegate_to_agent",
                            arguments={"agent_name": "audit", "task": "Tâche 1"},
                        )
                    ],
                ),
                ProviderResponse(text="Ne devrait pas être atteint"),
            ],
            [ProviderResponse(text="Sous-agent 1 OK")],
        ]
    )

    exhausted = run_agent(
        "Budget serré",
        role_name="orchestrator",
        provider=budget_provider,
        root=tmp_path,
        max_total_steps=2,
        log_path=log_file,
        verbose=False,
    )
    assert exhausted.status == "budget_exhausted"
