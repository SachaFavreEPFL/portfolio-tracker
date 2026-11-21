"""Boucle d'exécution agentique avec garde-fous et journalisation JSONL."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error

from harness.providers import LLMProvider, ProviderError
from harness.roles import build_system_prompt, get_role
from harness.tools import ToolBox

_MUTATING_ROLES = frozenset({"coder", "test", "doc"})


@dataclass
class _RunBudget:
    """Budget partagé d'appels LLM et de délégations pour une exécution et ses sous-agents."""

    remaining_steps: int
    remaining_delegations: int


def _loop_signature(
    call_name: str,
    arguments: dict[str, Any],
    tool_output: str,
    mutation_epoch: int,
) -> tuple[str, str, str]:
    """Calcule une signature stable pour la détection de boucle, y compris sur sorties variables."""
    serialized_args = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    if call_name == "run_checks":
        return (call_name, serialized_args, f"epoch:{mutation_epoch}")
    if call_name == "delegate_to_agent":
        return (call_name, serialized_args, "delegate")
    return (call_name, serialized_args, tool_output)


@dataclass(frozen=True)
class RunResult:
    """Bilan structuré d'une exécution d'agent."""

    role: str
    model: str
    output: str
    status: str
    steps: int
    tool_calls: list[str]
    duration_s: float


def _append_run_log(log_path: Path, prompt: str, result: RunResult) -> None:
    """Enregistre une ligne JSONL pour l'observabilité sans bloquer en cas d'erreur disque."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": result.role,
        "model": result.model,
        "prompt": prompt,
        "status": result.status,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "duration_s": round(result.duration_s, 3),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def run_agent(
    prompt: str,
    role_name: str,
    provider: LLMProvider,
    root: Path,
    *,
    max_steps: int = 15,
    max_total_steps: int | None = None,
    max_delegations: int = 6,
    log_path: Path | None = None,
    verbose: bool = True,
    role_providers: dict[str, LLMProvider] | None = None,
    _budget: _RunBudget | None = None,
) -> RunResult:
    """Exécute la boucle outil -> résultat pour un rôle donné avec détection de boucle et budget."""
    started = time.monotonic()
    workspace = root.resolve()
    role = get_role(role_name)
    resolved_log = log_path if log_path is not None else workspace / "harness" / "logs" / "runs.jsonl"
    overrides = role_providers or {}
    budget = (
        _budget
        if _budget is not None
        else _RunBudget(
            remaining_steps=max_total_steps if max_total_steps is not None else max_steps * 3,
            remaining_delegations=max_delegations,
        )
    )

    def _delegate(target_role: str, task: str) -> str:
        if target_role == "orchestrator":
            return "Erreur : l'orchestrateur ne peut pas se déléguer à lui-même."
        if budget.remaining_delegations <= 0:
            return "Erreur : nombre maximal de délégations atteint pour cette exécution."
        if budget.remaining_steps <= 0:
            return "Erreur : budget global d'étapes épuisé, délégation impossible."
        budget.remaining_delegations -= 1
        sub_provider = overrides.get(target_role, provider)
        if verbose:
            print(
                f"[délégation -> {target_role} | modèle : {sub_provider.model}]",
                file=sys.stderr,
            )
        sub_result = run_agent(
            prompt=task,
            role_name=target_role,
            provider=sub_provider,
            root=workspace,
            max_steps=min(max_steps, budget.remaining_steps),
            log_path=resolved_log,
            verbose=verbose,
            role_providers=overrides,
            _budget=budget,
        )
        return (
            f"[{target_role} ({sub_provider.model}) | {sub_result.status} | {sub_result.steps} étapes]\n"
            f"{sub_result.output}"
        )

    toolbox = ToolBox(
        workspace,
        set(role.allowed_tools),
        allowed_write_prefixes=role.allowed_write_prefixes,
        delegate_fn=_delegate if "delegate_to_agent" in role.allowed_tools else None,
    )
    system_prompt = build_system_prompt(role, workspace)
    session = provider.start_session(system_prompt, prompt, toolbox.schemas())

    called_tools: list[str] = []
    signature_counts: dict[tuple[str, str, str], int] = {}
    mutation_epoch = 0
    executed_steps = 0

    try:
        for step in range(1, max_steps + 1):
            if budget.remaining_steps <= 0:
                result = RunResult(
                    role=role.name,
                    model=provider.model,
                    output="Arrêt de sécurité : budget global d'étapes épuisé.",
                    status="budget_exhausted",
                    steps=executed_steps,
                    tool_calls=called_tools,
                    duration_s=time.monotonic() - started,
                )
                _append_run_log(resolved_log, prompt, result)
                return result

            budget.remaining_steps -= 1
            executed_steps = step
            response = session.next_turn()
            if not response.tool_calls:
                result = RunResult(
                    role=role.name,
                    model=provider.model,
                    output=response.text,
                    status="completed",
                    steps=step,
                    tool_calls=called_tools,
                    duration_s=time.monotonic() - started,
                )
                _append_run_log(resolved_log, prompt, result)
                return result

            for call in response.tool_calls:
                called_tools.append(call.name)
                if verbose:
                    print(
                        f"[{role.name} étape {step}] {call.name}({call.arguments})",
                        file=sys.stderr,
                    )
                tool_output = toolbox.execute(call.name, call.arguments)

                if call.name in {"edit_file", "write_file"} and "succès" in tool_output:
                    mutation_epoch += 1
                elif (
                    call.name == "delegate_to_agent"
                    and call.arguments.get("agent_name") in _MUTATING_ROLES
                    and not tool_output.startswith("Erreur")
                ):
                    mutation_epoch += 1

                # Garde-fou anti-boucle : bloque après 3 occurrences identiques (même alternées).
                signature = _loop_signature(call.name, call.arguments, tool_output, mutation_epoch)
                signature_counts[signature] = signature_counts.get(signature, 0) + 1

                if signature_counts[signature] >= 3:
                    result = RunResult(
                        role=role.name,
                        model=provider.model,
                        output=(
                            f"Arrêt de sécurité : boucle détectée sur '{call.name}' "
                            f"(appel répété {signature_counts[signature]} fois sans progrès)."
                            f"\nDernière sortie : {tool_output}"
                        ),
                        status="loop_detected",
                        steps=step,
                        tool_calls=called_tools,
                        duration_s=time.monotonic() - started,
                    )
                    _append_run_log(resolved_log, prompt, result)
                    return result

                session.add_tool_result(call, tool_output)
    except (ProviderError, urllib.error.URLError, TimeoutError) as exc:
        error_result = RunResult(
            role=role.name,
            model=provider.model,
            output=str(exc),
            status="error",
            steps=executed_steps,
            tool_calls=called_tools,
            duration_s=time.monotonic() - started,
        )
        _append_run_log(resolved_log, prompt, error_result)
        if isinstance(exc, ProviderError):
            raise
        raise ProviderError(
            f"Erreur fournisseur ({provider.model}) pour '{role.name}' : {exc}"
        ) from exc

    result = RunResult(
        role=role.name,
        model=provider.model,
        output=f"Nombre maximal d'étapes atteint ({max_steps}) sans réponse finale.",
        status="max_steps",
        steps=executed_steps,
        tool_calls=called_tools,
        duration_s=time.monotonic() - started,
    )
    _append_run_log(resolved_log, prompt, result)
    return result
