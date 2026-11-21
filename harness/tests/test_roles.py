"""Tests unitaires pour harness.roles."""

from pathlib import Path

import pytest

from harness.roles import ROLES, build_system_prompt, get_role


def test_get_role_returns_spec_and_rejects_unknown() -> None:
    role = get_role("audit")
    assert role.name == "audit"
    assert "write_file" not in role.allowed_tools
    assert "edit_file" not in role.allowed_tools

    with pytest.raises(ValueError, match="Rôle inconnu"):
        get_role("inexistant")


def test_role_guardrails_are_strict() -> None:
    assert ROLES["test"].allowed_write_prefixes == ("tests/",)
    assert ROLES["coder"].allowed_write_prefixes == ("src/", "scripts/")
    assert "write_file" not in ROLES["doc"].allowed_tools
    assert "delegate_to_agent" in ROLES["orchestrator"].allowed_tools
    assert "write_file" not in ROLES["orchestrator"].allowed_tools


def test_build_system_prompt_injects_agents_md_when_present(tmp_path: Path) -> None:
    role = get_role("coder")

    without_md = build_system_prompt(role, tmp_path)
    assert without_md == role.system_prompt

    (tmp_path / "AGENTS.md").write_text("# Règles\nUtiliser pytest.\n", encoding="utf-8")
    with_md = build_system_prompt(role, tmp_path)
    assert role.system_prompt in with_md
    assert "Utiliser pytest." in with_md
