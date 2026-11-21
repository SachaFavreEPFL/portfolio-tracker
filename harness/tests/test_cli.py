"""Tests unitaires pour harness.cli (parsing --role-model, codes de retour et erreurs)."""

from unittest.mock import MagicMock

import pytest

from harness import cli
from harness.providers import ProviderError
from harness.runner import RunResult


def test_parse_role_models_parses_valid_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    parsed = cli._parse_role_models(["coder=gemini-4-hydrogen", "audit=gemini-4-neon"])

    assert set(parsed) == {"coder", "audit"}
    assert parsed["coder"].model == "gemini-4-hydrogen"
    assert parsed["audit"].model == "gemini-4-neon"


@pytest.mark.parametrize(
    "invalid_entry",
    [
        "coder_sans_egal",
        "unknown_role=gemini-4-neon",
        "coder=",
        "coder=   ",
    ],
)
def test_parse_role_models_rejects_invalid_entries(invalid_entry: str) -> None:
    with pytest.raises(ValueError):
        cli._parse_role_models([invalid_entry])


def test_parse_role_models_rejects_non_gemini_model() -> None:
    with pytest.raises(ProviderError, match="Modèle refusé"):
        cli._parse_role_models(["coder=qwen3.8"])


def test_cli_main_returns_zero_on_completed_and_one_on_non_completed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_provider = MagicMock()
    fake_provider.model = "gemini-4-neon"
    monkeypatch.setattr(cli, "create_provider", lambda *args, **kwargs: fake_provider)

    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda *args, **kwargs: RunResult(
            role="audit",
            model="gemini-4-neon",
            output="VALIDÉ",
            status="completed",
            steps=1,
            tool_calls=[],
            duration_s=0.1,
        ),
    )
    assert cli.main(["--agent", "audit", "Vérifie le code"]) == 0
    assert "VALIDÉ" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda *args, **kwargs: RunResult(
            role="audit",
            model="gemini-4-neon",
            output="Boucle détectée",
            status="loop_detected",
            steps=3,
            tool_calls=["read_file"],
            duration_s=0.2,
        ),
    )
    assert cli.main(["--agent", "audit", "Vérifie le code"]) == 1


def test_cli_main_handles_provider_error_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_create_provider(*args: object, **kwargs: object) -> object:
        raise ProviderError("Clé API manquante")

    monkeypatch.setattr(cli, "create_provider", failing_create_provider)

    assert cli.main(["--agent", "audit", "Vérifie"]) == 1
    assert "Clé API manquante" in capsys.readouterr().err
