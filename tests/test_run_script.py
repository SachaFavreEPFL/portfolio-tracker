"""Tests unitaires pour scripts/run.py (orchestration CLI et gestion d'erreurs)."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from portfolio_tracker.contracts import StockData, StockEvaluation
from portfolio_tracker.evaluator import EvaluationError
from portfolio_tracker.market_data import MarketDataError
from scripts import run as run_script


def _sample_stock(symbol: str = "NVDA") -> StockData:
    return StockData(symbol=symbol, name=f"{symbol} Corp", currency="USD", price=120.0)


def _sample_evaluation(symbol: str = "NVDA") -> StockEvaluation:
    return StockEvaluation(
        symbol=symbol,
        verdict="ACHAT",
        score=8,
        summary="Fondamentaux solides.",
        strengths=["Croissance"],
        risks=["Valorisation"],
        model="gemini-4-hydrogen",
    )


def test_run_main_fetches_and_evaluates_requested_symbols(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = MagicMock()
    provider.get.return_value = _sample_stock("AAPL")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _sample_evaluation("AAPL")

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", lambda: evaluator)
    monkeypatch.setattr(sys, "argv", ["run.py", "AAPL"])

    exit_code = run_script.main()

    assert exit_code == 0
    provider.get.assert_called_once_with("AAPL")
    evaluator.evaluate.assert_called_once()
    stdout = capsys.readouterr().out
    assert '"symbol": "AAPL"' in stdout
    assert '"verdict": "ACHAT"' in stdout


def test_run_main_supports_force_refresh_and_no_eval_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    provider.get.return_value = _sample_stock("NVDA")
    evaluator_ctor = MagicMock()

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", evaluator_ctor)

    exit_code = run_script.main(["--force-refresh", "--no-eval", "NVDA"])

    assert exit_code == 0
    provider.get.assert_called_once_with("NVDA", force_refresh=True)
    evaluator_ctor.assert_not_called()


def test_run_main_continues_without_gemini_when_evaluator_init_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = MagicMock()
    provider.get.return_value = _sample_stock("NVDA")

    def failing_evaluator() -> object:
        raise EvaluationError("GEMINI_API_KEY manquante")

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", failing_evaluator)
    monkeypatch.setattr(sys, "argv", ["run.py", "NVDA"])

    exit_code = run_script.main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Évaluation Gemini désactivée" in captured.err
    assert json.loads(captured.out.split("---\n", 1)[1])["symbol"] == "NVDA"


def test_run_main_handles_market_and_evaluation_errors_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = MagicMock()
    provider.get.side_effect = [MarketDataError("introuvable"), _sample_stock("META")]
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = EvaluationError("quota dépassé")

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", lambda: evaluator)
    monkeypatch.setattr(sys, "argv", ["run.py", "BAD", "META"])

    exit_code = run_script.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[ERREUR] introuvable" in captured.err
    assert "[ERREUR ÉVALUATION] quota dépassé" in captured.err
    assert '"symbol": "META"' in captured.out


def test_run_script_cli_help_succeeds_outside_venv_via_reexec() -> None:
    script_path = Path(run_script.__file__).resolve()
    # Utilise l'interpréteur système (hors .venv) s'il est distinct pour vérifier le ré-exec avant imports.
    base_python = Path(sys.base_prefix) / "bin" / "python3"
    interpreter = str(base_python) if base_python.exists() else sys.executable

    completed = subprocess.run(
        [interpreter, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--force-refresh" in completed.stdout
    assert "--no-eval" in completed.stdout
