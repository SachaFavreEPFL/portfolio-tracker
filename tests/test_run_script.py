"""Tests unitaires pour scripts/run.py (orchestration CLI et gestion d'erreurs)."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from portfolio_tracker.contracts import (
    PortfolioRowResult,
    PortfolioUpdateResult,
    StockData,
    StockEvaluation,
)
from portfolio_tracker.evaluator import EvaluationError
from portfolio_tracker.market_data import MarketDataError
from portfolio_tracker.numbers_portfolio import NumbersPortfolioError
from scripts import run as run_script


def _sample_stock(symbol: str = "NVDA") -> StockData:
    return StockData(symbol=symbol, name=f"{symbol} Corp", currency="USD", price=120.0)


def _sample_evaluation(symbol: str = "NVDA") -> StockEvaluation:
    return StockEvaluation(
        symbol=symbol,
        verdict="ACHAT",
        score=80,
        comment="Croissance solide et marges robustes.",
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
    assert "--numbers-file" in completed.stdout
    assert "--output" in completed.stdout


# --- Mode Apple Numbers (--numbers-file / --output) ---


def test_run_main_numbers_mode_success_in_place_and_decision_column_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "suivi.numbers"
    provider = MagicMock()
    evaluator = MagicMock()
    captured: dict[str, object] = {}

    def fake_update(
        path: Path,
        *,
        provider: object,
        evaluator: object,
        output_path: Path | None = None,
        sheet_name: str | None = None,
        table_name: str | None = None,
        force_refresh: bool = False,
        on_row_processed: object = None,
    ) -> PortfolioUpdateResult:
        captured["path"] = path
        captured["provider"] = provider
        captured["evaluator"] = evaluator
        captured["output_path"] = output_path
        captured["sheet_name"] = sheet_name
        captured["table_name"] = table_name
        captured["force_refresh"] = force_refresh
        captured["has_progress_cb"] = callable(on_row_processed)
        return PortfolioUpdateResult(
            input_path=path,
            output_path=path,
            sheet_name="Portefeuille",
            table_name="Actions",
            decision_column_created=True,
            results=[
                PortfolioRowResult(
                    row_index=1,
                    symbol="UNM",
                    score=80,
                    comment="Rentabilité solide.",
                    decision="Achat",
                )
            ],
        )

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", lambda: evaluator)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", fake_update)

    exit_code = run_script.main(["--numbers-file", str(input_file)])

    assert exit_code == 0
    assert captured == {
        "path": input_file,
        "provider": provider,
        "evaluator": evaluator,
        "output_path": None,
        "sheet_name": None,
        "table_name": None,
        "force_refresh": False,
        "has_progress_cb": True,
    }
    out = capsys.readouterr().out
    assert f"Fichier sauvegardé : {input_file}" in out
    assert "Colonne « Décision » créée automatiquement." in out
    assert "[OK] UNM — Note : 80/100 | Décision : Achat | Rentabilité solide." in out
    assert "Total : 1 succès, 0 erreur(s)." in out


def test_run_main_numbers_mode_success_with_output_and_force_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "in.numbers"
    output_file = tmp_path / "out.numbers"
    provider = MagicMock()
    evaluator = MagicMock()
    update_mock = MagicMock(
        return_value=PortfolioUpdateResult(
            input_path=input_file,
            output_path=output_file,
            sheet_name="Feuille 1",
            table_name="Tableau 1",
            decision_column_created=False,
            results=[
                PortfolioRowResult(
                    row_index=1,
                    symbol="PEP",
                    score=50,
                    comment="Défensif mais valorisé.",
                    decision="Conserver",
                )
            ],
        )
    )

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", lambda: evaluator)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(
        ["--numbers-file", str(input_file), "--output", str(output_file), "--force-refresh"]
    )

    assert exit_code == 0
    assert update_mock.call_count == 1
    call_args, call_kwargs = update_mock.call_args
    assert call_args == (input_file,)
    assert call_kwargs["provider"] is provider
    assert call_kwargs["evaluator"] is evaluator
    assert call_kwargs["output_path"] == output_file
    assert call_kwargs["force_refresh"] is True
    assert callable(call_kwargs["on_row_processed"])
    out = capsys.readouterr().out
    assert f"Fichier sauvegardé : {output_file}" in out
    assert "créée automatiquement" not in out
    assert "[OK] PEP" in out
    assert "Total : 1 succès, 0 erreur(s)." in out


def test_run_main_numbers_mode_partial_row_error_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "partial.numbers"
    update_mock = MagicMock(
        return_value=PortfolioUpdateResult(
            input_path=input_file,
            output_path=input_file,
            sheet_name="Portefeuille",
            table_name="Actions",
            decision_column_created=False,
            results=[
                PortfolioRowResult(
                    row_index=1,
                    symbol="UNM",
                    score=80,
                    comment="Solide.",
                    decision="Achat",
                ),
                PortfolioRowResult(
                    row_index=2,
                    symbol="INCONNU",
                    error="Aucune donnée trouvée pour le symbole INCONNU.",
                ),
                PortfolioRowResult(
                    row_index=3,
                    symbol="MO",
                    error="Quota Gemini dépassé",
                ),
            ],
        )
    )

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", MagicMock)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", MagicMock)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(["--numbers-file", str(input_file)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[OK] UNM" in captured.out
    assert "Total : 1 succès, 2 erreur(s)." in captured.out
    assert "[ERREUR] INCONNU : Aucune donnée trouvée pour le symbole INCONNU." in captured.err
    assert "[ERREUR] MO : Quota Gemini dépassé" in captured.err


def test_run_main_numbers_mode_reports_unsaved_when_all_rows_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "all_fail.numbers"
    update_mock = MagicMock(
        return_value=PortfolioUpdateResult(
            input_path=input_file,
            output_path=input_file,
            sheet_name="Portefeuille",
            table_name="Actions",
            decision_column_created=False,
            saved=False,
            results=[
                PortfolioRowResult(
                    row_index=1,
                    symbol="UNM",
                    error="Panne réseau",
                ),
            ],
        )
    )

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", MagicMock)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", MagicMock)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(["--numbers-file", str(input_file)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert f"Aucune ligne mise à jour ; fichier non modifié : {input_file}" in captured.out
    assert "Fichier sauvegardé" not in captured.out
    assert "[ERREUR] UNM : Panne réseau" in captured.err


def test_run_main_numbers_mode_handles_numbers_portfolio_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "missing.numbers"
    update_mock = MagicMock(
        side_effect=NumbersPortfolioError("Colonne d'en-tête « Note » introuvable.")
    )

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", MagicMock)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", MagicMock)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(["--numbers-file", str(input_file)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[ERREUR] Colonne d'en-tête « Note » introuvable." in captured.err
    assert captured.out == ""


def test_run_main_numbers_mode_fails_when_gemini_evaluator_init_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "suivi.numbers"
    update_mock = MagicMock()

    def failing_evaluator() -> object:
        raise EvaluationError("GEMINI_API_KEY manquante")

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", MagicMock)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", failing_evaluator)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(["--numbers-file", str(input_file)])

    assert exit_code == 1
    update_mock.assert_not_called()
    captured = capsys.readouterr()
    assert "[ERREUR] Évaluation Gemini indisponible : GEMINI_API_KEY manquante" in captured.err


def test_run_main_numbers_mode_rejects_no_eval_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider_ctor = MagicMock()
    evaluator_ctor = MagicMock()
    update_mock = MagicMock()

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", provider_ctor)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", evaluator_ctor)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(["--numbers-file", str(tmp_path / "suivi.numbers"), "--no-eval"])

    assert exit_code == 1
    provider_ctor.assert_not_called()
    evaluator_ctor.assert_not_called()
    update_mock.assert_not_called()
    assert "incompatible avec --numbers-file" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected_error_fragment"),
    [
        (["--output", "out.numbers"], "--output ne peut être utilisé qu'avec --numbers-file"),
        (
            ["--numbers-file", "suivi.numbers", "NVDA"],
            "--numbers-file ne peut pas être combiné avec des symboles positionnels",
        ),
    ],
)
def test_run_main_rejects_invalid_cli_option_combinations(
    argv: list[str],
    expected_error_fragment: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_script.main(argv)

    assert exc_info.value.code == 2
    assert expected_error_fragment in capsys.readouterr().err


def test_run_main_numbers_mode_handles_keyboard_interrupt_with_exit_130(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_file = tmp_path / "suivi.numbers"
    update_mock = MagicMock(side_effect=KeyboardInterrupt())

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", MagicMock)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", MagicMock)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(["--numbers-file", str(input_file)])

    assert exit_code == 130
    assert "[INTERROMPU]" in capsys.readouterr().err


def test_run_main_numbers_mode_end_to_end_with_real_numbers_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from numbers_parser import Document

    numbers_file = tmp_path / "portefeuille.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=3, num_cols=3)
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Action")
    table.write(0, 1, "Note")
    table.write(0, 2, "Commentaire")
    table.write(1, 0, "NVDA")
    table.write(2, 0, "Total")
    doc.save(str(numbers_file))

    provider = MagicMock()
    provider.get.return_value = _sample_stock("NVDA")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _sample_evaluation("NVDA")

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", lambda: evaluator)

    exit_code = run_script.main(["--numbers-file", str(numbers_file)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[1/1] NVDA — Note : 80/100 | Achat" in captured.err
    assert f"Fichier sauvegardé : {numbers_file}" in captured.out
    assert "Total : 1 succès, 0 erreur(s)." in captured.out

    reloaded = Document(str(numbers_file))
    out_table = reloaded.sheets[0].tables[0]
    assert out_table.cell(0, 3).value == "Décision"
    assert out_table.cell(1, 1).value == 80
    assert out_table.cell(1, 2).value == "Croissance solide et marges robustes."
    assert out_table.cell(1, 3).value == "Achat"
    assert out_table.cell(2, 1).value is None


def test_run_main_passes_sheet_table_and_model_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "suivi.numbers"
    provider = MagicMock()
    evaluator = MagicMock()
    evaluator_ctor = MagicMock(return_value=evaluator)
    update_mock = MagicMock(
        return_value=PortfolioUpdateResult(
            input_path=input_file,
            output_path=input_file,
            sheet_name="PEA",
            table_name="Titres",
            results=[],
            saved=False,
        )
    )

    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-ignored")
    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", evaluator_ctor)
    monkeypatch.setattr(run_script, "update_numbers_portfolio", update_mock)

    exit_code = run_script.main(
        [
            "--numbers-file",
            str(input_file),
            "--sheet",
            "PEA",
            "--table",
            "Titres",
            "--model",
            "gemini-4-neon",
        ]
    )

    assert exit_code == 0
    evaluator_ctor.assert_called_once_with(model="gemini-4-neon")
    _, call_kwargs = update_mock.call_args
    assert call_kwargs["sheet_name"] == "PEA"
    assert call_kwargs["table_name"] == "Titres"


def test_run_main_uses_gemini_model_env_var_when_cli_model_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    provider.get.return_value = _sample_stock("NVDA")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _sample_evaluation("NVDA")
    evaluator_ctor = MagicMock(return_value=evaluator)

    monkeypatch.setenv("GEMINI_MODEL", "gemini-4-neon")
    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", evaluator_ctor)

    exit_code = run_script.main(["NVDA"])

    assert exit_code == 0
    evaluator_ctor.assert_called_once_with(model="gemini-4-neon")


@pytest.mark.parametrize(
    "argv",
    [
        ["--sheet", "Feuille 1", "NVDA"],
        ["--table", "Tableau 1", "NVDA"],
    ],
)
def test_run_main_rejects_sheet_or_table_without_numbers_file(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_script.main(argv)

    assert exc_info.value.code == 2
    assert "--sheet et --table" in capsys.readouterr().err


def test_run_main_positional_mode_handles_keyboard_interrupt_with_exit_130(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = MagicMock()
    provider.get.side_effect = KeyboardInterrupt()

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)
    monkeypatch.setattr(run_script, "GeminiStockEvaluator", lambda: MagicMock())

    exit_code = run_script.main(["NVDA"])

    assert exit_code == 130
    assert "[INTERROMPU]" in capsys.readouterr().err


def test_run_main_configures_logging_idempotently_and_supports_verbose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging

    provider = MagicMock()
    provider.get.return_value = _sample_stock("NVDA")

    monkeypatch.setattr(run_script, "YFinanceMarketDataProvider", lambda: provider)

    assert run_script.main(["--no-eval", "NVDA"]) == 0
    pkg_logger = logging.getLogger("portfolio_tracker")
    assert pkg_logger.level == logging.WARNING

    assert run_script.main(["--no-eval", "--verbose", "NVDA"]) == 0
    assert pkg_logger.level == logging.DEBUG
    cli_handlers = [h for h in pkg_logger.handlers if isinstance(h, run_script._CliStreamHandler)]
    assert len(cli_handlers) == 1
