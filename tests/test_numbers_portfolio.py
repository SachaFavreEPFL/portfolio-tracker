"""Tests unitaires isolés pour portfolio_tracker.numbers_portfolio (fichiers .numbers réels)."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from numbers_parser import Document

from portfolio_tracker.contracts import (
    MAX_COMMENT_LENGTH,
    PortfolioRow,
    StockData,
    StockEvaluation,
    Verdict,
)
from portfolio_tracker.evaluator import EvaluationError
from portfolio_tracker.market_data import MarketDataError, StockNotFoundError
from portfolio_tracker.numbers_portfolio import (
    DECISION_HEADER,
    NumbersPortfolioError,
    _save_document_atomically,
    read_portfolio_rows,
    update_numbers_portfolio,
)


def _create_numbers_file(
    path: Path,
    *,
    headers: list[str | None],
    rows: list[list[Any]],
    sheet_name: str = "Portefeuille",
    table_name: str = "Actions",
    num_cols: int | None = None,
    extra_sheet: tuple[str, str, list[list[Any]]] | None = None,
) -> Path:
    total_cols = num_cols if num_cols is not None else len(headers)
    total_rows = max(1 + len(rows), 2)
    doc = Document(
        sheet_name=sheet_name,
        table_name=table_name,
        num_rows=total_rows,
        num_cols=total_cols,
    )
    table = doc.sheets[0].tables[0]
    for col_idx, header in enumerate(headers):
        if header is not None:
            table.write(0, col_idx, header)
    for row_offset, row_values in enumerate(rows, start=1):
        for col_idx, value in enumerate(row_values):
            if value is not None:
                table.write(row_offset, col_idx, value)

    if extra_sheet is not None:
        other_sheet_name, other_table_name, other_rows = extra_sheet
        other_cols = max((len(r) for r in other_rows), default=2)
        doc.add_sheet(
            sheet_name=other_sheet_name,
            table_name=other_table_name,
            num_rows=max(len(other_rows), 2),
            num_cols=max(other_cols, 2),
        )
        other_table = doc.sheets[other_sheet_name].tables[0]
        for r_idx, row_values in enumerate(other_rows):
            for c_idx, value in enumerate(row_values):
                if value is not None:
                    other_table.write(r_idx, c_idx, value)

    doc.save(str(path))
    return path


def _make_stock(symbol: str) -> StockData:
    return StockData(symbol=symbol, name=f"{symbol} Corp", currency="USD", price=100.0)


def _make_evaluation(
    symbol: str,
    *,
    verdict: Verdict,
    score: int,
    comment: str,
) -> StockEvaluation:
    return StockEvaluation(
        symbol=symbol,
        verdict=verdict,
        score=score,
        comment=comment,
        summary=f"Synthèse pour {symbol}.",
        strengths=["Point fort"],
        risks=["Risque"],
        model="gemini-test",
    )


# --- 1. Lecture (read_portfolio_rows) ---


def test_read_portfolio_rows_reads_col_zero_and_ignores_blank_rows(tmp_path: Path) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "portfolio.numbers",
        headers=["Action", "Quantité", "Note", "Commentaire"],
        rows=[
            ["UNM", 10, None, None],
            [None, None, None, None],
            ["   ", 5, None, None],
            ["  mo  ", 20, None, None],
            ["PEP", 15, None, None],
        ],
    )

    rows = read_portfolio_rows(numbers_file)

    assert rows == [
        PortfolioRow(row_index=1, symbol="UNM"),
        PortfolioRow(row_index=4, symbol="MO"),
        PortfolioRow(row_index=5, symbol="PEP"),
    ]


def test_read_portfolio_rows_selects_sheet_and_table_by_name(tmp_path: Path) -> None:
    path = tmp_path / "multi.numbers"
    doc = Document(sheet_name="Synthèse", table_name="Résumé", num_rows=3, num_cols=2)
    doc.sheets[0].tables[0].write(0, 0, "Indicateur")
    doc.sheets[0].tables[0].write(1, 0, "Total")

    doc.add_sheet(sheet_name="Portefeuille", table_name="Info", num_rows=2, num_cols=2)
    sheet = doc.sheets["Portefeuille"]
    sheet.tables[0].write(0, 0, "Meta")
    sheet.tables[0].write(1, 0, "2026")

    sheet.add_table(table_name="Titres", num_rows=4, num_cols=2)
    target_table = sheet.tables["Titres"]
    target_table.write(0, 0, "Action")
    target_table.write(1, 0, "UNM")
    target_table.write(3, 0, "PEP")
    doc.save(str(path))

    rows = read_portfolio_rows(path, sheet_name="Portefeuille", table_name="Titres")

    assert rows == [
        PortfolioRow(row_index=1, symbol="UNM"),
        PortfolioRow(row_index=3, symbol="PEP"),
    ]


def test_read_portfolio_rows_raises_on_missing_file_corrupt_file_or_missing_sheet_table(
    tmp_path: Path,
) -> None:
    with pytest.raises(NumbersPortfolioError, match="introuvable"):
        read_portfolio_rows(tmp_path / "absent.numbers")

    corrupt_file = tmp_path / "corrupt.numbers"
    corrupt_file.write_text("pas une archive numbers", encoding="utf-8")
    with pytest.raises(NumbersPortfolioError, match="Impossible d'ouvrir"):
        read_portfolio_rows(corrupt_file)

    valid_file = _create_numbers_file(
        tmp_path / "valid.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[["UNUM", None, None]],
        sheet_name="Feuille A",
        table_name="Tableau A",
    )

    with pytest.raises(NumbersPortfolioError, match="Feuille « Inconnue » introuvable"):
        read_portfolio_rows(valid_file, sheet_name="Inconnue")

    with pytest.raises(NumbersPortfolioError, match="Tableau « Inconnu » introuvable"):
        read_portfolio_rows(valid_file, sheet_name="Feuille A", table_name="Inconnu")


# --- 2. Mise à jour (update_numbers_portfolio) ---


def test_update_creates_decision_in_first_empty_column_and_preserves_rows(
    tmp_path: Path,
) -> None:
    # Colonne 5 ("Secteur") est occupée ; la colonne 6 est la première colonne vide après Commentaire.
    numbers_file = _create_numbers_file(
        tmp_path / "portfolio.numbers",
        headers=["Action", "Quantité", "PRU", "Note", "Commentaire", "Secteur", None],
        rows=[
            ["UNM", 12, 45.5, None, None, "Assurance", None],
            [None, None, None, None, None, None, None],
            ["MO", 30, 42.0, None, None, "Tabac", None],
            ["PEP", 8, 165.25, None, None, "Boissons", None],
        ],
        num_cols=7,
    )

    evals = {
        "UNM": _make_evaluation(
            "UNM",
            verdict="ACHAT",
            score=8,
            comment="Assureur rentable avec valorisation modérée.",
        ),
        "MO": _make_evaluation(
            "MO",
            verdict="CONSERVER",
            score=5,
            comment="Dividende élevé mais volumes en repli progressif.",
        ),
        "PEP": _make_evaluation(
            "PEP",
            verdict="VENDRE",
            score=2,
            comment="Pression sur les marges et valorisation tendue.",
        ),
    }

    provider = MagicMock()
    provider.get.side_effect = lambda sym, *, force_refresh=False: _make_stock(sym)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda stock: evals[stock.symbol]

    result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
    )

    assert result.decision_column_created is True
    assert result.sheet_name == "Portefeuille"
    assert result.table_name == "Actions"
    assert len(result.succeeded) == 3
    assert result.failed == []

    reloaded = Document(str(numbers_file))
    table = reloaded.sheets["Portefeuille"].tables["Actions"]

    # En-têtes : Secteur intact en col 5, Décision créée dans la première colonne vide (col 6).
    assert table.cell(0, 5).value == "Secteur"
    assert table.cell(0, 6).value == DECISION_HEADER

    # Ligne 1 : UNM conservé tel quel en col 0, colonnes métier intactes, Note/Commentaire/Décision écrits.
    # numbers-parser relit les cellules numériques Decimal128 en float (arrondi IEEE 754 possible, ex. 12).
    assert table.cell(1, 0).value == "UNM"
    assert table.cell(1, 1).value == pytest.approx(12)
    assert table.cell(1, 2).value == pytest.approx(45.5)
    assert table.cell(1, 3).value == 8
    assert table.cell(1, 4).value == "Assureur rentable avec valorisation modérée."
    assert len(str(table.cell(1, 4).value)) <= MAX_COMMENT_LENGTH
    assert table.cell(1, 5).value == "Assurance"
    assert table.cell(1, 6).value == "Achat"

    # Ligne 2 : ligne vide préservée intacte.
    assert all(table.cell(2, col).value is None for col in range(7))

    # Ligne 3 : MO -> Conserver, note 5.
    assert table.cell(3, 0).value == "MO"
    assert table.cell(3, 1).value == pytest.approx(30)
    assert table.cell(3, 2).value == pytest.approx(42.0)
    assert table.cell(3, 3).value == 5
    assert table.cell(3, 4).value == "Dividende élevé mais volumes en repli progressif."
    assert table.cell(3, 5).value == "Tabac"
    assert table.cell(3, 6).value == "Conserver"

    # Ligne 4 : PEP -> Vendre, note 2.
    assert table.cell(4, 0).value == "PEP"
    assert table.cell(4, 1).value == pytest.approx(8)
    assert table.cell(4, 2).value == pytest.approx(165.25)
    assert table.cell(4, 3).value == 2
    assert table.cell(4, 4).value == "Pression sur les marges et valorisation tendue."
    assert table.cell(4, 5).value == "Boissons"
    assert table.cell(4, 6).value == "Vendre"


def test_update_preserves_other_sheets_intact(tmp_path: Path) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "multisheet.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[["UNM", None, None]],
        num_cols=4,
        extra_sheet=(
            "Notes",
            "Suivi",
            [
                ["Date", "Observation"],
                ["2026-11-21", "Revue trimestrielle conservée"],
            ],
        ),
    )

    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=8,
        comment="Solide.",
    )

    update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
    )

    reloaded = Document(str(numbers_file))
    other_table = reloaded.sheets["Notes"].tables[0]
    assert other_table.cell(0, 0).value == "Date"
    assert other_table.cell(1, 1).value == "Revue trimestrielle conservée"


def test_update_appends_decision_column_when_no_empty_column_exists_after_commentaire(
    tmp_path: Path,
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "tight.numbers",
        headers=["Action", "Quantité", "Note", "Commentaire"],
        rows=[["UNM", 10, None, None]],
        num_cols=4,
    )

    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=9,
        comment="Solide génération de trésorerie.",
    )

    result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
    )

    assert result.decision_column_created is True
    assert len(result.succeeded) == 1

    reloaded = Document(str(numbers_file))
    table = reloaded.sheets[0].tables[0]
    assert table.num_cols == 5
    assert table.cell(0, 4).value == DECISION_HEADER
    assert table.cell(1, 2).value == 9
    assert table.cell(1, 3).value == "Solide génération de trésorerie."
    assert table.cell(1, 4).value == "Achat"


@pytest.mark.parametrize(
    "existing_decision_header",
    ["Décision", "  decision  ", "DECISION", "  DÉCISION "],
)
def test_update_reuses_existing_decision_column_case_and_accent_insensitive(
    tmp_path: Path,
    existing_decision_header: str,
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "existing_decision.numbers",
        headers=["Action", "Note", "Commentaire", existing_decision_header],
        rows=[["PEP", 1, "Ancien avis", "Vendre"]],
        num_cols=4,
    )

    provider = MagicMock()
    provider.get.return_value = _make_stock("PEP")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "PEP",
        verdict="CONSERVER",
        score=6,
        comment="Nouvel avis mis à jour.",
    )

    result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
    )

    assert result.decision_column_created is False
    reloaded = Document(str(numbers_file))
    table = reloaded.sheets[0].tables[0]
    assert table.num_cols == 4
    assert table.cell(0, 3).value == existing_decision_header
    assert table.cell(1, 1).value == 6
    assert table.cell(1, 2).value == "Nouvel avis mis à jour."
    assert table.cell(1, 3).value == "Conserver"


def test_update_tolerates_row_errors_and_continues_processing_other_rows(tmp_path: Path) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "partial.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[
            ["UNM", None, None, None],
            ["INCONNU", 4, "Note précédente", "Conserver"],
            ["MO", None, None, None],
            ["PEP", None, None, None],
            ["AAPL", None, None, None],
        ],
        num_cols=4,
    )

    def fake_get(symbol: str, *, force_refresh: bool = False) -> StockData:
        if symbol == "INCONNU":
            raise StockNotFoundError(f"Aucune donnée trouvée pour le symbole {symbol}.")
        if symbol == "MO":
            raise MarketDataError("Erreur réseau yfinance sur MO")
        return _make_stock(symbol)

    def fake_evaluate(stock: StockData) -> StockEvaluation:
        if stock.symbol == "PEP":
            raise EvaluationError("Quota Gemini dépassé sur PEP")
        if stock.symbol == "UNM":
            return _make_evaluation(
                "UNM",
                verdict="ACHAT",
                score=10,
                comment="Excellents fondamentaux.",
            )
        return _make_evaluation(
            stock.symbol,
            verdict="VENDRE",
            score=0,
            comment="Perspectives dégradées.",
        )

    provider = MagicMock()
    provider.get.side_effect = fake_get
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = fake_evaluate

    result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
    )

    assert len(result.results) == 5
    assert [r.symbol for r in result.succeeded] == ["UNM", "AAPL"]
    assert [r.symbol for r in result.failed] == ["INCONNU", "MO", "PEP"]

    failed_by_symbol = {r.symbol: r for r in result.failed}
    assert "Aucune donnée trouvée" in (failed_by_symbol["INCONNU"].error or "")
    assert "yfinance" in (failed_by_symbol["MO"].error or "")
    assert "Quota Gemini" in (failed_by_symbol["PEP"].error or "")

    reloaded = Document(str(numbers_file))
    table = reloaded.sheets[0].tables[0]
    assert table.cell(1, 1).value == 10
    assert table.cell(1, 3).value == "Achat"
    # La ligne en erreur garde ses anciennes valeurs intactes.
    assert table.cell(2, 1).value == 4
    assert table.cell(2, 2).value == "Note précédente"
    assert table.cell(2, 3).value == "Conserver"
    # Dernière ligne traitée malgré les erreurs intermédiaires.
    assert table.cell(5, 1).value == 0
    assert table.cell(5, 2).value == "Perspectives dégradées."
    assert table.cell(5, 3).value == "Vendre"


def test_update_saves_to_output_path_without_modifying_input_file(tmp_path: Path) -> None:
    input_file = _create_numbers_file(
        tmp_path / "input.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[["UNM", None, None]],
        num_cols=4,
    )
    original_bytes = input_file.read_bytes()
    output_file = tmp_path / "exports" / "output.numbers"

    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=7,
        comment="Valorisation attractive.",
    )

    result = update_numbers_portfolio(
        input_file,
        provider=provider,
        evaluator=evaluator,
        output_path=output_file,
    )

    assert result.input_path == input_file
    assert result.output_path == output_file
    assert output_file.is_file()
    assert input_file.read_bytes() == original_bytes

    reloaded_input = Document(str(input_file))
    assert reloaded_input.sheets[0].tables[0].cell(1, 1).value is None

    reloaded_output = Document(str(output_file))
    out_table = reloaded_output.sheets[0].tables[0]
    assert out_table.cell(0, 3).value == DECISION_HEADER
    assert out_table.cell(1, 1).value == 7
    assert out_table.cell(1, 2).value == "Valorisation attractive."
    assert out_table.cell(1, 3).value == "Achat"


def test_update_atomic_in_place_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    work_dir = tmp_path / "in_place"
    work_dir.mkdir()
    target_file = _create_numbers_file(
        work_dir / "suivi.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[["MO", None, None]],
        num_cols=3,
    )

    provider = MagicMock()
    provider.get.return_value = _make_stock("MO")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "MO",
        verdict="CONSERVER",
        score=5,
        comment="Rendement stable.",
    )

    result = update_numbers_portfolio(
        target_file,
        provider=provider,
        evaluator=evaluator,
    )

    assert result.input_path == target_file
    assert result.output_path == target_file
    assert [entry.name for entry in work_dir.iterdir()] == ["suivi.numbers"]

    reloaded = Document(str(target_file))
    assert reloaded.sheets[0].tables[0].cell(1, 1).value == 5


@pytest.mark.parametrize("force_refresh", [False, True])
def test_update_propagates_force_refresh_to_provider(
    tmp_path: Path, force_refresh: bool
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "refresh.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[["UNM", None, None, None]],
        num_cols=4,
    )

    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=8,
        comment="Très bon profil.",
    )

    update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
        force_refresh=force_refresh,
    )

    provider.get.assert_called_once_with("UNM", force_refresh=force_refresh)


@pytest.mark.parametrize(
    ("headers", "expected_missing"),
    [
        (["Action", "Commentaire", "Décision"], "Note"),
        (["Action", "Note", "Décision"], "Commentaire"),
    ],
)
def test_update_raises_numbers_portfolio_error_when_required_column_missing(
    tmp_path: Path,
    headers: list[str | None],
    expected_missing: str,
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "missing_header.numbers",
        headers=headers,
        rows=[["UNM", None, None]],
    )
    provider = MagicMock()
    evaluator = MagicMock()

    with pytest.raises(NumbersPortfolioError, match=expected_missing):
        update_numbers_portfolio(
            numbers_file,
            provider=provider,
            evaluator=evaluator,
        )

    provider.get.assert_not_called()
    evaluator.evaluate.assert_not_called()


def test_update_does_not_save_when_table_has_no_data_rows(tmp_path: Path) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "empty.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[],
        num_cols=3,
    )
    original_bytes = numbers_file.read_bytes()
    output_file = tmp_path / "out" / "empty_out.numbers"

    provider = MagicMock()
    evaluator = MagicMock()

    in_place_result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
    )

    assert in_place_result.saved is False
    assert in_place_result.decision_column_created is False
    assert in_place_result.results == []
    assert numbers_file.read_bytes() == original_bytes

    output_result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
        output_path=output_file,
    )

    assert output_result.saved is False
    assert output_result.decision_column_created is False
    assert not output_file.exists()
    provider.get.assert_not_called()
    evaluator.evaluate.assert_not_called()


def test_update_does_not_save_or_create_decision_column_when_all_rows_fail(
    tmp_path: Path,
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "all_fail.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[
            ["UNM", None, None],
            ["PEP", None, None],
        ],
        num_cols=3,
    )
    original_bytes = numbers_file.read_bytes()
    output_file = tmp_path / "out.numbers"

    provider = MagicMock()
    provider.get.side_effect = MarketDataError("Panne réseau")
    evaluator = MagicMock()

    in_place_result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
    )

    assert in_place_result.saved is False
    assert in_place_result.decision_column_created is False
    assert in_place_result.succeeded == []
    assert [r.symbol for r in in_place_result.failed] == ["UNM", "PEP"]
    assert numbers_file.read_bytes() == original_bytes
    reloaded = Document(str(numbers_file))
    assert reloaded.sheets[0].tables[0].num_cols == 3

    output_result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
        output_path=output_file,
    )

    assert output_result.saved is False
    assert output_result.decision_column_created is False
    assert not output_file.exists()
    evaluator.evaluate.assert_not_called()


def test_save_document_atomically_wraps_os_error_when_parent_is_a_file(tmp_path: Path) -> None:
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("bloque", encoding="utf-8")
    target_path = blocking_file / "portfolio.numbers"

    doc = MagicMock()

    with pytest.raises(NumbersPortfolioError, match="Impossible d'enregistrer"):
        _save_document_atomically(doc, target_path)

    doc.save.assert_not_called()


def test_save_document_atomically_cleans_up_temp_file_on_save_failure(tmp_path: Path) -> None:
    work_dir = tmp_path / "atomic_fail"
    work_dir.mkdir()
    target_path = work_dir / "portfolio.numbers"

    doc = MagicMock()
    doc.save.side_effect = RuntimeError("Disque plein")

    with pytest.raises(NumbersPortfolioError, match="Disque plein"):
        _save_document_atomically(doc, target_path)

    assert list(work_dir.iterdir()) == []


# --- 3. Robustesse P0/P1 : paquets, métadonnées, en-têtes, formules, dédoublonnage, atomicité ---


def test_read_and_update_apple_numbers_package_format_in_place_and_output(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "pkg_work"
    pkg_dir.mkdir()
    pkg_path = pkg_dir / "portfolio.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=3, num_cols=4)
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Action")
    table.write(0, 1, "Note")
    table.write(0, 2, "Commentaire")
    table.write(1, 0, "UNM")
    doc.save(str(pkg_path), package=True)
    assert pkg_path.is_dir()

    rows = read_portfolio_rows(pkg_path)
    assert rows == [PortfolioRow(row_index=1, symbol="UNM")]

    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=8,
        comment="Paquet Numbers mis à jour.",
    )

    output_pkg = tmp_path / "exports" / "out.numbers"
    result_out = update_numbers_portfolio(
        pkg_path,
        provider=provider,
        evaluator=evaluator,
        output_path=output_pkg,
    )
    assert result_out.saved is True
    assert output_pkg.is_dir()
    reloaded_out = Document(str(output_pkg))
    assert reloaded_out.sheets[0].tables[0].cell(1, 1).value == 8
    assert reloaded_out.sheets[0].tables[0].cell(1, 3).value == "Achat"

    result_inplace = update_numbers_portfolio(
        pkg_path,
        provider=provider,
        evaluator=evaluator,
    )
    assert result_inplace.saved is True
    assert pkg_path.is_dir()
    assert [entry.name for entry in pkg_dir.iterdir()] == ["portfolio.numbers"]
    reloaded_inplace = Document(str(pkg_path))
    assert reloaded_inplace.sheets[0].tables[0].cell(1, 1).value == 8
    assert reloaded_inplace.sheets[0].tables[0].cell(1, 2).value == "Paquet Numbers mis à jour."


def test_update_preserves_file_permissions_and_follows_symlink(tmp_path: Path) -> None:
    import stat

    real_file = _create_numbers_file(
        tmp_path / "real.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[["UNM", None, None]],
        num_cols=3,
    )
    real_file.chmod(0o644)
    link_file = tmp_path / "link.numbers"
    link_file.symlink_to(real_file)

    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=9,
        comment="Permissions et symlink préservés.",
    )

    update_numbers_portfolio(link_file, provider=provider, evaluator=evaluator)

    assert link_file.is_symlink()
    assert link_file.resolve() == real_file.resolve()
    assert stat.S_IMODE(real_file.stat().st_mode) == 0o644
    reloaded = Document(str(real_file))
    assert reloaded.sheets[0].tables[0].cell(1, 1).value == 9


def test_read_and_update_respect_multi_header_rows_and_ignore_summary_or_numeric_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multi_header.numbers"
    doc = Document(
        sheet_name="Portefeuille",
        table_name="Actions",
        num_rows=6,
        num_cols=3,
        num_header_rows=2,
    )
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Suivi 2026")
    table.write(1, 0, "Action")
    table.write(1, 1, "Note")
    table.write(1, 2, "Commentaire")
    table.write(2, 0, "UNM")
    table.write(3, 0, 42)
    table.write(4, 0, "Total")
    table.write(4, 1, 99)
    table.write(5, 0, "Moyenne")
    doc.save(str(path))

    assert read_portfolio_rows(path) == [PortfolioRow(row_index=2, symbol="UNM")]

    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=8,
        comment="En-tête ligne 1 respecté.",
    )

    result = update_numbers_portfolio(path, provider=provider, evaluator=evaluator)

    assert len(result.succeeded) == 1
    assert result.failed == []
    assert result.decision_column_created is True
    provider.get.assert_called_once_with("UNM", force_refresh=False)

    reloaded = Document(str(path))
    reloaded_table = reloaded.sheets[0].tables[0]
    # L'en-tête Décision doit être créé sur la ligne d'en-tête réelle (1), pas sur le titre (0).
    assert reloaded_table.cell(0, 3).value is None
    assert reloaded_table.cell(1, 3).value == DECISION_HEADER
    assert reloaded_table.cell(2, 1).value == 8
    assert reloaded_table.cell(2, 3).value == "Achat"
    # La ligne Total ne doit pas avoir été touchée.
    assert reloaded_table.cell(4, 1).value == pytest.approx(99)


@pytest.mark.parametrize(
    ("headers", "expected_fragment"),
    [
        (["Action", "Note", "Commentaire", "note"], "Plusieurs colonnes"),
        (["Action", "Note", "Commentaire", "COMMENTAIRE"], "Plusieurs colonnes"),
        (["Action", "Note", "Commentaire", "Décision", "decision"], "Plusieurs colonnes"),
        (["Note", "Commentaire", "Autre"], "colonne 0"),
    ],
)
def test_update_rejects_duplicate_or_misplaced_headers(
    tmp_path: Path,
    headers: list[str | None],
    expected_fragment: str,
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "bad_headers.numbers",
        headers=headers,
        rows=[["UNM"] + [None] * (len(headers) - 1)],
    )
    provider = MagicMock()
    evaluator = MagicMock()

    with pytest.raises(NumbersPortfolioError, match=expected_fragment):
        update_numbers_portfolio(numbers_file, provider=provider, evaluator=evaluator)

    provider.get.assert_not_called()
    evaluator.evaluate.assert_not_called()


def test_update_protects_formula_cells_in_col_zero_and_target_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "formulas.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[
            ["CALC", None, None, None],
            ["UNM", 7, "Ancien", "Achat"],
            ["PEP", None, None, None],
        ],
        num_cols=4,
    )
    real_doc = Document(str(numbers_file))
    real_table = real_doc.sheets[0].tables[0]
    original_cell_fn = real_table.cell

    class _FormulaCellProxy:
        def __init__(self, wrapped: Any, formula: str) -> None:
            self._wrapped = wrapped
            self.formula = formula

        @property
        def value(self) -> Any:
            return self._wrapped.value

    def patched_cell(row_idx: int, col_idx: int) -> Any:
        cell = original_cell_fn(row_idx, col_idx)
        # Ligne 1 col 0 : formule en colonne Action -> ligne ignorée.
        if (row_idx, col_idx) == (1, 0):
            return _FormulaCellProxy(cell, "=CONCAT(\"CA\",\"LC\")")
        # Ligne 2 col 1 (Note) : formule en colonne cible -> erreur de protection, rien écrasé.
        if (row_idx, col_idx) == (2, 1):
            return _FormulaCellProxy(cell, "=AVERAGE(B2:B3)")
        return cell

    monkeypatch.setattr(real_table, "cell", patched_cell)
    monkeypatch.setattr("portfolio_tracker.numbers_portfolio._open_document", lambda p: real_doc)

    provider = MagicMock()
    provider.get.side_effect = lambda sym, *, force_refresh=False: _make_stock(sym)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda stock: _make_evaluation(
        stock.symbol,
        verdict="CONSERVER",
        score=5,
        comment="Ok.",
    )

    result = update_numbers_portfolio(numbers_file, provider=provider, evaluator=evaluator)

    assert [r.symbol for r in result.succeeded] == ["PEP"]
    assert [r.symbol for r in result.failed] == ["UNM"]
    assert "formule détectée" in (result.failed[0].error or "")
    # Ni CALC (col 0 formule) ni UNM (col Note formule) n'ont déclenché d'appel API.
    provider.get.assert_called_once_with("PEP", force_refresh=False)

    reloaded = Document(str(numbers_file))
    reloaded_table = reloaded.sheets[0].tables[0]
    assert reloaded_table.cell(2, 1).value == 7
    assert reloaded_table.cell(2, 2).value == "Ancien"
    assert reloaded_table.cell(3, 1).value == 5


def test_update_deduplicates_identical_symbols_within_single_run(tmp_path: Path) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "duplicates.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[
            ["AAPL", None, None, None],
            ["INCONNU", None, None, None],
            ["  aapl  ", None, None, None],
            ["inconnu", None, None, None],
        ],
        num_cols=4,
    )

    def fake_get(symbol: str, *, force_refresh: bool = False) -> StockData:
        if symbol == "INCONNU":
            raise StockNotFoundError("Introuvable")
        return _make_stock(symbol)

    provider = MagicMock(side_effect=None)
    provider.get.side_effect = fake_get
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "AAPL",
        verdict="ACHAT",
        score=8,
        comment="Évaluation unique partagée.",
    )

    result = update_numbers_portfolio(
        numbers_file,
        provider=provider,
        evaluator=evaluator,
        force_refresh=True,
    )

    assert provider.get.call_count == 2
    assert evaluator.evaluate.call_count == 1
    assert [r.row_index for r in result.succeeded] == [1, 3]
    assert [r.row_index for r in result.failed] == [2, 4]

    reloaded = Document(str(numbers_file))
    table = reloaded.sheets[0].tables[0]
    assert table.cell(1, 1).value == 8
    assert table.cell(3, 1).value == 8
    assert table.cell(1, 2).value == table.cell(3, 2).value == "Évaluation unique partagée."
    assert table.cell(1, 3).value == table.cell(3, 3).value == "Achat"


def test_update_aborts_without_saving_on_partial_table_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "write_fail.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[
            ["UNM", 4, "Initial UNM", "Conserver"],
            ["PEP", 5, "Initial PEP", "Conserver"],
        ],
        num_cols=4,
    )
    original_bytes = numbers_file.read_bytes()
    real_doc = Document(str(numbers_file))
    real_table = real_doc.sheets[0].tables[0]
    original_write = real_table.write

    def flaky_write(row: int, col: int, value: Any, *args: Any, **kwargs: Any) -> None:
        # Laisse passer la ligne 1 et la Note de la ligne 2, puis échoue sur Commentaire ligne 2.
        if (row, col) == (2, 2):
            raise RuntimeError("Erreur interne écriture cellule")
        original_write(row, col, value, *args, **kwargs)

    monkeypatch.setattr(real_table, "write", flaky_write)
    monkeypatch.setattr("portfolio_tracker.numbers_portfolio._open_document", lambda p: real_doc)

    provider = MagicMock()
    provider.get.side_effect = lambda sym, *, force_refresh=False: _make_stock(sym)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda stock: _make_evaluation(
        stock.symbol,
        verdict="ACHAT",
        score=9,
        comment="Nouveau commentaire.",
    )

    with pytest.raises(NumbersPortfolioError, match="Erreur lors de l'écriture de la ligne 2"):
        update_numbers_portfolio(numbers_file, provider=provider, evaluator=evaluator)

    # Le fichier sur disque ne doit pas avoir été sauvegardé avec la ligne 2 à moitié écrite.
    assert numbers_file.read_bytes() == original_bytes


def test_update_does_not_swallow_unexpected_programming_errors(tmp_path: Path) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "prog_err.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[["UNM", None, None, None]],
        num_cols=4,
    )
    provider = MagicMock()
    provider.get.return_value = _make_stock("UNM")
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = TypeError("bug interne inattendu")

    with pytest.raises(TypeError, match="bug interne inattendu"):
        update_numbers_portfolio(numbers_file, provider=provider, evaluator=evaluator)


def test_update_saves_completed_rows_on_keyboard_interrupt_and_reports_progress(
    tmp_path: Path,
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "interrupt.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[
            ["UNM", None, None, None],
            ["PEP", None, None, None],
        ],
        num_cols=4,
    )
    progress_events: list[tuple[str, int, int]] = []

    def fake_get(symbol: str, *, force_refresh: bool = False) -> StockData:
        if symbol == "PEP":
            raise KeyboardInterrupt()
        return _make_stock(symbol)

    provider = MagicMock()
    provider.get.side_effect = fake_get
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _make_evaluation(
        "UNM",
        verdict="ACHAT",
        score=8,
        comment="Sauvegardé avant interruption.",
    )

    with pytest.raises(KeyboardInterrupt):
        update_numbers_portfolio(
            numbers_file,
            provider=provider,
            evaluator=evaluator,
            on_row_processed=lambda res, idx, total: progress_events.append(
                (res.symbol, idx, total)
            ),
        )

    assert progress_events == [("UNM", 1, 2)]
    reloaded = Document(str(numbers_file))
    table = reloaded.sheets[0].tables[0]
    assert table.cell(1, 1).value == 8
    assert table.cell(1, 2).value == "Sauvegardé avant interruption."
    assert table.cell(2, 1).value is None


def test_update_does_not_save_partially_mutated_row_on_keyboard_interrupt_mid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    numbers_file = _create_numbers_file(
        tmp_path / "interrupt_mid_write.numbers",
        headers=["Action", "Note", "Commentaire", "Décision"],
        rows=[
            ["UNM", 4, "Initial UNM", "Conserver"],
            ["PEP", 5, "Initial PEP", "Conserver"],
        ],
        num_cols=4,
    )
    original_bytes = numbers_file.read_bytes()
    real_doc = Document(str(numbers_file))
    real_table = real_doc.sheets[0].tables[0]
    original_write = real_table.write

    def interrupting_write(row: int, col: int, value: Any, *args: Any, **kwargs: Any) -> None:
        # Laisse finir la ligne 1 et écrire la Note de la ligne 2, puis interrompt avant Commentaire/Décision.
        if (row, col) == (2, 2):
            raise KeyboardInterrupt()
        original_write(row, col, value, *args, **kwargs)

    monkeypatch.setattr(real_table, "write", interrupting_write)
    monkeypatch.setattr("portfolio_tracker.numbers_portfolio._open_document", lambda p: real_doc)

    provider = MagicMock()
    provider.get.side_effect = lambda sym, *, force_refresh=False: _make_stock(sym)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda stock: _make_evaluation(
        stock.symbol,
        verdict="ACHAT",
        score=9,
        comment="Nouveau commentaire.",
    )

    with pytest.raises(KeyboardInterrupt):
        update_numbers_portfolio(numbers_file, provider=provider, evaluator=evaluator)

    # Le fichier sur disque ne doit pas contenir la ligne 2 corrompue (Note=9 + ancien Commentaire/Décision).
    assert numbers_file.read_bytes() == original_bytes


def test_save_package_safely_restores_original_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    pkg_dir = tmp_path / "pkg_interrupt"
    pkg_dir.mkdir()
    pkg_path = pkg_dir / "portfolio.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=2, num_cols=2)
    doc.sheets[0].tables[0].write(0, 0, "Action")
    doc.sheets[0].tables[0].write(1, 0, "UNM")
    doc.save(str(pkg_path), package=True)

    original_rename = os.rename
    rename_calls = 0

    def interrupting_second_rename(src: Any, dst: Any) -> None:
        nonlocal rename_calls
        rename_calls += 1
        if rename_calls == 2:
            raise KeyboardInterrupt()
        original_rename(src, dst)

    monkeypatch.setattr(os, "rename", interrupting_second_rename)

    with pytest.raises(KeyboardInterrupt):
        _save_document_atomically(doc, pkg_path, package=True)

    # Le paquet original doit avoir été restauré et aucun dossier temporaire ne doit subsister.
    assert pkg_path.is_dir()
    assert [entry.name for entry in pkg_dir.iterdir()] == ["portfolio.numbers"]
    reloaded = Document(str(pkg_path))
    assert reloaded.sheets[0].tables[0].cell(1, 0).value == "UNM"


def test_save_package_safely_preserves_backup_when_rollback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    pkg_dir = tmp_path / "pkg_rollback_fail"
    pkg_dir.mkdir()
    pkg_path = pkg_dir / "portfolio.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=2, num_cols=2)
    doc.sheets[0].tables[0].write(0, 0, "Action")
    doc.sheets[0].tables[0].write(1, 0, "PEP")
    doc.save(str(pkg_path), package=True)

    original_rename = os.rename
    rename_calls = 0

    def failing_replace_and_rollback(src: Any, dst: Any) -> None:
        nonlocal rename_calls
        rename_calls += 1
        if rename_calls == 1:
            original_rename(src, dst)
            return
        raise OSError("rename refusé")

    monkeypatch.setattr(os, "rename", failing_replace_and_rollback)

    with pytest.raises(NumbersPortfolioError, match="Sauvegarde originale conservée") as exc_info:
        _save_document_atomically(doc, pkg_path, package=True)

    backups = list(pkg_dir.glob(".portfolio.*/backup.numbers"))
    assert len(backups) == 1
    assert str(backups[0]) in str(exc_info.value)
    reloaded_backup = Document(str(backups[0]))
    assert reloaded_backup.sheets[0].tables[0].cell(1, 0).value == "PEP"


def test_save_package_rejects_overwriting_existing_regular_file(tmp_path: Path) -> None:
    single_file = _create_numbers_file(
        tmp_path / "target.numbers",
        headers=["Action", "Note", "Commentaire"],
        rows=[["UNM", None, None]],
        num_cols=3,
    )
    single_file.chmod(0o644)
    original_bytes = single_file.read_bytes()
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=2, num_cols=2)

    with pytest.raises(NumbersPortfolioError, match="format paquet"):
        _save_document_atomically(doc, single_file, package=True)

    assert single_file.is_file()
    assert single_file.read_bytes() == original_bytes


def test_save_single_file_cleans_up_temp_file_on_keyboard_interrupt(tmp_path: Path) -> None:
    work_dir = tmp_path / "single_interrupt"
    work_dir.mkdir()
    target_path = work_dir / "portfolio.numbers"

    doc = MagicMock()
    doc.save.side_effect = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _save_document_atomically(doc, target_path, package=False)

    assert list(work_dir.iterdir()) == []


@pytest.mark.parametrize("dir_name", ["exports", "fake.numbers"])
def test_save_package_refuses_to_overwrite_existing_non_numbers_directory(
    tmp_path: Path,
    dir_name: str,
) -> None:
    pkg_path = tmp_path / "source.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=2, num_cols=3)
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Action")
    table.write(0, 1, "Note")
    table.write(0, 2, "Commentaire")
    table.write(1, 0, "UNM")
    doc.save(str(pkg_path), package=True)
    assert pkg_path.is_dir()

    existing_dir = tmp_path / dir_name
    existing_dir.mkdir()
    important_file = existing_dir / "taxes_2026.pdf"
    important_file.write_text("DONNÉES CRITIQUES", encoding="utf-8")

    # Refuse directement dans _save_document_atomically sans toucher au répertoire existant.
    with pytest.raises(NumbersPortfolioError, match="paquet Numbers valide"):
        _save_document_atomically(doc, existing_dir, package=True)

    assert important_file.is_file()
    assert important_file.read_text(encoding="utf-8") == "DONNÉES CRITIQUES"

    # Refuse aussi dès l'entrée de update_numbers_portfolio avant tout appel réseau/LLM.
    provider = MagicMock()
    evaluator = MagicMock()
    with pytest.raises(NumbersPortfolioError, match="paquet Numbers valide"):
        update_numbers_portfolio(
            pkg_path,
            provider=provider,
            evaluator=evaluator,
            output_path=existing_dir,
        )

    provider.get.assert_not_called()
    evaluator.evaluate.assert_not_called()
    assert important_file.is_file()
    assert important_file.read_text(encoding="utf-8") == "DONNÉES CRITIQUES"


def test_save_package_refuses_symlink_to_non_numbers_dir_and_source_parent_dir(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    pkg_path = work_dir / "portfolio.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=2, num_cols=3)
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Action")
    table.write(0, 1, "Note")
    table.write(0, 2, "Commentaire")
    table.write(1, 0, "UNM")
    doc.save(str(pkg_path), package=True)

    # Cas 1 : symlink pointant vers un répertoire ordinaire existant.
    real_dir = tmp_path / "user_docs"
    real_dir.mkdir()
    marker = real_dir / "keep.txt"
    marker.write_text("INTACT", encoding="utf-8")
    link_to_dir = tmp_path / "export.numbers"
    link_to_dir.symlink_to(real_dir)

    provider = MagicMock()
    evaluator = MagicMock()
    with pytest.raises(NumbersPortfolioError, match="paquet Numbers valide"):
        update_numbers_portfolio(
            pkg_path,
            provider=provider,
            evaluator=evaluator,
            output_path=link_to_dir,
        )

    assert link_to_dir.is_symlink()
    assert marker.read_text(encoding="utf-8") == "INTACT"

    # Cas 2 : répertoire parent de la source passé comme --output (ex. --output .).
    with pytest.raises(NumbersPortfolioError, match="répertoire parent"):
        update_numbers_portfolio(
            pkg_path,
            provider=provider,
            evaluator=evaluator,
            output_path=work_dir,
        )

    assert pkg_path.is_dir()
    provider.get.assert_not_called()
    evaluator.evaluate.assert_not_called()


def test_update_skips_column_with_merged_cell_when_creating_decision_column(tmp_path: Path) -> None:
    # Cas 1 : C2:D2 est fusionnée -> D (col 3) contient un MergedCell (exclue pour Décision)
    # et C2 (ancre de fusion sur Commentaire) protège la ligne AAPL contre l'écrasement.
    merged_last_col = tmp_path / "merged_last.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=3, num_cols=4)
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Action")
    table.write(0, 1, "Note")
    table.write(0, 2, "Commentaire")
    table.write(1, 0, "AAPL")
    table.write(1, 1, 5)
    table.write(1, 2, "Commentaire fusionné")
    table.write(2, 0, "NVDA")
    table.merge_cells("C2:D2")
    doc.save(str(merged_last_col))

    provider = MagicMock()
    provider.get.side_effect = lambda sym, *, force_refresh=False: _make_stock(sym)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda stock: _make_evaluation(
        stock.symbol,
        verdict="ACHAT",
        score=8,
        comment=f"Avis {stock.symbol}.",
    )

    result = update_numbers_portfolio(merged_last_col, provider=provider, evaluator=evaluator)

    assert result.decision_column_created is True
    assert [r.symbol for r in result.succeeded] == ["NVDA"]
    assert [r.symbol for r in result.failed] == ["AAPL"]
    assert "Commentaire" in (result.failed[0].error or "")
    provider.get.assert_called_once_with("NVDA", force_refresh=False)

    reloaded = Document(str(merged_last_col))
    out_table = reloaded.sheets[0].tables[0]
    assert out_table.num_cols == 5
    assert out_table.cell(0, 3).value is None
    assert out_table.cell(0, 4).value == DECISION_HEADER
    assert out_table.cell(1, 1).value == 5
    assert out_table.cell(1, 2).value == "Commentaire fusionné"
    assert out_table.cell(2, 1).value == 8
    assert out_table.cell(2, 2).value == "Avis NVDA."
    assert out_table.cell(2, 4).value == "Achat"

    # Cas 2 : en-tête vide fusionné D1:E1 (D1 est ancre is_merged=True, E1 est MergedCell) ->
    # les deux colonnes D et E sont ignorées, et la colonne F (index 5) véritablement vide est choisie.
    merged_header_then_empty = tmp_path / "merged_header_then_empty.numbers"
    doc2 = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=2, num_cols=6)
    table2 = doc2.sheets[0].tables[0]
    table2.write(0, 0, "Action")
    table2.write(0, 1, "Note")
    table2.write(0, 2, "Commentaire")
    table2.write(1, 0, "AAPL")
    table2.merge_cells("D1:E1")
    doc2.save(str(merged_header_then_empty))

    provider.reset_mock()
    result2 = update_numbers_portfolio(
        merged_header_then_empty, provider=provider, evaluator=evaluator
    )
    assert result2.decision_column_created is True
    assert [r.symbol for r in result2.succeeded] == ["AAPL"]
    reloaded2 = Document(str(merged_header_then_empty))
    out_table2 = reloaded2.sheets[0].tables[0]
    assert out_table2.num_cols == 6
    assert out_table2.cell(0, 3).value is None
    assert out_table2.cell(0, 5).value == DECISION_HEADER
    assert out_table2.cell(1, 5).value == "Achat"


def test_update_skips_row_when_target_cell_is_merged_without_calling_provider(
    tmp_path: Path,
) -> None:
    numbers_file = tmp_path / "merged_target.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=3, num_cols=4)
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Action")
    table.write(0, 1, "Note")
    table.write(0, 2, "Commentaire")
    table.write(0, 3, "Décision")
    table.write(1, 0, "AAPL")
    table.write(1, 1, 5)
    table.write(1, 2, "Ancien commentaire")
    table.merge_cells("C2:D2")
    table.write(2, 0, "NVDA")
    doc.save(str(numbers_file))

    provider = MagicMock()
    provider.get.side_effect = lambda sym, *, force_refresh=False: _make_stock(sym)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda stock: _make_evaluation(
        stock.symbol,
        verdict="CONSERVER",
        score=6,
        comment="Évaluation NVDA.",
    )

    result = update_numbers_portfolio(numbers_file, provider=provider, evaluator=evaluator)

    assert [r.symbol for r in result.succeeded] == ["NVDA"]
    assert [r.symbol for r in result.failed] == ["AAPL"]
    assert "cellule fusionnée détectée" in (result.failed[0].error or "")
    assert "Commentaire" in (result.failed[0].error or "")
    assert "Décision" in (result.failed[0].error or "")
    provider.get.assert_called_once_with("NVDA", force_refresh=False)

    reloaded = Document(str(numbers_file))
    out_table = reloaded.sheets[0].tables[0]
    # La ligne fusionnée AAPL reste intacte et la ligne NVDA est mise à jour.
    assert out_table.cell(1, 1).value == 5
    assert out_table.cell(1, 2).value == "Ancien commentaire"
    assert out_table.cell(2, 1).value == 6
    assert out_table.cell(2, 2).value == "Évaluation NVDA."
    assert out_table.cell(2, 3).value == "Conserver"


def test_update_protects_vertical_and_cross_column_merge_anchor_cells(tmp_path: Path) -> None:
    # B2:B3 fusionne verticalement Note sur AAPL (ancre B2) et NVDA (couverte B3).
    # C4:D4 fusionne horizontalement Commentaire (ancre C4) avec Secteur (colonne non cible D4) sur MSFT.
    numbers_file = tmp_path / "merged_anchors.numbers"
    doc = Document(sheet_name="Portefeuille", table_name="Actions", num_rows=5, num_cols=5)
    table = doc.sheets[0].tables[0]
    table.write(0, 0, "Action")
    table.write(0, 1, "Note")
    table.write(0, 2, "Commentaire")
    table.write(0, 3, "Secteur")
    table.write(0, 4, "Décision")
    table.write(1, 0, "AAPL")
    table.write(1, 1, 7)
    table.write(1, 2, "Note partagée")
    table.write(2, 0, "NVDA")
    table.write(3, 0, "MSFT")
    table.write(3, 2, "Fusion avec secteur")
    table.write(4, 0, "PEP")
    table.merge_cells("B2:B3")
    table.merge_cells("C4:D4")
    doc.save(str(numbers_file))

    provider = MagicMock()
    provider.get.side_effect = lambda sym, *, force_refresh=False: _make_stock(sym)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda stock: _make_evaluation(
        stock.symbol,
        verdict="ACHAT",
        score=9,
        comment=f"Avis {stock.symbol}.",
    )

    result = update_numbers_portfolio(numbers_file, provider=provider, evaluator=evaluator)

    assert [r.symbol for r in result.succeeded] == ["PEP"]
    assert [r.symbol for r in result.failed] == ["AAPL", "NVDA", "MSFT"]
    failed_by_symbol = {r.symbol: r.error or "" for r in result.failed}
    assert "Note" in failed_by_symbol["AAPL"]
    assert "Note" in failed_by_symbol["NVDA"]
    assert "Commentaire" in failed_by_symbol["MSFT"]
    # Aucun appel réseau pour les lignes contenant une ancre ou une cellule couverte fusionnée.
    provider.get.assert_called_once_with("PEP", force_refresh=False)

    reloaded = Document(str(numbers_file))
    out_table = reloaded.sheets[0].tables[0]
    assert out_table.cell(1, 1).value == 7
    assert out_table.cell(1, 2).value == "Note partagée"
    assert out_table.cell(3, 2).value == "Fusion avec secteur"
    assert out_table.cell(4, 1).value == 9
    assert out_table.cell(4, 2).value == "Avis PEP."
    assert out_table.cell(4, 4).value == "Achat"
