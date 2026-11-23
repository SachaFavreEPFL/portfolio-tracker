import logging
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from numbers_parser import Document, MergedCell

from portfolio_tracker.contracts import (
    MarketDataProvider,
    PortfolioRow,
    PortfolioRowResult,
    PortfolioUpdateResult,
    StockEvaluation,
    StockEvaluator,
    format_verdict,
)
from portfolio_tracker.evaluator import EvaluationError
from portfolio_tracker.market_data import MarketDataError

logger = logging.getLogger(__name__)

DECISION_HEADER = "Décision"

_SUMMARY_ROW_LABELS = frozenset(
    {
        "total",
        "totaux",
        "sous-total",
        "sous total",
        "subtotal",
        "moyenne",
        "average",
        "somme",
        "sum",
        "synthese",
        "bilan",
        "resume",
    }
)

ProgressCallback = Callable[[PortfolioRowResult, int, int], None]


class NumbersPortfolioError(Exception):
    """Erreur lors de la lecture, l'analyse de structure ou l'écriture d'un fichier Numbers."""


def _normalize_header(value: Any) -> str:
    """Normalise un libellé d'en-tête (espaces compressés, sans accents, minuscules)."""
    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.split())
    decomposed = unicodedata.normalize("NFKD", collapsed)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_accents.lower()


def _is_empty_cell_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _cell_has_formula(table: Any, row_idx: int, col_idx: int) -> bool:
    if row_idx < 0 or col_idx < 0 or row_idx >= table.num_rows or col_idx >= table.num_cols:
        return False
    return getattr(table.cell(row_idx, col_idx), "formula", None) is not None


def _cell_is_merged(table: Any, row_idx: int, col_idx: int) -> bool:
    """Indique si la cellule appartient à une fusion (ancre ou cellule couverte)."""
    if row_idx < 0 or col_idx < 0 or row_idx >= table.num_rows or col_idx >= table.num_cols:
        return False
    cell = table.cell(row_idx, col_idx)
    # numbers-parser marque l'ancre (haut-gauche) via cell.is_merged=True et les autres cellules en MergedCell.
    return isinstance(cell, MergedCell) or getattr(cell, "is_merged", False) is True


def _table_header_row_count(table: Any) -> int:
    """Retourne le nombre de lignes d'en-tête à ignorer (au moins 1 si le tableau n'est pas vide)."""
    num_rows = int(table.num_rows)
    if num_rows <= 0:
        return 0
    raw = getattr(table, "num_header_rows", 1)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
        return min(raw, num_rows)
    return 1


def _open_document(input_path: Path) -> Any:
    # Apple Numbers stocke les classeurs soit en archive mono-fichier, soit en paquet (répertoire .numbers/).
    if not input_path.exists():
        raise NumbersPortfolioError(f"Fichier Numbers introuvable : {input_path}")
    try:
        return Document(str(input_path))
    except Exception as exc:
        raise NumbersPortfolioError(
            f"Impossible d'ouvrir le fichier Numbers {input_path} : {exc}"
        ) from exc


def _select_sheet_and_table(
    doc: Any,
    *,
    sheet_name: str | None = None,
    table_name: str | None = None,
) -> tuple[Any, Any]:
    sheets = list(doc.sheets)
    if not sheets:
        raise NumbersPortfolioError("Le fichier Numbers ne contient aucune feuille.")

    if sheet_name is None:
        sheet = sheets[0]
    else:
        matched_sheets = [item for item in sheets if item.name == sheet_name]
        if not matched_sheets:
            raise NumbersPortfolioError(
                f"Feuille « {sheet_name} » introuvable dans le fichier Numbers."
            )
        sheet = matched_sheets[0]

    tables = list(sheet.tables)
    if not tables:
        raise NumbersPortfolioError(f"La feuille « {sheet.name} » ne contient aucun tableau.")

    if table_name is None:
        table = tables[0]
    else:
        matched_tables = [item for item in tables if item.name == table_name]
        if not matched_tables:
            raise NumbersPortfolioError(
                f"Tableau « {table_name} » introuvable dans la feuille « {sheet.name} »."
            )
        table = matched_tables[0]

    return sheet, table


def _extract_rows_from_table(
    table: Any,
    *,
    start_row: int | None = None,
) -> list[PortfolioRow]:
    if table.num_rows <= 1 or table.num_cols == 0:
        return []

    data_start = start_row if start_row is not None else _table_header_row_count(table)
    if data_start >= table.num_rows:
        return []

    rows: list[PortfolioRow] = []
    for row_idx in range(data_start, table.num_rows):
        cell = table.cell(row_idx, 0)
        # Ignore les cellules calculées (formules), non textuelles ou les lignes de synthèse (Total, Moyenne...).
        if getattr(cell, "formula", None) is not None:
            continue
        raw_value = cell.value
        if _is_empty_cell_value(raw_value) or not isinstance(raw_value, str):
            continue
        if _normalize_header(raw_value) in _SUMMARY_ROW_LABELS:
            continue
        symbol = raw_value.strip().upper()
        if symbol:
            rows.append(PortfolioRow(row_index=row_idx, symbol=symbol))
    return rows


def _is_column_empty(table: Any, col: int, num_rows: int) -> bool:
    return all(
        _is_empty_cell_value(table.cell(row_idx, col).value)
        and not _cell_has_formula(table, row_idx, col)
        and not _cell_is_merged(table, row_idx, col)
        for row_idx in range(num_rows)
    )


def _find_header_row_index(table: Any) -> int:
    """Repère la ligne d'en-tête portant Note/Commentaire parmi les lignes d'en-tête du tableau."""
    header_limit = _table_header_row_count(table)
    num_cols = table.num_cols
    best_partial: int | None = None
    best_full: int | None = None
    for row_idx in range(header_limit):
        headers = {_normalize_header(table.cell(row_idx, col).value) for col in range(num_cols)}
        has_note = "note" in headers
        has_comment = "commentaire" in headers
        if has_note and has_comment:
            best_full = row_idx
        elif has_note or has_comment:
            best_partial = row_idx
    if best_full is not None:
        return best_full
    if best_partial is not None:
        return best_partial
    return 0


def _locate_unique_column(
    headers: list[str],
    normalized_name: str,
    display_label: str,
    *,
    required: bool,
) -> int | None:
    matches = [idx for idx, name in enumerate(headers) if name == normalized_name]
    if len(matches) > 1:
        raise NumbersPortfolioError(
            f"Plusieurs colonnes d'en-tête « {display_label} » détectées dans le tableau."
        )
    if not matches:
        if required:
            raise NumbersPortfolioError(
                f"Colonne d'en-tête « {display_label} » introuvable dans le tableau."
            )
        return None
    col_idx = matches[0]
    if col_idx == 0:
        raise NumbersPortfolioError(
            f"La colonne d'en-tête « {display_label} » ne peut pas occuper la colonne 0 (réservée aux symboles)."
        )
    return col_idx


def _locate_columns(table: Any) -> tuple[int, int, int, int, bool]:
    """Repère la ligne d'en-tête, Note, Commentaire et Décision (sans muter le tableau avant écriture)."""
    num_rows = table.num_rows
    num_cols = table.num_cols
    if num_rows == 0 or num_cols == 0:
        raise NumbersPortfolioError("Le tableau cible est vide.")

    header_row_idx = _find_header_row_index(table)
    headers = [_normalize_header(table.cell(header_row_idx, col).value) for col in range(num_cols)]

    note_col = _locate_unique_column(headers, "note", "Note", required=True)
    comment_col = _locate_unique_column(headers, "commentaire", "Commentaire", required=True)
    assert note_col is not None and comment_col is not None

    decision_col = _locate_unique_column(headers, "decision", DECISION_HEADER, required=False)
    if decision_col is not None:
        return header_row_idx, note_col, comment_col, decision_col, False

    # Cherche d'abord une colonne entièrement vide après Commentaire pour ne pas écraser de données.
    for col in range(comment_col + 1, num_cols):
        if _is_column_empty(table, col, num_rows):
            return header_row_idx, note_col, comment_col, col, True

    # Écrire à l'index num_cols étend automatiquement le tableau d'une colonne dans numbers-parser.
    return header_row_idx, note_col, comment_col, num_cols, True


def _is_numbers_package_dir(path: Path) -> bool:
    """Vérifie qu'un répertoire existant correspond bien à un paquet Apple Numbers."""
    return (
        path.is_dir()
        and path.suffix.lower() == ".numbers"
        and (
            (path / "Index.zip").is_file()
            or (path / "Index" / "Document.iwa").is_file()
        )
    )


def _validate_save_target(
    target_path: Path,
    *,
    package: bool,
    source_path: Path | None = None,
) -> Path:
    """Résout et vérifie que la cible de sauvegarde ne risque pas d'écraser un répertoire arbitraire."""
    resolved_target = target_path.resolve() if target_path.is_symlink() else target_path
    canonical_target = resolved_target.resolve()

    if canonical_target == canonical_target.parent:
        raise NumbersPortfolioError(f"Chemin de sortie invalide : {target_path}")

    if source_path is not None and canonical_target in source_path.resolve().parents:
        raise NumbersPortfolioError(
            f"Le chemin de sortie {target_path} ne peut pas être un répertoire parent "
            f"du fichier source {source_path}."
        )

    if package:
        if resolved_target.exists():
            if not resolved_target.is_dir():
                raise NumbersPortfolioError(
                    f"La cible {resolved_target} est un fichier alors que la sauvegarde est au format paquet."
                )
            if not _is_numbers_package_dir(resolved_target):
                raise NumbersPortfolioError(
                    f"La cible {resolved_target} est un répertoire existant qui n'est pas un paquet "
                    "Numbers valide ; précisez un chemin complet se terminant par .numbers."
                )
    elif resolved_target.is_dir():
        raise NumbersPortfolioError(
            f"La cible {resolved_target} est un répertoire (paquet) alors que la sauvegarde est en mono-fichier."
        )

    return resolved_target


def _save_package_safely(doc: Any, resolved_target: Path) -> None:
    if resolved_target.exists():
        if not resolved_target.is_dir():
            raise NumbersPortfolioError(
                f"La cible {resolved_target} est un fichier alors que la sauvegarde est au format paquet."
            )
        if not _is_numbers_package_dir(resolved_target):
            raise NumbersPortfolioError(
                f"La cible {resolved_target} est un répertoire existant qui n'est pas un paquet "
                "Numbers valide ; précisez un chemin complet se terminant par .numbers."
            )

    staging_dir = Path(
        tempfile.mkdtemp(
            dir=resolved_target.parent,
            prefix=f".{resolved_target.stem}.",
        )
    )
    pkg_name = (
        resolved_target.name
        if resolved_target.suffix.lower() == ".numbers"
        else f"{resolved_target.name}.numbers"
    )
    new_pkg = staging_dir / pkg_name
    backup_path = staging_dir / "backup.numbers"
    moved_existing = False
    preserve_backup = False
    try:
        doc.save(str(new_pkg), package=True)
        if resolved_target.exists():
            # Re-vérifie juste avant le remplacement pour éviter d'effacer un répertoire non-Numbers.
            if not _is_numbers_package_dir(resolved_target):
                raise NumbersPortfolioError(
                    f"La cible {resolved_target} n'est pas un paquet Numbers valide ; "
                    "remplacement annulé."
                )
            shutil.copymode(resolved_target, new_pkg)
            os.rename(resolved_target, backup_path)
            moved_existing = True
        os.rename(new_pkg, resolved_target)
    except BaseException as exc:
        if moved_existing and backup_path.exists() and not resolved_target.exists():
            try:
                os.rename(backup_path, resolved_target)
            except OSError as rollback_exc:
                # Préserve impérativement le dossier temporaire contenant l'unique copie de l'original.
                preserve_backup = True
                if isinstance(exc, Exception):
                    raise NumbersPortfolioError(
                        f"Impossible de remplacer le paquet Numbers {resolved_target} ({exc}) "
                        f"et échec de restauration ({rollback_exc}). "
                        f"Sauvegarde originale conservée dans : {backup_path}"
                    ) from exc
                logger.error(
                    "Interruption pendant la sauvegarde de %s et échec de restauration (%s). "
                    "Sauvegarde originale conservée dans : %s",
                    resolved_target,
                    rollback_exc,
                    backup_path,
                )
                raise
        raise
    finally:
        if not preserve_backup:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _save_document_atomically(
    doc: Any,
    target_path: Path,
    *,
    package: bool = False,
) -> None:
    resolved_target = _validate_save_target(target_path, package=package)
    tmp_path: Path | None = None
    replaced = False
    try:
        resolved_target.parent.mkdir(parents=True, exist_ok=True)
        if package:
            _save_package_safely(doc, resolved_target)
            return

        fd, tmp_name = tempfile.mkstemp(
            dir=resolved_target.parent,
            prefix=f".{resolved_target.stem}.",
            suffix=".numbers",
        )
        tmp_path = Path(tmp_name)
        os.close(fd)
        doc.save(str(tmp_path))
        if resolved_target.is_file():
            shutil.copymode(resolved_target, tmp_path)
        os.replace(tmp_path, resolved_target)
        replaced = True
    except NumbersPortfolioError:
        raise
    except Exception as exc:
        raise NumbersPortfolioError(
            f"Impossible d'enregistrer le fichier Numbers {target_path} : {exc}"
        ) from exc
    finally:
        if tmp_path is not None and not replaced:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass


def read_portfolio_rows(
    input_path: Path,
    *,
    sheet_name: str | None = None,
    table_name: str | None = None,
) -> list[PortfolioRow]:
    """Lit les lignes d'actions non vides de la colonne 0 sans modifier le fichier Numbers."""
    doc = _open_document(Path(input_path))
    _, table = _select_sheet_and_table(doc, sheet_name=sheet_name, table_name=table_name)
    return _extract_rows_from_table(table)


def _find_protected_formula_columns(
    table: Any,
    row_index: int,
    columns: tuple[tuple[str, int], ...],
) -> list[str]:
    return [
        label for label, col_idx in columns if _cell_has_formula(table, row_index, col_idx)
    ]


def _find_protected_merged_columns(
    table: Any,
    row_index: int,
    columns: tuple[tuple[str, int], ...],
) -> list[str]:
    return [
        label for label, col_idx in columns if _cell_is_merged(table, row_index, col_idx)
    ]


def update_numbers_portfolio(
    input_path: Path,
    *,
    provider: MarketDataProvider,
    evaluator: StockEvaluator,
    output_path: Path | None = None,
    sheet_name: str | None = None,
    table_name: str | None = None,
    force_refresh: bool = False,
    on_row_processed: ProgressCallback | None = None,
) -> PortfolioUpdateResult:
    """Met à jour les colonnes Note, Commentaire et Décision d'un fichier portefeuille Numbers."""
    source_path = Path(input_path)
    target_path = Path(output_path) if output_path is not None else source_path
    is_package = source_path.is_dir()

    doc = _open_document(source_path)
    _validate_save_target(target_path, package=is_package, source_path=source_path)
    sheet, table = _select_sheet_and_table(doc, sheet_name=sheet_name, table_name=table_name)
    header_row_idx, note_col, comment_col, decision_col, needs_decision_header = _locate_columns(
        table
    )
    data_start_row = max(header_row_idx + 1, _table_header_row_count(table))
    portfolio_rows = _extract_rows_from_table(table, start_row=data_start_row)

    target_columns = (
        ("Note", note_col),
        ("Commentaire", comment_col),
        (DECISION_HEADER, decision_col),
    )

    decision_column_created = False
    updated_count = 0
    results: list[PortfolioRowResult] = []
    # Évite de ré-interroger yfinance et Gemini pour un symbole présent sur plusieurs lignes.
    symbol_cache: dict[str, tuple[StockEvaluation, str] | str] = {}
    total_rows = len(portfolio_rows)
    mutating_row = False

    try:
        for position, row in enumerate(portfolio_rows, start=1):
            formula_cols = _find_protected_formula_columns(table, row.row_index, target_columns)
            merged_cols = _find_protected_merged_columns(table, row.row_index, target_columns)
            if formula_cols or merged_cols:
                reasons: list[str] = []
                if formula_cols:
                    reasons.append(
                        f"formule détectée dans la/les colonne(s) cible(s) ({', '.join(formula_cols)})"
                    )
                if merged_cols:
                    reasons.append(
                        "cellule fusionnée détectée dans la/les colonne(s) cible(s) "
                        f"({', '.join(merged_cols)})"
                    )
                row_result = PortfolioRowResult(
                    row_index=row.row_index,
                    symbol=row.symbol,
                    error=f"Ligne {row.row_index} ignorée : {' ; '.join(reasons)}.",
                )
                results.append(row_result)
                if on_row_processed is not None:
                    on_row_processed(row_result, position, total_rows)
                continue

            if row.symbol not in symbol_cache:
                try:
                    stock = provider.get(row.symbol, force_refresh=force_refresh)
                    evaluation = evaluator.evaluate(stock)
                    symbol_cache[row.symbol] = (evaluation, format_verdict(evaluation.verdict))
                except (MarketDataError, EvaluationError) as exc:
                    symbol_cache[row.symbol] = str(exc)

            cached_outcome = symbol_cache[row.symbol]
            if isinstance(cached_outcome, str):
                row_result = PortfolioRowResult(
                    row_index=row.row_index,
                    symbol=row.symbol,
                    error=cached_outcome,
                )
                results.append(row_result)
                if on_row_processed is not None:
                    on_row_processed(row_result, position, total_rows)
                continue

            evaluation, decision_text = cached_outcome
            # Toute erreur ou interruption pendant l'écriture empêche de sauvegarder un document partiellement muté.
            mutating_row = True
            try:
                if needs_decision_header and not decision_column_created:
                    table.write(header_row_idx, decision_col, DECISION_HEADER)
                    decision_column_created = True

                table.write(row.row_index, note_col, evaluation.score)
                table.write(row.row_index, comment_col, evaluation.comment)
                table.write(row.row_index, decision_col, decision_text)
            except Exception as exc:
                raise NumbersPortfolioError(
                    f"Erreur lors de l'écriture de la ligne {row.row_index} ({row.symbol}) : {exc}"
                ) from exc
            # Ne réinitialise mutating_row qu'après succès complet (jamais dans finally, exécuté sur KeyboardInterrupt).
            mutating_row = False

            updated_count += 1
            row_result = PortfolioRowResult(
                row_index=row.row_index,
                symbol=row.symbol,
                score=evaluation.score,
                comment=evaluation.comment,
                decision=decision_text,
            )
            results.append(row_result)
            if on_row_processed is not None:
                on_row_processed(row_result, position, total_rows)
    except KeyboardInterrupt:
        # Préserve les lignes déjà entièrement mises à jour uniquement si aucune écriture de ligne n'était en cours.
        if mutating_row:
            logger.warning(
                "Interruption en cours d'écriture sur %s ; sauvegarde annulée pour éviter toute corruption.",
                target_path,
            )
        elif updated_count > 0:
            _save_document_atomically(doc, target_path, package=is_package)
        raise

    # Évite de ré-sérialiser le classeur (potentiellement destructif sur les styles) si rien n'a réussi.
    saved = updated_count > 0
    if saved:
        _save_document_atomically(doc, target_path, package=is_package)

    return PortfolioUpdateResult(
        input_path=source_path,
        output_path=target_path,
        sheet_name=sheet.name,
        table_name=table.name,
        decision_column_created=decision_column_created,
        saved=saved,
        results=results,
    )
