#!/usr/bin/env python3
"""Script CLI de récupération yfinance (avec cache local) et d'évaluation Gemini."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _reexec_in_venv_if_needed() -> None:
    """Relance le script dans le .venv du projet si exécuté avec un autre interpréteur."""
    if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


if __name__ == "__main__":
    # Ré-exécute dans le .venv avant d'importer portfolio_tracker (pydantic/yfinance).
    _reexec_in_venv_if_needed()

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from portfolio_tracker.contracts import PortfolioRowResult  # noqa: E402
from portfolio_tracker.evaluator import EvaluationError, GeminiStockEvaluator  # noqa: E402
from portfolio_tracker.market_data import MarketDataError, YFinanceMarketDataProvider  # noqa: E402
from portfolio_tracker.numbers_portfolio import (  # noqa: E402
    NumbersPortfolioError,
    update_numbers_portfolio,
)

DEFAULT_SYMBOLS = ["NVDA", "AAPL", "AMZN", "META"]


class _CliStreamHandler(logging.StreamHandler[TextIO]):
    """Handler stderr dédié à la CLI qui suit dynamiquement sys.stderr sans doublon."""

    def flush(self) -> None:
        self.stream = sys.stderr
        if self.stream and not getattr(self.stream, "closed", False):
            super().flush()

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def _configure_logging(*, verbose: bool = False) -> None:
    """Configure les journaux de portfolio_tracker sur stderr de façon idempotente."""
    pkg_logger = logging.getLogger("portfolio_tracker")
    pkg_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if not any(isinstance(h, _CliStreamHandler) for h in pkg_logger.handlers):
        handler = _CliStreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        pkg_logger.addHandler(handler)


def _resolve_cli_model(cli_model: str | None) -> str | None:
    """Résout le modèle Gemini demandé via --model ou la variable d'environnement GEMINI_MODEL."""
    if cli_model is not None:
        return cli_model
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    return env_model or None


def _build_evaluator(cli_model: str | None) -> GeminiStockEvaluator:
    selected_model = _resolve_cli_model(cli_model)
    if selected_model is not None:
        return GeminiStockEvaluator(model=selected_model)
    return GeminiStockEvaluator()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Récupère les données boursières via yfinance et lance l'évaluation Gemini."
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help=f"Symboles boursiers à analyser (défaut : {', '.join(DEFAULT_SYMBOLS)}).",
    )
    parser.add_argument(
        "--numbers-file",
        type=Path,
        default=None,
        help="Chemin vers un fichier Apple Numbers (.numbers) de suivi de portefeuille à mettre à jour.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin de sortie optionnel pour sauvegarder le fichier Numbers mis à jour.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Nom de la feuille Apple Numbers à traiter (défaut : première feuille).",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Nom du tableau Apple Numbers à traiter (défaut : premier tableau de la feuille).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modèle Gemini à utiliser (surcharge GEMINI_MODEL et le modèle par défaut).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore le cache local et force un rafraîchissement depuis yfinance.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Désactive l'évaluation qualitative via Gemini.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Active les journaux détaillés (cache, retries, appels API) sur stderr.",
    )
    return parser


def _run_numbers_mode(args: argparse.Namespace) -> int:
    if args.no_eval:
        print(
            "[ERREUR] L'option --no-eval est incompatible avec --numbers-file.",
            file=sys.stderr,
        )
        return 1

    provider = YFinanceMarketDataProvider()
    try:
        evaluator = _build_evaluator(args.model)
    except EvaluationError as exc:
        print(f"[ERREUR] Évaluation Gemini indisponible : {exc}", file=sys.stderr)
        return 1

    completed_rows = 0

    def _report_progress(item: PortfolioRowResult, index: int, total: int) -> None:
        nonlocal completed_rows
        if item.ok:
            completed_rows += 1
            print(
                f"[{index}/{total}] {item.symbol} — Note : {item.score}/100 | {item.decision}",
                file=sys.stderr,
            )
        else:
            print(f"[{index}/{total}] {item.symbol} — ERREUR : {item.error}", file=sys.stderr)

    try:
        update_result = update_numbers_portfolio(
            args.numbers_file,
            provider=provider,
            evaluator=evaluator,
            output_path=args.output,
            sheet_name=args.sheet,
            table_name=args.table,
            force_refresh=args.force_refresh,
            on_row_processed=_report_progress,
        )
    except KeyboardInterrupt:
        if completed_rows > 0:
            print(
                f"\n[INTERROMPU] Traitement interrompu après {completed_rows} ligne(s) finalisée(s) "
                "(sauvegardées sauf interruption en cours d'écriture).",
                file=sys.stderr,
            )
        else:
            print(
                "\n[INTERROMPU] Traitement interrompu ; aucune ligne n'a été modifiée.",
                file=sys.stderr,
            )
        return 130
    except NumbersPortfolioError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1

    if update_result.saved:
        print(
            f"Fichier sauvegardé : {update_result.output_path} "
            f"(feuille « {update_result.sheet_name} », tableau « {update_result.table_name} »)"
        )
    else:
        print(
            f"Aucune ligne mise à jour ; fichier non modifié : {update_result.output_path} "
            f"(feuille « {update_result.sheet_name} », tableau « {update_result.table_name} »)"
        )
    if update_result.decision_column_created:
        print("Colonne « Décision » créée automatiquement.")

    for item in update_result.results:
        if item.ok:
            print(
                f"  [OK] {item.symbol} — "
                f"Note : {item.score}/100 | Décision : {item.decision} | {item.comment}"
            )
        else:
            print(
                f"  [ERREUR] {item.symbol} : {item.error}",
                file=sys.stderr,
            )

    print(
        f"Total : {len(update_result.succeeded)} succès, {len(update_result.failed)} erreur(s)."
    )
    return 1 if update_result.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    if args.output is not None and args.numbers_file is None:
        parser.error("--output ne peut être utilisé qu'avec --numbers-file.")
    if (args.sheet is not None or args.table is not None) and args.numbers_file is None:
        parser.error("--sheet et --table ne peuvent être utilisés qu'avec --numbers-file.")
    if args.numbers_file is not None and args.symbols:
        parser.error("--numbers-file ne peut pas être combiné avec des symboles positionnels.")

    if args.numbers_file is not None:
        return _run_numbers_mode(args)

    symbols: list[str] = args.symbols or DEFAULT_SYMBOLS

    provider = YFinanceMarketDataProvider()
    evaluator: GeminiStockEvaluator | None = None
    if not args.no_eval:
        try:
            evaluator = _build_evaluator(args.model)
        except EvaluationError as exc:
            print(f"[AVERTISSEMENT] Évaluation Gemini désactivée : {exc}", file=sys.stderr)

    has_errors = False
    try:
        for symbol in symbols:
            print(f"\n--- Récupération de {symbol} ---")
            try:
                stock = (
                    provider.get(symbol, force_refresh=True)
                    if args.force_refresh
                    else provider.get(symbol)
                )
                print(json.dumps(stock.model_dump(mode="json"), indent=2, ensure_ascii=False))
            except MarketDataError as exc:
                print(f"[ERREUR] {exc}", file=sys.stderr)
                has_errors = True
                continue

            if evaluator is not None:
                print(f"\n--- Évaluation Gemini ({symbol}) ---")
                try:
                    evaluation = evaluator.evaluate(stock)
                    print(
                        json.dumps(evaluation.model_dump(mode="json"), indent=2, ensure_ascii=False)
                    )
                except EvaluationError as exc:
                    print(f"[ERREUR ÉVALUATION] {exc}", file=sys.stderr)
                    has_errors = True
    except KeyboardInterrupt:
        print("\n[INTERROMPU] Traitement interrompu par l'utilisateur.", file=sys.stderr)
        return 130

    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
