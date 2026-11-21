#!/usr/bin/env python3
"""Script CLI de récupération yfinance (avec cache local) et d'évaluation Gemini."""

import argparse
import json
import os
import sys
from pathlib import Path

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

from portfolio_tracker.evaluator import EvaluationError, GeminiStockEvaluator  # noqa: E402
from portfolio_tracker.market_data import MarketDataError, YFinanceMarketDataProvider  # noqa: E402

DEFAULT_SYMBOLS = ["NVDA", "AAPL", "AMZN", "META"]


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
        "--force-refresh",
        action="store_true",
        help="Ignore le cache local et force un rafraîchissement depuis yfinance.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Désactive l'évaluation qualitative via Gemini.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    symbols: list[str] = args.symbols or DEFAULT_SYMBOLS

    provider = YFinanceMarketDataProvider()
    evaluator: GeminiStockEvaluator | None = None
    if not args.no_eval:
        try:
            evaluator = GeminiStockEvaluator()
        except EvaluationError as exc:
            print(f"[AVERTISSEMENT] Évaluation Gemini désactivée : {exc}", file=sys.stderr)

    has_errors = False
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
                print(json.dumps(evaluation.model_dump(mode="json"), indent=2, ensure_ascii=False))
            except EvaluationError as exc:
                print(f"[ERREUR ÉVALUATION] {exc}", file=sys.stderr)
                has_errors = True

    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
