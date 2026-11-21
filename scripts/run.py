#!/usr/bin/env python3
"""Script de test réel sur l'API yfinance avec mise en cache locale."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# Relance le script dans le .venv du projet si exécuté avec le Python système.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

sys.path.insert(0, str(ROOT / "src"))

from portfolio_tracker.market_data import MarketDataError, YFinanceMarketDataProvider  # noqa: E402

DEFAULT_SYMBOLS = ["NVDA", "AAPL", "AMZN", "META"]


def main() -> int:
    symbols = sys.argv[1:] or DEFAULT_SYMBOLS
    provider = YFinanceMarketDataProvider()

    for symbol in symbols:
        print(f"\n--- Récupération de {symbol} ---")
        try:
            stock = provider.get(symbol)
            print(json.dumps(stock.model_dump(mode="json"), indent=2, ensure_ascii=False))
        except MarketDataError as exc:
            print(f"[ERREUR] {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
