from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from pydantic import ValidationError
import yfinance as yf

from portfolio_tracker.contracts import StockData

DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "stocks_cache.json"
DEFAULT_CACHE_TTL = timedelta(hours=24)


class MarketDataError(Exception):
    """Erreur générale lors de la récupération des données boursières."""


class StockNotFoundError(MarketDataError):
    """Levée lorsqu'un symbole boursier est vide ou introuvable."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class YFinanceMarketDataProvider:
    """Récupère les données financières via yfinance avec mise en cache JSON locale."""

    def __init__(
        self,
        cache_path: Path = DEFAULT_CACHE_PATH,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        cache_ttl: timedelta = DEFAULT_CACHE_TTL,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._cache_path = cache_path
        self._ticker_factory = ticker_factory
        self._cache_ttl = cache_ttl
        self._clock = clock

    def get(self, symbol: str, *, force_refresh: bool = False) -> StockData:
        normalized = symbol.strip().upper()
        if not normalized:
            raise StockNotFoundError("Le symbole boursier ne peut pas être vide.")

        now = self._clock()
        # Normalise en UTC si une horloge personnalisée renvoie un datetime naïf.
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cache = self._load_cache()
        if not force_refresh and normalized in cache:
            cached_stock = self._read_fresh_entry(cache[normalized], now)
            if cached_stock is not None:
                return cached_stock

        stock = self._fetch_from_yfinance(normalized)
        cache[normalized] = {
            "fetched_at": now.isoformat(),
            "data": stock.model_dump(mode="json"),
        }
        self._save_cache(cache)
        return stock

    def _read_fresh_entry(self, entry: Any, now: datetime) -> StockData | None:
        if not isinstance(entry, dict):
            return None
        fetched_at_raw = entry.get("fetched_at")
        data = entry.get("data")
        if not isinstance(fetched_at_raw, str) or not isinstance(data, dict):
            return None
        try:
            fetched_at = datetime.fromisoformat(fetched_at_raw)
        except ValueError:
            return None
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age = now - fetched_at
        # Rejette les dates futures (dérive d'horloge) autant que les entrées expirées.
        if age < timedelta(0) or age >= self._cache_ttl:
            return None
        try:
            return StockData.model_validate(data)
        except ValidationError:
            return None

    def _fetch_from_yfinance(self, symbol: str) -> StockData:
        try:
            raw_info = self._ticker_factory(symbol).info
        except Exception as exc:
            raise MarketDataError(
                f"Erreur lors de la récupération de {symbol} : {exc}"
            ) from exc

        if raw_info is None:
            info: dict[str, Any] = {}
        elif not isinstance(raw_info, dict):
            raise MarketDataError(
                f"Format de données inattendu pour {symbol} : {type(raw_info).__name__}."
            )
        else:
            info = raw_info

        price = _extract_price(info)
        name = info.get("shortName") or info.get("longName")
        # yfinance renvoie souvent un dict quasi vide pour un ticker inconnu au lieu de lever une erreur.
        if price is None and not name:
            raise StockNotFoundError(f"Aucune donnée trouvée pour le symbole {symbol}.")

        try:
            return StockData(
                symbol=symbol,
                name=name or symbol,
                currency=info.get("currency"),
                price=price,
                fifty_two_week_low=_to_float(info.get("fiftyTwoWeekLow")),
                fifty_two_week_high=_to_float(info.get("fiftyTwoWeekHigh")),
                market_cap=_to_float(info.get("marketCap")),
                trailing_pe=_to_float(info.get("trailingPE")),
                forward_pe=_to_float(info.get("forwardPE")),
                debt_to_equity=_to_float(info.get("debtToEquity")),
                profit_margins=_to_float(info.get("profitMargins")),
                revenue_growth=_to_float(info.get("revenueGrowth")),
                recommendation_key=info.get("recommendationKey"),
            )
        except ValidationError as exc:
            raise MarketDataError(
                f"Données invalides reçues pour {symbol} : {exc}"
            ) from exc

    def _load_cache(self) -> dict[str, Any]:
        if not self._cache_path.exists():
            return {}
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        payload = json.dumps(cache, indent=2, ensure_ascii=False)
        tmp_path: str | None = None
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Écriture dans un fichier temporaire du même dossier puis remplacement atomique.
            fd, tmp_path = tempfile.mkstemp(
                dir=self._cache_path.parent,
                prefix=f".{self._cache_path.name}.",
                suffix=".tmp",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, self._cache_path)
        except OSError:
            # Le cache reste une optimisation : un échec disque ne doit pas bloquer get().
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _extract_price(info: dict[str, Any]) -> float | None:
    # Ordre de repli : cours temps réel/régulier, valeur liquidative (ETF), puis clôture veille.
    for key in ("currentPrice", "regularMarketPrice", "navPrice", "previousClose"):
        value = _to_float(info.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    # Exclut les booléens (sous-classe de int en Python) et les valeurs NaN/Inf non sérialisables en JSON standard.
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
