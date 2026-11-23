import json
import logging
import math
import os
import tempfile
import threading
import time
import urllib.error
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf
from pydantic import ValidationError

from portfolio_tracker.contracts import StockData

logger = logging.getLogger(__name__)

CACHE_PATH_ENV_VAR = "PORTFOLIO_TRACKER_CACHE_PATH"
DEFAULT_CACHE_TTL = timedelta(hours=24)
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
# Légèrement supérieur au timeout interne de yfinance (30s) pour laisser les sockets se fermer proprement.
DEFAULT_TIMEOUT_SECONDS = 35.0
DEFAULT_MAX_STALLED_WORKERS = 6
MAX_TIMEOUT_RETRIES_PER_SYMBOL = 1


def _resolve_default_cache_path() -> Path:
    """Résout le chemin du cache depuis l'environnement, le dépôt source ou le dossier courant."""
    env_override = os.environ.get(CACHE_PATH_ENV_VAR, "").strip()
    if env_override:
        return Path(env_override).expanduser()

    repo_root = Path(__file__).resolve().parents[2]
    # En installation editable depuis le dépôt, on conserve le dossier cache/ à la racine du projet.
    if (repo_root / "pyproject.toml").is_file():
        return repo_root / "cache" / "stocks_cache.json"
    return Path.cwd() / "cache" / "stocks_cache.json"


DEFAULT_CACHE_PATH = _resolve_default_cache_path()


class MarketDataError(Exception):
    """Erreur générale lors de la récupération des données boursières."""


class StockNotFoundError(MarketDataError):
    """Levée lorsqu'un symbole boursier est vide ou introuvable."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class YFinanceMarketDataProvider:
    """Récupère les données financières via yfinance avec mise en cache JSON locale."""

    def __init__(
        self,
        cache_path: Path | None = None,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        cache_ttl: timedelta = DEFAULT_CACHE_TTL,
        clock: Callable[[], datetime] = _utc_now,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_stalled_workers: int = DEFAULT_MAX_STALLED_WORKERS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(cache_ttl, timedelta) or cache_ttl < timedelta(0):
            raise MarketDataError("cache_ttl doit être une durée supérieure ou égale à 0.")
        if isinstance(max_retries, bool) or max_retries < 0:
            raise MarketDataError("max_retries doit être un entier supérieur ou égal à 0.")
        if (
            isinstance(backoff_seconds, bool)
            or not math.isfinite(backoff_seconds)
            or backoff_seconds < 0
        ):
            raise MarketDataError(
                "backoff_seconds doit être un nombre fini supérieur ou égal à 0."
            )
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise MarketDataError("timeout doit être un nombre fini strictement positif.")
        if isinstance(max_stalled_workers, bool) or max_stalled_workers < 1:
            raise MarketDataError("max_stalled_workers doit être un entier supérieur ou égal à 1.")
        self._cache_path = cache_path if cache_path is not None else _resolve_default_cache_path()
        self._ticker_factory = ticker_factory
        self._cache_ttl = cache_ttl
        self._clock = clock
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._timeout = float(timeout)
        self._max_stalled_workers = max_stalled_workers
        self._stalled_threads: list[threading.Thread] = []
        self._stalled_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._sleep = sleep
        if ticker_factory is yf.Ticker:
            _ensure_yfinance_exceptions_visible()

    def get(self, symbol: str, *, force_refresh: bool = False) -> StockData:
        normalized = symbol.strip().upper()
        if not normalized:
            raise StockNotFoundError("Le symbole boursier ne peut pas être vide.")

        now = self._clock()
        # Normalise en UTC si une horloge personnalisée renvoie un datetime naïf.
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        if not force_refresh:
            with self._cache_lock:
                cache = self._load_cache()
            if normalized in cache:
                cached_stock = self._read_fresh_entry(cache[normalized], now)
                if cached_stock is not None:
                    logger.debug("Cache hit pour %s.", normalized)
                    return cached_stock

        logger.info("Récupération yfinance pour %s (force_refresh=%s).", normalized, force_refresh)
        stock = self._fetch_from_yfinance(normalized)
        with self._cache_lock:
            latest_cache, can_save = self._read_cache()
            if can_save:
                latest_cache[normalized] = {
                    "fetched_at": now.isoformat(),
                    "data": stock.model_dump(mode="json"),
                }
                self._save_cache(latest_cache)
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
            fetched_at = fetched_at.replace(tzinfo=UTC)
        age = now - fetched_at
        # Rejette les dates futures (dérive d'horloge) autant que les entrées expirées.
        if age < timedelta(0) or age >= self._cache_ttl:
            return None
        try:
            return StockData.model_validate(data)
        except ValidationError:
            return None

    def _active_stalled_count(self) -> int:
        with self._stalled_lock:
            self._stalled_threads = [t for t in self._stalled_threads if t.is_alive()]
            return len(self._stalled_threads)

    def _record_stalled_thread(self, thread: threading.Thread) -> None:
        with self._stalled_lock:
            self._stalled_threads = [t for t in self._stalled_threads if t.is_alive()]
            self._stalled_threads.append(thread)

    def _run_with_timeout(self, fn: Callable[[], Any]) -> Any:
        stalled_count = self._active_stalled_count()
        if stalled_count >= self._max_stalled_workers:
            raise MarketDataError(
                f"Trop d'appels yfinance déjà bloqués en arrière-plan "
                f"({stalled_count}/{self._max_stalled_workers})."
            )
        return _call_with_timeout(
            fn,
            self._timeout,
            on_timeout=self._record_stalled_thread,
        )

    def _fetch_from_yfinance(self, symbol: str) -> StockData:
        _ensure_yfinance_exceptions_visible()
        raw_info: Any = None
        timeout_retries_used = 0
        max_timeout_retries = min(self._max_retries, MAX_TIMEOUT_RETRIES_PER_SYMBOL)
        for attempt in range(self._max_retries + 1):
            try:
                raw_info = self._run_with_timeout(lambda: self._ticker_factory(symbol).info)
                break
            except MarketDataError:
                raise
            except (TypeError, AttributeError, KeyError) as exc:
                # Les erreurs de programmation ou de structure interne ne sont pas transitoires.
                raise MarketDataError(
                    f"Erreur lors de la récupération de {symbol} : {exc}"
                ) from exc
            except TimeoutError as exc:
                # Plafonne les retries après timeout pour éviter qu'un seul ticker lent ne sature tous les workers.
                can_retry_timeout = (
                    attempt < self._max_retries
                    and timeout_retries_used < max_timeout_retries
                    and self._active_stalled_count() < self._max_stalled_workers
                )
                if can_retry_timeout:
                    timeout_retries_used += 1
                    delay = self._backoff_seconds * (2**attempt)
                    logger.warning(
                        "Timeout yfinance (%ss) pour %s (tentative %d/%d), nouvelle tentative dans %.1fs.",
                        self._timeout,
                        symbol,
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                    )
                    self._sleep(delay)
                    continue
                raise MarketDataError(
                    f"Délai d'attente dépassé ({self._timeout}s) lors de la récupération de {symbol}."
                ) from exc
            except Exception as exc:
                status_code = _extract_http_status_code(exc)
                if status_code == 404:
                    raise StockNotFoundError(
                        f"Aucune donnée trouvée pour le symbole {symbol}."
                    ) from exc
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise MarketDataError(
                        f"Erreur HTTP {status_code} lors de la récupération de {symbol} : {exc}"
                    ) from exc
                if attempt < self._max_retries:
                    delay = self._backoff_seconds * (2**attempt)
                    logger.warning(
                        "Erreur lors de la récupération yfinance de %s (tentative %d/%d) : %s — nouvelle tentative dans %.1fs.",
                        symbol,
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        delay,
                    )
                    self._sleep(delay)
                    continue
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
        name = _extract_name(info)
        # yfinance renvoie souvent un dict quasi vide (ou un nom blanc) pour un ticker inconnu au lieu de lever une erreur.
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

    def _read_cache(self) -> tuple[dict[str, Any], bool]:
        if not self._cache_path.exists():
            return {}, True
        try:
            raw_text = self._cache_path.read_text(encoding="utf-8")
        except OSError as exc:
            # Ne pas écraser un cache existant simplement illisible temporairement.
            logger.warning("Impossible de charger le cache %s : %s", self._cache_path, exc)
            return {}, False
        except UnicodeDecodeError as exc:
            logger.warning("Impossible de charger le cache %s : %s", self._cache_path, exc)
            return {}, True
        try:
            data = json.loads(raw_text, parse_constant=_reject_non_finite_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Impossible de charger le cache %s : %s", self._cache_path, exc)
            return {}, True
        if not isinstance(data, dict):
            logger.warning("Format de cache invalide dans %s (objet JSON attendu).", self._cache_path)
            return {}, True
        return data, True

    def _load_cache(self) -> dict[str, Any]:
        cache, _ = self._read_cache()
        return cache

    def _save_cache(self, cache: dict[str, Any]) -> None:
        try:
            payload = json.dumps(cache, indent=2, ensure_ascii=False, allow_nan=False)
        except ValueError as exc:
            logger.warning("Impossible d'enregistrer le cache %s : %s", self._cache_path, exc)
            return
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
        except OSError as exc:
            # Le cache reste une optimisation : un échec disque ne doit pas bloquer get().
            logger.warning("Impossible d'enregistrer le cache %s : %s", self._cache_path, exc)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _reject_non_finite_json_constant(value: str) -> Any:
    """Rejette NaN/Infinity dans le cache JSON afin de déclencher une auto-réparation."""
    raise ValueError(f"Constante JSON non finie refusée : {value}")


def _extract_name(info: dict[str, Any]) -> Any:
    """Retourne le nom court ou long en ignorant les chaînes vides/blanches."""
    for key in ("shortName", "longName"):
        raw = info.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            cleaned = raw.strip()
            if cleaned:
                return cleaned
            continue
        return raw
    return None


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


def _call_with_timeout(
    fn: Callable[[], Any],
    timeout: float,
    *,
    on_timeout: Callable[[threading.Thread], None] | None = None,
) -> Any:
    """Exécute un appel bloquant yfinance dans un thread daemon sans bloquer la sortie sur timeout."""
    done = threading.Event()
    outcome: list[Any] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            outcome.append(fn())
        except BaseException as exc:
            error.append(exc)
        finally:
            done.set()

    # Thread daemon : un appel réseau figé ne doit pas bloquer atexit à la fermeture du processus.
    thread = threading.Thread(target=_worker, name="yfinance-fetch", daemon=True)
    thread.start()
    if not done.wait(timeout=timeout):
        if on_timeout is not None:
            on_timeout(thread)
        raise TimeoutError(f"Délai d'attente dépassé ({timeout}s).")
    if error:
        raise error[0]
    return outcome[0]


def _ensure_yfinance_exceptions_visible() -> None:
    """Désactive hide_exceptions dans yfinance (>=1.x) afin que les erreurs HTTP remontent."""
    config = getattr(yf, "config", None)
    debug_cfg = getattr(config, "debug", None)
    if debug_cfg is not None and getattr(debug_cfg, "hide_exceptions", False):
        debug_cfg.hide_exceptions = False


def _extract_http_status_code(exc: BaseException) -> int | None:
    """Extrait un code HTTP depuis urllib, requests, curl_cffi ou yfinance sans coupler au backend."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    rate_limit_cls = getattr(getattr(yf, "exceptions", None), "YFRateLimitError", None)
    if isinstance(rate_limit_cls, type) and isinstance(exc, rate_limit_cls):
        return 429
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code
    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599:
        return code
    return None
