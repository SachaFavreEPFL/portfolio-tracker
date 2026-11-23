import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from portfolio_tracker.market_data import (
    CACHE_PATH_ENV_VAR,
    MarketDataError,
    StockNotFoundError,
    YFinanceMarketDataProvider,
)


def _make_mock_ticker(info: dict) -> MagicMock:
    ticker = MagicMock()
    ticker.info = info
    return ticker


def test_get_fetches_from_yfinance_and_writes_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache" / "stocks.json"
    mock_ticker = _make_mock_ticker(
        {
            "shortName": "NVIDIA Corporation",
            "currency": "USD",
            "currentPrice": 140.5,
            "fiftyTwoWeekLow": 60.0,
            "fiftyTwoWeekHigh": 150.0,
            "marketCap": 3_400_000_000_000,
            "trailingPE": 52.3,
            "forwardPE": 35.1,
            "debtToEquity": 17.2,
            "profitMargins": 0.55,
            "revenueGrowth": 0.94,
            "recommendationKey": "buy",
        }
    )
    ticker_factory = MagicMock(return_value=mock_ticker)
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
    )

    stock = provider.get("  nvda ")

    ticker_factory.assert_called_once_with("NVDA")
    assert stock.symbol == "NVDA"
    assert stock.name == "NVIDIA Corporation"
    assert stock.currency == "USD"
    assert stock.price == 140.5
    assert stock.fifty_two_week_low == 60.0
    assert stock.fifty_two_week_high == 150.0
    assert stock.market_cap == 3_400_000_000_000
    assert stock.trailing_pe == 52.3
    assert stock.forward_pe == 35.1
    assert stock.debt_to_equity == 17.2
    assert stock.profit_margins == 0.55
    assert stock.revenue_growth == 0.94
    assert stock.recommendation_key == "buy"

    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "NVDA" in cached
    assert "fetched_at" in cached["NVDA"]
    assert cached["NVDA"]["data"]["price"] == 140.5


def test_get_reads_from_cache_without_calling_yfinance(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    mock_ticker = _make_mock_ticker(
        {
            "shortName": "Apple Inc.",
            "currency": "USD",
            "currentPrice": 225.0,
        }
    )
    first_factory = MagicMock(return_value=mock_ticker)
    first_provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=first_factory,
    )

    first = first_provider.get("AAPL")

    # Une nouvelle instance prouve que la lecture vient bien du disque et non d'un état en mémoire.
    second_factory = MagicMock()
    second_provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=second_factory,
    )
    second = second_provider.get("aapl")

    first_factory.assert_called_once_with("AAPL")
    second_factory.assert_not_called()
    assert second == first


def test_get_raises_on_empty_or_unknown_symbol_without_caching(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    ticker_factory = MagicMock(return_value=_make_mock_ticker({}))
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
    )

    with pytest.raises(StockNotFoundError):
        provider.get("   ")

    with pytest.raises(StockNotFoundError):
        provider.get("INVALID999")

    ticker_factory.assert_called_once_with("INVALID999")
    assert not cache_file.exists()


def test_get_handles_missing_optional_fields(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    mock_ticker = _make_mock_ticker(
        {
            "longName": "TotalEnergies SE",
            "regularMarketPrice": 58.4,
        }
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(return_value=mock_ticker),
    )

    stock = provider.get("TTE.PA")

    assert stock.symbol == "TTE.PA"
    assert stock.name == "TotalEnergies SE"
    assert stock.price == 58.4
    assert stock.debt_to_equity is None
    assert stock.forward_pe is None


def test_get_refreshes_cache_after_24h_ttl(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    current_time = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
    ticker_factory = MagicMock(
        side_effect=[
            _make_mock_ticker({"shortName": "Apple Inc.", "currentPrice": 220.0}),
            _make_mock_ticker({"shortName": "Apple Inc.", "currentPrice": 235.0}),
        ]
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
        clock=lambda: current_time,
    )

    first = provider.get("AAPL")
    assert first.price == 220.0

    current_time += timedelta(hours=23, minutes=59)
    still_cached = provider.get("AAPL")
    assert still_cached.price == 220.0
    assert ticker_factory.call_count == 1

    current_time += timedelta(minutes=1)
    refreshed = provider.get("AAPL")
    assert refreshed.price == 235.0
    assert ticker_factory.call_count == 2

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["AAPL"]["data"]["price"] == 235.0
    assert cached["AAPL"]["fetched_at"] == current_time.isoformat()


def test_get_force_refresh_bypasses_fresh_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    current_time = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
    ticker_factory = MagicMock(
        side_effect=[
            _make_mock_ticker({"shortName": "Meta", "currentPrice": 600.0}),
            _make_mock_ticker({"shortName": "Meta", "currentPrice": 615.0}),
        ]
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
        clock=lambda: current_time,
    )

    assert provider.get("META").price == 600.0
    current_time += timedelta(minutes=30)
    assert provider.get("META", force_refresh=True).price == 615.0
    assert ticker_factory.call_count == 2

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["META"]["data"]["price"] == 615.0
    assert cached["META"]["fetched_at"] == current_time.isoformat()


@pytest.mark.parametrize(
    "raw_cache_bytes",
    [
        b"{invalid json",
        b"[]",
        b"null",
        b"\xff\xfe\x00\x80",
        json.dumps({"NVDA": {"symbol": "NVDA", "name": "Legacy", "price": 100.0}}).encode(),
        json.dumps({"NVDA": {"fetched_at": "not-a-date", "data": {"symbol": "NVDA"}}}).encode(),
        json.dumps({"NVDA": {"fetched_at": "2026-11-17T10:00:00+00:00", "data": {"price": "bad"}}}).encode(),
    ],
)
def test_get_recovers_from_corrupted_or_legacy_cache(
    tmp_path: Path, raw_cache_bytes: bytes
) -> None:
    cache_file = tmp_path / "stocks.json"
    cache_file.write_bytes(raw_cache_bytes)
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    ticker_factory = MagicMock(
        return_value=_make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 150.0})
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
        clock=lambda: now,
    )

    stock = provider.get("NVDA")

    assert stock.price == 150.0
    ticker_factory.assert_called_once_with("NVDA")
    healed = json.loads(cache_file.read_text(encoding="utf-8"))
    assert healed["NVDA"]["data"]["price"] == 150.0
    assert healed["NVDA"]["fetched_at"] == now.isoformat()


def test_get_preserves_other_valid_entries_when_healing_one_symbol(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    cache_file.write_text(
        json.dumps(
            {
                "AAPL": {
                    "fetched_at": now.isoformat(),
                    "data": {"symbol": "AAPL", "name": "Apple Inc.", "price": 225.0},
                },
                "NVDA": {"fetched_at": "invalid-date", "data": {}},
            }
        ),
        encoding="utf-8",
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 150.0})
        ),
        clock=lambda: now,
    )

    assert provider.get("NVDA").price == 150.0
    healed = json.loads(cache_file.read_text(encoding="utf-8"))
    assert healed["AAPL"]["data"]["price"] == 225.0
    assert healed["NVDA"]["data"]["price"] == 150.0


def test_get_raises_market_data_error_on_api_failure(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    provider_factory_err = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(side_effect=RuntimeError("network timeout")),
        max_retries=0,
    )

    with pytest.raises(MarketDataError) as exc_info:
        provider_factory_err.get("NVDA")
    assert not isinstance(exc_info.value, StockNotFoundError)
    assert not cache_file.exists()

    # Simule une panne réseau lors de l'accès paresseux à la propriété .info de yfinance.
    failing_ticker = MagicMock()
    type(failing_ticker).info = PropertyMock(side_effect=RuntimeError("http 503"))
    provider_info_err = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(return_value=failing_ticker),
        max_retries=0,
    )

    with pytest.raises(MarketDataError) as exc_info:
        provider_info_err.get("NVDA")
    assert not isinstance(exc_info.value, StockNotFoundError)
    assert not cache_file.exists()


def test_get_preserves_expired_cache_when_api_fails(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    expired_ts = "2026-11-15T10:00:00+00:00"
    original_payload = {
        "NVDA": {
            "fetched_at": expired_ts,
            "data": {"symbol": "NVDA", "name": "NVIDIA", "price": 140.0},
        }
    }
    cache_file.write_text(json.dumps(original_payload), encoding="utf-8")
    failing_ticker = MagicMock()
    type(failing_ticker).info = PropertyMock(side_effect=RuntimeError("offline"))
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(return_value=failing_ticker),
        clock=lambda: datetime(2026, 11, 17, 12, 0, tzinfo=UTC),
        max_retries=0,
    )

    with pytest.raises(MarketDataError):
        provider.get("NVDA")

    assert json.loads(cache_file.read_text(encoding="utf-8")) == original_payload


def test_get_ignores_non_finite_values_and_uses_price_fallbacks(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    mock_ticker = _make_mock_ticker(
        {
            "shortName": "ETF World",
            "currentPrice": float("nan"),
            "regularMarketPrice": float("inf"),
            "navPrice": 102.5,
            "trailingPE": "N/A",
            "forwardPE": True,
        }
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(return_value=mock_ticker),
    )

    stock = provider.get("CW8.PA")

    assert stock.price == 102.5
    assert stock.trailing_pe is None
    assert stock.forward_pe is None


def test_get_returns_stock_even_if_cache_write_fails(tmp_path: Path) -> None:
    # Crée un fichier à la place du dossier parent pour provoquer un OSError lors de mkdir/mkstemp.
    blocked_parent = tmp_path / "not_a_dir"
    blocked_parent.write_text("file", encoding="utf-8")
    cache_file = blocked_parent / "stocks.json"
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "Apple Inc.", "currentPrice": 225.0})
        ),
    )

    stock = provider.get("AAPL")

    assert stock.symbol == "AAPL"
    assert stock.price == 225.0


def test_get_wraps_pydantic_validation_error_in_market_data_error(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker(
                {"shortName": "Broken Corp", "currentPrice": 10.0, "currency": 123}
            )
        ),
    )

    with pytest.raises(MarketDataError) as exc_info:
        provider.get("BRK")

    assert not isinstance(exc_info.value, StockNotFoundError)
    assert not cache_file.exists()


def test_get_raises_market_data_error_when_info_is_not_dict(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    ticker = MagicMock()
    ticker.info = ["unexpected", "list"]
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(return_value=ticker),
    )

    with pytest.raises(MarketDataError) as exc_info:
        provider.get("NVDA")

    assert not isinstance(exc_info.value, StockNotFoundError)
    assert not cache_file.exists()


def test_get_handles_naive_clock_and_rejects_future_cache_timestamp(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    naive_now = datetime(2026, 11, 17, 12, 0)
    future_ts = (naive_now.replace(tzinfo=UTC) + timedelta(hours=1)).isoformat()
    cache_file.write_text(
        json.dumps(
            {
                "NVDA": {
                    "fetched_at": future_ts,
                    "data": {"symbol": "NVDA", "name": "Stale Future", "price": 99.0},
                }
            }
        ),
        encoding="utf-8",
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 155.0})
        ),
        clock=lambda: naive_now,
    )

    stock = provider.get("NVDA")

    assert stock.price == 155.0
    healed = json.loads(cache_file.read_text(encoding="utf-8"))
    assert healed["NVDA"]["fetched_at"] == naive_now.replace(tzinfo=UTC).isoformat()


def test_get_accepts_naive_fetched_at_in_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    naive_fetched_at = "2026-11-17T10:00:00"
    cache_file.write_text(
        json.dumps(
            {
                "NVDA": {
                    "fetched_at": naive_fetched_at,
                    "data": {"symbol": "NVDA", "name": "NVIDIA", "price": 148.0},
                }
            }
        ),
        encoding="utf-8",
    )
    ticker_factory = MagicMock()
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
        clock=lambda: now,
    )

    stock = provider.get("NVDA")

    assert stock.price == 148.0
    ticker_factory.assert_not_called()


def test_default_cache_path_uses_env_var_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_cache = tmp_path / "custom" / "stocks.json"
    monkeypatch.setenv(CACHE_PATH_ENV_VAR, str(custom_cache))
    provider = YFinanceMarketDataProvider(
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "Apple Inc.", "currentPrice": 225.0})
        )
    )

    provider.get("AAPL")

    assert custom_cache.is_file()
    cached = json.loads(custom_cache.read_text(encoding="utf-8"))
    assert cached["AAPL"]["data"]["price"] == 225.0


def test_cache_read_and_write_errors_emit_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corrupted_cache = tmp_path / "corrupted.json"
    corrupted_cache.write_text("{invalid json", encoding="utf-8")
    provider_read = YFinanceMarketDataProvider(
        cache_path=corrupted_cache,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 150.0})
        ),
    )

    with caplog.at_level(logging.WARNING, logger="portfolio_tracker.market_data"):
        provider_read.get("NVDA")
    assert any("Impossible de charger le cache" in rec.message for rec in caplog.records)

    caplog.clear()
    blocked_parent = tmp_path / "blocked_parent"
    blocked_parent.write_text("not a dir", encoding="utf-8")
    provider_write = YFinanceMarketDataProvider(
        cache_path=blocked_parent / "stocks.json",
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "Apple Inc.", "currentPrice": 225.0})
        ),
    )

    with caplog.at_level(logging.WARNING, logger="portfolio_tracker.market_data"):
        provider_write.get("AAPL")
    assert any("Impossible d'enregistrer le cache" in rec.message for rec in caplog.records)


def test_get_reloads_cache_before_saving_to_preserve_concurrent_writes(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

    def fetch_with_concurrent_write(symbol: str) -> MagicMock:
        # Simule un autre appel ayant écrit MSFT dans le cache pendant l'appel réseau de NVDA.
        cache_file.write_text(
            json.dumps(
                {
                    "MSFT": {
                        "fetched_at": now.isoformat(),
                        "data": {"symbol": "MSFT", "name": "Microsoft", "price": 420.0},
                    }
                }
            ),
            encoding="utf-8",
        )
        return _make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 150.0})

    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=fetch_with_concurrent_write,
        clock=lambda: now,
    )

    provider.get("NVDA")

    saved = json.loads(cache_file.read_text(encoding="utf-8"))
    assert saved["MSFT"]["data"]["price"] == 420.0
    assert saved["NVDA"]["data"]["price"] == 150.0


def test_get_retries_transient_yfinance_failures_with_backoff(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    delays: list[float] = []
    ticker_factory = MagicMock(
        side_effect=[
            RuntimeError("HTTP 429 Too Many Requests"),
            RuntimeError("temporary reset"),
            _make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 152.0}),
        ]
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
        max_retries=2,
        backoff_seconds=0.5,
        sleep=delays.append,
    )

    stock = provider.get("NVDA")

    assert stock.price == 152.0
    assert ticker_factory.call_count == 3
    assert delays == [0.5, 1.0]


def test_get_raises_after_exhausting_yfinance_retries(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    delays: list[float] = []
    ticker_factory = MagicMock(side_effect=RuntimeError("network down"))
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=ticker_factory,
        max_retries=2,
        backoff_seconds=0.25,
        sleep=delays.append,
    )

    with pytest.raises(MarketDataError, match="network down"):
        provider.get("NVDA")

    assert ticker_factory.call_count == 3
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize("invalid_retries", [-1, True, False])
def test_provider_rejects_invalid_max_retries(tmp_path: Path, invalid_retries: int) -> None:
    with pytest.raises(MarketDataError):
        YFinanceMarketDataProvider(cache_path=tmp_path / "stocks.json", max_retries=invalid_retries)


@pytest.mark.parametrize(
    "invalid_backoff", [-0.1, float("nan"), float("inf"), float("-inf"), True, False]
)
def test_provider_rejects_invalid_backoff_seconds(
    tmp_path: Path, invalid_backoff: float
) -> None:
    with pytest.raises(MarketDataError):
        YFinanceMarketDataProvider(
            cache_path=tmp_path / "stocks.json", backoff_seconds=invalid_backoff
        )


@pytest.mark.parametrize("invalid_ttl", [timedelta(seconds=-1), timedelta(hours=-2)])
def test_provider_rejects_negative_cache_ttl(tmp_path: Path, invalid_ttl: timedelta) -> None:
    with pytest.raises(MarketDataError, match="cache_ttl"):
        YFinanceMarketDataProvider(cache_path=tmp_path / "stocks.json", cache_ttl=invalid_ttl)


@pytest.mark.parametrize("invalid_workers", [0, -1, True, False])
def test_provider_rejects_invalid_max_stalled_workers(
    tmp_path: Path, invalid_workers: int
) -> None:
    with pytest.raises(MarketDataError, match="max_stalled_workers"):
        YFinanceMarketDataProvider(
            cache_path=tmp_path / "stocks.json",
            max_stalled_workers=invalid_workers,
        )


@pytest.mark.parametrize("exc", [TypeError("bad type"), AttributeError("missing"), KeyError("k")])
def test_get_does_not_retry_programming_errors(tmp_path: Path, exc: Exception) -> None:
    delays: list[float] = []
    ticker_factory = MagicMock(side_effect=exc)
    provider = YFinanceMarketDataProvider(
        cache_path=tmp_path / "stocks.json",
        ticker_factory=ticker_factory,
        max_retries=2,
        backoff_seconds=0.5,
        sleep=delays.append,
    )

    with pytest.raises(MarketDataError):
        provider.get("NVDA")

    assert ticker_factory.call_count == 1
    assert delays == []


@pytest.mark.parametrize("invalid_timeout", [0, -1, -0.5, float("nan"), float("inf"), True, False])
def test_provider_rejects_invalid_timeout(tmp_path: Path, invalid_timeout: float) -> None:
    with pytest.raises(MarketDataError):
        YFinanceMarketDataProvider(cache_path=tmp_path / "stocks.json", timeout=invalid_timeout)


def test_get_times_out_and_retries_hanging_yfinance_call(tmp_path: Path) -> None:
    import threading

    release = threading.Event()
    delays: list[float] = []

    def hanging_factory(symbol: str) -> MagicMock:
        release.wait(timeout=2.0)
        return _make_mock_ticker({"shortName": "Late", "currentPrice": 1.0})

    ticker_factory = MagicMock(side_effect=hanging_factory)
    provider = YFinanceMarketDataProvider(
        cache_path=tmp_path / "stocks.json",
        ticker_factory=ticker_factory,
        max_retries=1,
        backoff_seconds=0.1,
        timeout=0.05,
        sleep=delays.append,
    )

    try:
        with pytest.raises(MarketDataError, match="Délai d'attente dépassé"):
            provider.get("NVDA")
    finally:
        release.set()

    assert ticker_factory.call_count == 2
    assert delays == [0.1]


def test_timeout_worker_runs_in_daemon_thread(tmp_path: Path) -> None:
    import threading

    seen_daemon: list[bool] = []
    release = threading.Event()

    def hanging_factory(symbol: str) -> MagicMock:
        seen_daemon.append(threading.current_thread().daemon)
        release.wait(timeout=2.0)
        return _make_mock_ticker({"shortName": "Late", "currentPrice": 1.0})

    provider = YFinanceMarketDataProvider(
        cache_path=tmp_path / "stocks.json",
        ticker_factory=hanging_factory,
        max_retries=0,
        timeout=0.05,
    )

    try:
        with pytest.raises(MarketDataError, match="Délai d'attente dépassé"):
            provider.get("NVDA")
    finally:
        release.set()

    assert seen_daemon == [True]


def test_get_maps_http_404_to_stock_not_found_without_retry(tmp_path: Path) -> None:
    class _FakeHTTPError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.response = MagicMock(status_code=status_code)

    delays: list[float] = []
    ticker_factory = MagicMock(side_effect=_FakeHTTPError(404))
    provider = YFinanceMarketDataProvider(
        cache_path=tmp_path / "stocks.json",
        ticker_factory=ticker_factory,
        max_retries=2,
        backoff_seconds=0.5,
        sleep=delays.append,
    )

    with pytest.raises(StockNotFoundError):
        provider.get("UNKNOWN")

    assert ticker_factory.call_count == 1
    assert delays == []


def test_get_does_not_retry_non_429_http_4xx_and_retries_429(tmp_path: Path) -> None:
    class _FakeHTTPError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.response = MagicMock(status_code=status_code)

    delays_403: list[float] = []
    factory_403 = MagicMock(side_effect=_FakeHTTPError(403))
    provider_403 = YFinanceMarketDataProvider(
        cache_path=tmp_path / "stocks.json",
        ticker_factory=factory_403,
        max_retries=2,
        backoff_seconds=0.5,
        sleep=delays_403.append,
    )
    with pytest.raises(MarketDataError, match="HTTP 403"):
        provider_403.get("NVDA")
    assert factory_403.call_count == 1
    assert delays_403 == []

    delays_429: list[float] = []
    factory_429 = MagicMock(
        side_effect=[
            _FakeHTTPError(429),
            _make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 150.0}),
        ]
    )
    provider_429 = YFinanceMarketDataProvider(
        cache_path=tmp_path / "stocks.json",
        ticker_factory=factory_429,
        max_retries=2,
        backoff_seconds=0.25,
        sleep=delays_429.append,
    )
    assert provider_429.get("NVDA").price == 150.0
    assert factory_429.call_count == 2
    assert delays_429 == [0.25]


def test_provider_caps_stalled_daemon_threads_and_recovers_after_exit(tmp_path: Path) -> None:
    import threading

    release = threading.Event()
    calls = 0

    def hanging_factory(symbol: str) -> MagicMock:
        nonlocal calls
        calls += 1
        release.wait(timeout=2.0)
        return _make_mock_ticker({"shortName": symbol, "currentPrice": 50.0})

    provider = YFinanceMarketDataProvider(
        cache_path=tmp_path / "stocks.json",
        ticker_factory=hanging_factory,
        max_retries=0,
        timeout=0.05,
        max_stalled_workers=2,
    )

    try:
        with pytest.raises(MarketDataError, match="Délai d'attente dépassé"):
            provider.get("AAA")
        with pytest.raises(MarketDataError, match="Délai d'attente dépassé"):
            provider.get("BBB")
        # 2 threads bloqués = plafond atteint : refuse immédiatement sans lancer un 3e thread zombie.
        with pytest.raises(MarketDataError, match="bloqués en arrière-plan"):
            provider.get("CCC")
        assert calls == 2
    finally:
        release.set()

    # Une fois les threads bloqués terminés, le fournisseur redevient opérationnel.
    for thread in list(provider._stalled_threads):
        thread.join(timeout=1.0)
    stock = provider.get("DDD")
    assert stock.symbol == "DDD"
    assert calls == 3


def test_cache_os_error_on_read_does_not_overwrite_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_file = tmp_path / "stocks.json"
    original_content = json.dumps(
        {
            "AAPL": {
                "fetched_at": datetime(2026, 11, 17, 10, 0, tzinfo=UTC).isoformat(),
                "data": {"symbol": "AAPL", "name": "Apple Inc.", "price": 225.0},
            }
        }
    )
    cache_file.write_text(original_content, encoding="utf-8")

    real_read_text = Path.read_text

    def failing_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == cache_file:
            raise OSError("lecture temporairement impossible")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", failing_read_text)

    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 150.0})
        ),
    )

    stock = provider.get("NVDA")
    assert stock.price == 150.0

    # Le cache existant ne doit pas avoir été écrasé par un dictionnaire ne contenant que NVDA.
    monkeypatch.undo()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == json.loads(original_content)


def test_get_rejects_whitespace_only_name_without_price_and_falls_back_to_long_name(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "stocks.json"
    blank_provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "   ", "longName": "\t\n"})
        ),
    )
    with pytest.raises(StockNotFoundError):
        blank_provider.get("GHOST")
    assert not cache_file.exists()

    fallback_provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "   ", "longName": "  Apple Inc.  "})
        ),
    )
    stock = fallback_provider.get("AAPL")
    assert stock.name == "Apple Inc."


def test_stock_data_rejects_invalid_fields_and_cache_heals_non_finite_json(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from portfolio_tracker.contracts import StockData

    for bad_kwargs in (
        {"symbol": "", "name": "Apple"},
        {"symbol": "   ", "name": "Apple"},
        {"symbol": "AAPL", "name": "   "},
        {"symbol": "AAPL", "name": "Apple", "price": float("nan")},
        {"symbol": "AAPL", "name": "Apple", "price": float("inf")},
    ):
        with pytest.raises(ValidationError):
            StockData(**bad_kwargs)  # type: ignore[arg-type]

    cache_file = tmp_path / "nan_cache.json"
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    cache_file.write_text(
        '{"NVDA": {"fetched_at": "'
        + now.isoformat()
        + '", "data": {"symbol": "NVDA", "name": "NVIDIA", "price": NaN}}}',
        encoding="utf-8",
    )
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=MagicMock(
            return_value=_make_mock_ticker({"shortName": "NVIDIA", "currentPrice": 150.0})
        ),
        clock=lambda: now,
    )

    stock = provider.get("NVDA")
    assert stock.price == 150.0
    reloaded = json.loads(cache_file.read_text(encoding="utf-8"))
    assert reloaded["NVDA"]["data"]["price"] == 150.0


def test_timeout_retries_are_capped_and_skip_sleep_when_stalled_limit_reached(
    tmp_path: Path,
) -> None:
    import threading

    release = threading.Event()
    delays_cap: list[float] = []

    def hanging_factory(symbol: str) -> MagicMock:
        release.wait(timeout=2.0)
        return _make_mock_ticker({"shortName": symbol, "currentPrice": 1.0})

    factory_cap = MagicMock(side_effect=hanging_factory)
    provider_cap = YFinanceMarketDataProvider(
        cache_path=tmp_path / "cap.json",
        ticker_factory=factory_cap,
        max_retries=3,
        backoff_seconds=0.1,
        timeout=0.05,
        max_stalled_workers=5,
        sleep=delays_cap.append,
    )

    delays_full: list[float] = []
    factory_full = MagicMock(side_effect=hanging_factory)
    provider_full = YFinanceMarketDataProvider(
        cache_path=tmp_path / "full.json",
        ticker_factory=factory_full,
        max_retries=2,
        backoff_seconds=0.5,
        timeout=0.05,
        max_stalled_workers=1,
        sleep=delays_full.append,
    )

    try:
        # Même avec max_retries=3, un timeout ne déclenche qu'1 retry (2 appels max par symbole).
        with pytest.raises(MarketDataError, match="Délai d'attente dépassé"):
            provider_cap.get("NVDA")
        assert factory_cap.call_count == 2
        assert delays_cap == [0.1]

        # Avec max_stalled_workers=1 atteint dès la 1re tentative, aucun sleep inutile avant l'échec.
        with pytest.raises(MarketDataError, match="Délai d'attente dépassé"):
            provider_full.get("AAPL")
        assert factory_full.call_count == 1
        assert delays_full == []
    finally:
        release.set()


def test_provider_concurrent_get_calls_preserve_all_cache_entries(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    cache_file = tmp_path / "concurrent.json"
    symbols = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "PEP"]
    provider = YFinanceMarketDataProvider(
        cache_path=cache_file,
        ticker_factory=lambda sym: _make_mock_ticker({"shortName": f"{sym} Corp", "currentPrice": 100.0}),
    )

    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        results = list(pool.map(provider.get, symbols))

    assert {stock.symbol for stock in results} == set(symbols)
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert set(cached) == set(symbols)


def test_provider_disables_yfinance_hide_exceptions_and_handles_real_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yfinance as yf
    from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError

    original_hide_exceptions = yf.config.debug.hide_exceptions
    yf.config.debug.hide_exceptions = True

    def _make_fake_response(status_code: int) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = '{"timeseries":{"result":[]}}'
        resp.url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/NVDA"

        def _raise_for_status() -> None:
            if status_code >= 400:
                raise CurlHTTPError(f"HTTP Error {status_code}", response=resp)

        resp.raise_for_status.side_effect = _raise_for_status
        return resp

    class _FakeSession:
        def __init__(self, status_code: int) -> None:
            self._resp = _make_fake_response(status_code)

        def get(self, *args: object, **kwargs: object) -> MagicMock:
            return self._resp

    def _make_real_ticker_factory(status_code: int) -> MagicMock:
        def _factory(symbol: str) -> yf.Ticker:
            ticker = yf.Ticker(symbol)
            monkeypatch.setattr(ticker._data, "_session", _FakeSession(status_code))
            monkeypatch.setattr(
                ticker._data,
                "_get_cookie_and_crumb",
                MagicMock(return_value=("crumb", "basic")),
            )
            monkeypatch.setattr(ticker._data, "_set_cookie_strategy", MagicMock())
            return ticker

        return MagicMock(side_effect=_factory)

    try:
        # 503 transitoire via un vrai yf.Ticker : ne doit pas être avalé en StockNotFoundError, et doit être retenté.
        delays_503: list[float] = []
        factory_503 = _make_real_ticker_factory(503)
        provider_503 = YFinanceMarketDataProvider(
            cache_path=tmp_path / "c503.json",
            ticker_factory=factory_503,
            max_retries=2,
            backoff_seconds=0.1,
            sleep=delays_503.append,
        )
        with pytest.raises(MarketDataError) as exc_503:
            provider_503.get("NVDA")
        assert not isinstance(exc_503.value, StockNotFoundError)
        assert yf.config.debug.hide_exceptions is False
        assert factory_503.call_count == 3
        assert delays_503 == [0.1, 0.2]

        # 404 via un vrai yf.Ticker : mappé en StockNotFoundError sans retry.
        delays_404: list[float] = []
        factory_404 = _make_real_ticker_factory(404)
        provider_404 = YFinanceMarketDataProvider(
            cache_path=tmp_path / "c404.json",
            ticker_factory=factory_404,
            max_retries=2,
            backoff_seconds=0.1,
            sleep=delays_404.append,
        )
        with pytest.raises(StockNotFoundError):
            provider_404.get("UNKNOWN")
        assert factory_404.call_count == 1
        assert delays_404 == []

        # 403 via un vrai yf.Ticker : MarketDataError immédiate sans retry (pas StockNotFoundError).
        delays_403: list[float] = []
        factory_403 = _make_real_ticker_factory(403)
        provider_403 = YFinanceMarketDataProvider(
            cache_path=tmp_path / "c403.json",
            ticker_factory=factory_403,
            max_retries=2,
            backoff_seconds=0.1,
            sleep=delays_403.append,
        )
        with pytest.raises(MarketDataError, match="HTTP 403") as exc_403:
            provider_403.get("NVDA")
        assert not isinstance(exc_403.value, StockNotFoundError)
        assert factory_403.call_count == 1
        assert delays_403 == []

        # 429 via un vrai yf.Ticker (YFRateLimitError) : retenté avec backoff.
        delays_429: list[float] = []
        factory_429 = _make_real_ticker_factory(429)
        provider_429 = YFinanceMarketDataProvider(
            cache_path=tmp_path / "c429.json",
            ticker_factory=factory_429,
            max_retries=1,
            backoff_seconds=0.25,
            sleep=delays_429.append,
        )
        with pytest.raises(MarketDataError) as exc_429:
            provider_429.get("NVDA")
        assert not isinstance(exc_429.value, StockNotFoundError)
        assert factory_429.call_count == 2
        assert delays_429 == [0.25]
    finally:
        yf.config.debug.hide_exceptions = original_hide_exceptions


def test_extract_http_status_code_maps_yfinance_rate_limit_and_ignores_curl_codes() -> None:
    from curl_cffi.curl import CurlError
    from yfinance.exceptions import YFRateLimitError

    from portfolio_tracker.market_data import _extract_http_status_code

    assert _extract_http_status_code(YFRateLimitError()) == 429
    assert _extract_http_status_code(CurlError("operation timed out", code=28)) is None
    assert _extract_http_status_code(CurlError("ok", code=0)) is None
