from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from portfolio_tracker.market_data import (
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
    current_time = datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc)
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
    current_time = datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc)
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
    now = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
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
    now = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
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
        clock=lambda: datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc),
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
    future_ts = (naive_now.replace(tzinfo=timezone.utc) + timedelta(hours=1)).isoformat()
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
    assert healed["NVDA"]["fetched_at"] == naive_now.replace(tzinfo=timezone.utc).isoformat()


def test_get_accepts_naive_fetched_at_in_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "stocks.json"
    now = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
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
