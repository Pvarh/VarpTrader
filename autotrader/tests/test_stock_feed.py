"""Tests for stock feed caching and Polygon rate budgeting."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.feed_stocks import StockFeed
from journal.models import OHLCV


def _sample_bars() -> list[OHLCV]:
    return [
        OHLCV(
            timestamp=datetime(2026, 3, 24, 12, 0, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            symbol="AAPL",
            timeframe="5m",
            market="stock",
        )
    ]


def test_polygon_bars_are_served_from_fresh_cache() -> None:
    feed = StockFeed(polygon_api_key="test-key")
    bars = _sample_bars()

    with patch.object(feed, "_fetch_polygon_bars", return_value=bars) as polygon_fetch:
        first = feed.get_historical_bars("AAPL", period="2d", interval="5m")
        second = feed.get_historical_bars("AAPL", period="2d", interval="5m")

    assert first == bars
    assert second == bars
    assert polygon_fetch.call_count == 1


def test_polygon_budget_uses_stale_cache_before_yfinance() -> None:
    feed = StockFeed(polygon_api_key="test-key")
    bars = _sample_bars()
    cache_key = ("AAPL", "2d", "5m")
    feed._bars_cache[cache_key] = (time.monotonic() - 3600, bars)
    now = time.monotonic()
    feed._polygon_request_times.extend([now - 1] * feed._polygon_calls_per_minute)

    with patch.object(feed, "_fetch_polygon_bars", return_value=[]) as polygon_fetch, patch.object(
        feed, "_fetch_yfinance_bars", return_value=[]
    ) as yfinance_fetch:
        result = feed.get_historical_bars("AAPL", period="2d", interval="5m")

    assert result == bars
    polygon_fetch.assert_not_called()
    yfinance_fetch.assert_not_called()


def test_hourly_bars_skip_polygon_on_low_tier_budget() -> None:
    feed = StockFeed(polygon_api_key="test-key")
    bars = _sample_bars()

    with patch.object(feed, "_fetch_polygon_bars", return_value=bars) as polygon_fetch, patch.object(
        feed, "_fetch_yfinance_bars", return_value=bars
    ) as yfinance_fetch:
        result = feed.get_historical_bars("AAPL", period="1mo", interval="1h")

    assert result == bars
    polygon_fetch.assert_not_called()
    yfinance_fetch.assert_called_once()
