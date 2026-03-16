"""Tests for regime detector, session bias, polymarket sentiment, and trend filter."""

import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, ".")

from journal.models import OHLCV
from signals.regime_detector import RegimeDetector
from signals.session_bias import SessionBias
from signals.polymarket_sentiment import PolymarketSentiment
from signals.base_signal import SignalDirection


def _make_candles(prices: list[float], volume: float = 1000.0) -> list[OHLCV]:
    """Build OHLCV bars from a list of close prices (ascending timestamps)."""
    candles = []
    for i, p in enumerate(prices):
        candles.append(
            OHLCV(
                timestamp=datetime(2026, 1, 1, i // 60, i % 60, tzinfo=timezone.utc),
                open=p * 0.999,
                high=p * 1.002,
                low=p * 0.998,
                close=p,
                volume=volume,
                symbol="TEST",
                timeframe="1h",
                market="stock",
            )
        )
    return candles


# ===========================================================================
# ADX indicator
# ===========================================================================
class TestADX:
    """Tests for the ADX indicator."""

    def test_adx_returns_correct_length(self) -> None:
        from signals.indicators import Indicators

        candles = _make_candles([100 + i * 0.5 for i in range(60)])
        result = Indicators.adx(candles, 14)
        assert len(result) == len(candles)

    def test_adx_trending_higher_than_ranging(self) -> None:
        from signals.indicators import Indicators

        # Strong uptrend
        trending = _make_candles([100 + i * 2.0 for i in range(60)])
        adx_trend = Indicators.adx(trending, 14)

        # Sideways (oscillating)
        ranging_prices = [100 + (i % 5) * 0.5 for i in range(60)]
        ranging = _make_candles(ranging_prices)
        adx_range = Indicators.adx(ranging, 14)

        # Trending ADX should be higher at the end
        assert adx_trend[-1] > adx_range[-1]

    def test_adx_empty_input(self) -> None:
        from signals.indicators import Indicators

        assert Indicators.adx([], 14) == []
        assert Indicators.adx(_make_candles([100]), 14) == [0.0]


# ===========================================================================
# Regime Detector
# ===========================================================================
class TestRegimeDetector:
    """Tests for EMA+ADX market regime detection."""

    def test_trending_up(self) -> None:
        """Strong uptrend should be classified as trending_up."""
        detector = RegimeDetector(adx_threshold=20.0)
        # 60 bars of strong upward movement
        candles = _make_candles([100 + i * 3.0 for i in range(60)])
        regime = detector.detect(candles)
        assert regime == "trending_up"

    def test_trending_down(self) -> None:
        """Strong downtrend should be classified as trending_down."""
        detector = RegimeDetector(adx_threshold=20.0)
        candles = _make_candles([300 - i * 3.0 for i in range(60)])
        regime = detector.detect(candles)
        assert regime == "trending_down"

    def test_ranging_with_low_adx(self) -> None:
        """Sideways market should be classified as ranging."""
        detector = RegimeDetector(adx_threshold=25.0)
        # Oscillating prices that go nowhere
        prices = [100 + (i % 4) * 0.3 - 0.6 for i in range(60)]
        candles = _make_candles(prices)
        regime = detector.detect(candles)
        assert regime == "ranging"

    def test_insufficient_data_returns_ranging(self) -> None:
        detector = RegimeDetector()
        assert detector.detect([]) == "ranging"
        assert detector.detect(_make_candles([100] * 10)) == "ranging"


# ===========================================================================
# Session Bias
# ===========================================================================
class TestSessionBias:
    """Tests for daily directional bias."""

    def test_long_bias(self) -> None:
        bias = SessionBias()
        # Strong uptrend on 4h candles → long bias
        candles = _make_candles([100 + i * 2.0 for i in range(60)])
        result = bias.evaluate(candles)
        assert result == "long"

    def test_short_bias(self) -> None:
        bias = SessionBias()
        candles = _make_candles([300 - i * 2.0 for i in range(60)])
        result = bias.evaluate(candles)
        assert result == "short"

    def test_blocks_opposite_direction(self) -> None:
        bias = SessionBias()
        candles = _make_candles([100 + i * 2.0 for i in range(60)])
        bias.evaluate(candles)
        assert bias.should_block("short") is True
        assert bias.should_block("long") is False

    def test_neutral_blocks_nothing(self) -> None:
        bias = SessionBias()
        assert bias.should_block("long") is False
        assert bias.should_block("short") is False

    def test_insufficient_data_neutral(self) -> None:
        bias = SessionBias()
        assert bias.evaluate([]) == "neutral"


# ===========================================================================
# Polymarket Sentiment
# ===========================================================================
class TestPolymarketSentiment:
    """Tests for Polymarket sentiment layer."""

    def test_only_btc_eth(self) -> None:
        """Non-BTC/ETH assets should never be blocked."""
        poly = PolymarketSentiment(block_short_threshold=0.65)
        assert poly.should_block_short("SOL") is False
        assert poly.should_block_short("AAPL") is False

    def test_fallback_neutral(self) -> None:
        """When cache is empty, get_up_probability should fetch or default 0.5."""
        poly = PolymarketSentiment(block_short_threshold=0.99)
        # With threshold 0.99, even if API returns high prob, shouldn't block
        # (unless market literally at 99%)
        # At minimum, it shouldn't crash
        prob = poly.get_up_probability("BTC")
        assert 0.0 <= prob <= 1.0


# ===========================================================================
# Regime Allows (trend filter gate)
# ===========================================================================
class TestRegimeAllows:
    """Tests for the _regime_allows static method."""

    def test_rsi_momentum_short_blocked_in_uptrend(self) -> None:
        from main import AutoTrader

        assert AutoTrader._regime_allows(
            "rsi_momentum", SignalDirection.SHORT, "trending_up"
        ) is False

    def test_rsi_momentum_short_allowed_in_downtrend(self) -> None:
        from main import AutoTrader

        assert AutoTrader._regime_allows(
            "rsi_momentum", SignalDirection.SHORT, "trending_down"
        ) is True

    def test_rsi_momentum_short_allowed_in_ranging(self) -> None:
        from main import AutoTrader

        assert AutoTrader._regime_allows(
            "rsi_momentum", SignalDirection.SHORT, "ranging"
        ) is True

    def test_rsi_momentum_long_blocked_in_downtrend(self) -> None:
        from main import AutoTrader

        assert AutoTrader._regime_allows(
            "rsi_momentum", SignalDirection.LONG, "trending_down"
        ) is False

    def test_rsi_momentum_long_allowed_in_uptrend(self) -> None:
        from main import AutoTrader

        assert AutoTrader._regime_allows(
            "rsi_momentum", SignalDirection.LONG, "trending_up"
        ) is True

    def test_bollinger_fade_only_ranging(self) -> None:
        from main import AutoTrader

        assert AutoTrader._regime_allows(
            "bollinger_fade", SignalDirection.SHORT, "ranging"
        ) is True
        assert AutoTrader._regime_allows(
            "bollinger_fade", SignalDirection.SHORT, "trending_up"
        ) is False
        assert AutoTrader._regime_allows(
            "bollinger_fade", SignalDirection.LONG, "trending_down"
        ) is False

    def test_ema_cross_always_allowed(self) -> None:
        from main import AutoTrader

        for regime in ("trending_up", "trending_down", "ranging"):
            assert AutoTrader._regime_allows(
                "ema_cross", SignalDirection.LONG, regime
            ) is True
            assert AutoTrader._regime_allows(
                "ema_cross", SignalDirection.SHORT, regime
            ) is True

    def test_first_candle_always_allowed(self) -> None:
        from main import AutoTrader

        for regime in ("trending_up", "trending_down", "ranging"):
            assert AutoTrader._regime_allows(
                "first_candle", SignalDirection.LONG, regime
            ) is True
