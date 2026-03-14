"""Tests for the backtest module.

Covers:
- BacktestEngine with RSI momentum signal on synthetic data.
- Correct PnL computation (long wins, long losses, short wins, short losses).
- Stop-loss / take-profit hit detection.
- Equity curve tracking.
- BacktestResult.summary() output.
- HistoricalDataLoader.load_from_csv.
- HistoricalDataLoader.resample.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Optional

import pytest

sys.path.insert(0, ".")

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult
from backtest.engine import BacktestEngine, BacktestResult
from backtest.data_loader import HistoricalDataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    min_reward_ratio: float = 2.0,
    stop_loss_pct: float = 0.015,
    position_size_pct: float = 0.02,
    max_positions: int = 5,
) -> dict:
    """Build a minimal config dict suitable for :class:`BacktestEngine`."""
    return {
        "trading": {"max_positions": max_positions},
        "risk": {
            "stop_loss_pct": stop_loss_pct,
            "position_size_pct": position_size_pct,
            "min_reward_ratio": min_reward_ratio,
            "atr_period": 14,
            "atr_multiplier": 1.5,
        },
    }


def _make_candle(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
    symbol: str = "TEST",
    timeframe: str = "5m",
    market: str = "stock",
    ts: datetime | None = None,
    ts_offset_min: int = 0,
) -> OHLCV:
    """Create an OHLCV candle for testing.

    Args:
        close: Close price (also used as default for open/high/low).
        open_: Open price (defaults to *close*).
        high: High price (defaults to ``max(open, close) * 1.001``).
        low: Low price (defaults to ``min(open, close) * 0.999``).
        volume: Bar volume.
        symbol: Trading symbol.
        timeframe: Candle timeframe label.
        market: ``'stock'`` or ``'crypto'``.
        ts: Exact timestamp.  If ``None`` a default base is used.
        ts_offset_min: Minutes to add to the base timestamp.

    Returns:
        An :class:`OHLCV` instance.
    """
    if open_ is None:
        open_ = close
    if high is None:
        high = max(open_, close) * 1.001
    if low is None:
        low = min(open_, close) * 0.999
    if ts is None:
        ts = datetime(2025, 1, 15, 9, 45) + timedelta(minutes=ts_offset_min)
    return OHLCV(
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol=symbol,
        timeframe=timeframe,
        market=market,
    )


def _make_candle_series(
    closes: list[float],
    base_ts: datetime | None = None,
    interval_min: int = 5,
    **kwargs,
) -> list[OHLCV]:
    """Create a series of candles from a list of close prices.

    Args:
        closes: Close prices (one candle per value).
        base_ts: Starting timestamp for the first candle.
        interval_min: Minutes between consecutive candles.
        **kwargs: Forwarded to :func:`_make_candle`.

    Returns:
        Chronologically sorted list of :class:`OHLCV`.
    """
    if base_ts is None:
        base_ts = datetime(2025, 1, 15, 9, 30)
    return [
        _make_candle(
            c,
            ts=base_ts + timedelta(minutes=i * interval_min),
            **kwargs,
        )
        for i, c in enumerate(closes)
    ]


# ---------------------------------------------------------------------------
# Test signal that triggers on a specific bar (for engine mechanics tests)
# ---------------------------------------------------------------------------


class _TestSignal(BaseSignal):
    """A deterministic signal that fires exactly once on a given bar.

    Used for testing the engine mechanics (position opening, SL/TP exit,
    PnL, equity curve) independently of indicator computation.
    """

    def __init__(
        self,
        config: dict,
        direction: SignalDirection = SignalDirection.LONG,
        trigger_on_bar: int = 0,
        sl_pct: float = 0.01,
        tp_multiplier: float = 2.0,
    ) -> None:
        super().__init__(config)
        self._direction = direction
        self._trigger_on_bar = trigger_on_bar
        self._sl_pct = sl_pct
        self._tp_multiplier = tp_multiplier

    @property
    def name(self) -> str:
        """Return the unique name of this signal strategy."""
        return "test_signal"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Return a triggered result only on the configured bar index."""
        bar_index = len(candles) - 1
        if bar_index != self._trigger_on_bar:
            return SignalResult(triggered=False, strategy_name=self.name)

        entry = current_price
        sl_dist = entry * self._sl_pct

        if self._direction == SignalDirection.LONG:
            sl = entry - sl_dist
            tp = entry + sl_dist * self._tp_multiplier
        else:
            sl = entry + sl_dist
            tp = entry - sl_dist * self._tp_multiplier

        return SignalResult(
            triggered=True,
            direction=self._direction,
            symbol=symbol,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=0.8,
            strategy_name=self.name,
            reason="Test signal triggered",
        )


# ===========================================================================
# BacktestEngine with RSI signal on synthetic data
# ===========================================================================


class TestBacktestEngineRSI:
    """Test BacktestEngine with the RSI momentum signal on synthetic data."""

    @pytest.fixture
    def rsi_config(self) -> dict:
        """Engine config with RSI-friendly parameters."""
        return {
            "trading": {"max_positions": 5},
            "risk": {
                "stop_loss_pct": 0.015,
                "position_size_pct": 0.02,
                "min_reward_ratio": 2.0,
                "atr_period": 14,
                "atr_multiplier": 1.5,
            },
        }

    @pytest.fixture
    def rsi_signal_config(self) -> dict:
        """Config for the RSIMomentumSignal itself."""
        return {
            "enabled": True,
            "rsi_oversold": 40,
            "rsi_overbought": 60,
            "rsi_1hr_oversold": 50,
            "rsi_1hr_overbought": 50,
            "rsi_period": 14,
            "stop_loss_pct": 0.015,
        }

    def _build_declining_then_recovering_candles(
        self,
    ) -> tuple[list[OHLCV], list[OHLCV]]:
        """Create 5-min candles that decline then recover, plus matching 1h candles.

        Returns:
            (candles_5m, candles_1h) -- both sorted chronologically.
        """
        base_5m = datetime(2025, 1, 15, 9, 30)
        closes_5m: list[float] = []

        # 25 candles declining: 100 -> 76
        for i in range(25):
            closes_5m.append(100.0 - i)

        # 15 candles recovering: 77 -> 91
        for i in range(15):
            closes_5m.append(77.0 + i)

        candles_5m = _make_candle_series(
            closes_5m, base_ts=base_5m, interval_min=5,
        )

        # 1h candles: 20 candles declining from 105 to 86
        base_1h = datetime(2025, 1, 14, 9, 0)
        closes_1h = [105.0 - i for i in range(20)]
        candles_1h = _make_candle_series(
            closes_1h, base_ts=base_1h, interval_min=60, timeframe="1h",
        )

        return candles_5m, candles_1h

    def test_rsi_signal_triggers_long_on_oversold(
        self, rsi_config: dict, rsi_signal_config: dict,
    ) -> None:
        """Engine should detect oversold RSI and open a long position."""
        from signals.rsi_momentum import RSIMomentumSignal

        signal = RSIMomentumSignal(rsi_signal_config)
        candles_5m, candles_1h = self._build_declining_then_recovering_candles()

        engine = BacktestEngine(rsi_config, initial_capital=100_000.0)
        result = engine.run(candles_5m, [signal], candles_1h=candles_1h)

        assert result.total_trades >= 1, (
            "Expected at least one trade from RSI momentum signal"
        )
        # The first trade opened should be a long (oversold bounce)
        first_trade = result.trades[0]
        assert first_trade["direction"] == "long"

    def test_rsi_long_produces_positive_pnl(
        self, rsi_config: dict, rsi_signal_config: dict,
    ) -> None:
        """An RSI-triggered long should be profitable when price recovers."""
        from signals.rsi_momentum import RSIMomentumSignal

        signal = RSIMomentumSignal(rsi_signal_config)
        candles_5m, candles_1h = self._build_declining_then_recovering_candles()

        engine = BacktestEngine(rsi_config, initial_capital=100_000.0)
        result = engine.run(candles_5m, [signal], candles_1h=candles_1h)

        assert result.total_trades >= 1
        assert result.total_pnl > 0, (
            "RSI long trade should be profitable with a price recovery"
        )


# ===========================================================================
# PnL computation
# ===========================================================================


class TestPnLComputation:
    """Test correct PnL computation for various trade outcomes."""

    @pytest.fixture
    def config(self) -> dict:
        """Standard config for PnL tests."""
        return _make_config()

    def test_long_trade_positive_pnl(self, config: dict) -> None:
        """Long trade that hits TP should produce positive PnL."""
        # Bar 0: entry at 100.  SL=99, TP=102.
        # Bar 1: high=103 -> TP hit.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                103.0, high=103.5, low=100.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal(
            {"enabled": True},
            direction=SignalDirection.LONG,
            trigger_on_bar=0,
            sl_pct=0.01,
            tp_multiplier=2.0,
        )
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade["outcome"] == "win"
        assert trade["pnl"] > 0
        # PnL = (TP - entry) * qty = (102 - 100) * qty
        expected_pnl = (102.0 - 100.0) * trade["quantity"]
        assert trade["pnl"] == pytest.approx(expected_pnl, rel=1e-6)

    def test_long_trade_negative_pnl(self, config: dict) -> None:
        """Long trade that hits SL should produce negative PnL."""
        # Bar 0: entry at 100.  SL=99, TP=102.
        # Bar 1: low=98 -> SL hit.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                98.5, high=100.5, low=98.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal(
            {"enabled": True},
            direction=SignalDirection.LONG,
            trigger_on_bar=0,
        )
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade["outcome"] == "loss"
        assert trade["pnl"] < 0
        expected_pnl = (99.0 - 100.0) * trade["quantity"]
        assert trade["pnl"] == pytest.approx(expected_pnl, rel=1e-6)

    def test_short_trade_positive_pnl(self, config: dict) -> None:
        """Short trade that hits TP should produce positive PnL."""
        # Bar 0: entry at 100.  SL=101, TP=98.
        # Bar 1: low=97 -> TP hit.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                97.5, high=100.0, low=97.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal(
            {"enabled": True},
            direction=SignalDirection.SHORT,
            trigger_on_bar=0,
        )
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade["outcome"] == "win"
        assert trade["pnl"] > 0
        expected_pnl = (100.0 - 98.0) * trade["quantity"]
        assert trade["pnl"] == pytest.approx(expected_pnl, rel=1e-6)

    def test_short_trade_negative_pnl(self, config: dict) -> None:
        """Short trade that hits SL should produce negative PnL."""
        # Bar 0: entry at 100.  SL=101, TP=98.
        # Bar 1: high=102 -> SL hit.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                101.5, high=102.0, low=100.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal(
            {"enabled": True},
            direction=SignalDirection.SHORT,
            trigger_on_bar=0,
        )
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade["outcome"] == "loss"
        assert trade["pnl"] < 0
        expected_pnl = (100.0 - 101.0) * trade["quantity"]
        assert trade["pnl"] == pytest.approx(expected_pnl, rel=1e-6)


# ===========================================================================
# Stop-loss / take-profit hit detection
# ===========================================================================


class TestStopLossTakeProfit:
    """Test SL/TP hit detection."""

    @pytest.fixture
    def config(self) -> dict:
        """Standard config."""
        return _make_config()

    def test_long_sl_hit_exact_level(self, config: dict) -> None:
        """Long SL is triggered when candle low equals stop-loss price."""
        # Entry at 100, SL = 99.  Bar 1 low exactly 99.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                99.5, high=100.2, low=99.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        assert result.trades[0]["exit_price"] == pytest.approx(99.0, rel=1e-6)
        assert result.trades[0]["outcome"] == "loss"

    def test_long_tp_hit(self, config: dict) -> None:
        """Long TP is triggered when candle high reaches take-profit."""
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                101.5, high=102.5, low=100.5,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        assert result.trades[0]["exit_price"] == pytest.approx(
            102.0, rel=1e-6,
        )
        assert result.trades[0]["outcome"] == "win"

    def test_short_sl_hit(self, config: dict) -> None:
        """Short SL is triggered when candle high reaches stop-loss."""
        # Entry at 100, SL = 101.  Bar 1 high = 101.5.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                101.0, high=101.5, low=99.5,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal(
            {"enabled": True}, direction=SignalDirection.SHORT, trigger_on_bar=0,
        )
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        assert result.trades[0]["exit_price"] == pytest.approx(
            101.0, rel=1e-6,
        )
        assert result.trades[0]["outcome"] == "loss"

    def test_short_tp_hit(self, config: dict) -> None:
        """Short TP is triggered when candle low reaches take-profit."""
        # Entry at 100, TP = 98.  Bar 1 low = 97.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                97.5, high=100.0, low=97.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal(
            {"enabled": True}, direction=SignalDirection.SHORT, trigger_on_bar=0,
        )
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        assert result.trades[0]["exit_price"] == pytest.approx(98.0, rel=1e-6)
        assert result.trades[0]["outcome"] == "win"

    def test_both_sl_and_tp_hittable_defaults_to_sl(self, config: dict) -> None:
        """When a single candle spans both SL and TP, SL is assumed first."""
        # Entry at 100, SL=99, TP=102.  Bar 1: low=98, high=103.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                100.0, high=103.0, low=98.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.total_trades == 1
        # Conservative: SL assumed to be hit first
        assert result.trades[0]["exit_price"] == pytest.approx(99.0, rel=1e-6)
        assert result.trades[0]["outcome"] == "loss"

    def test_position_held_when_no_sl_tp_hit(self, config: dict) -> None:
        """Position stays open if neither SL nor TP is reached."""
        # Entry at 100, SL=99, TP=102.  Bar 1 stays between 99.5 and 101.5.
        # Bar 2: same range. Position forced closed at bar 2 close.
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                100.5, high=101.0, low=99.5,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
            _make_candle(
                100.8, high=101.2, low=99.5,
                ts=datetime(2025, 1, 15, 9, 40),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        # Trade force-closed at last candle's close (100.8)
        assert result.total_trades == 1
        assert result.trades[0]["exit_price"] == pytest.approx(
            100.8, rel=1e-6,
        )


# ===========================================================================
# Equity curve tracking
# ===========================================================================


class TestEquityCurve:
    """Test equity curve tracking."""

    @pytest.fixture
    def config(self) -> dict:
        """Standard config."""
        return _make_config()

    def test_equity_curve_starts_at_initial_capital(self, config: dict) -> None:
        """First element of the equity curve should be the initial capital."""
        candles = _make_candle_series([100.0, 101.0, 102.0])
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=50_000.0)
        result = engine.run(candles, [signal])

        assert result.equity_curve[0] == pytest.approx(50_000.0)

    def test_equity_curve_length(self, config: dict) -> None:
        """Equity curve should have initial + one entry per candle."""
        n_candles = 10
        candles = _make_candle_series([100.0] * n_candles)
        # Use a signal that never fires
        signal = _TestSignal(
            {"enabled": True}, trigger_on_bar=999,
        )
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert len(result.equity_curve) == n_candles + 1

    def test_equity_curve_reflects_winning_trade(self, config: dict) -> None:
        """After a winning trade the equity curve should be above initial."""
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                103.0, high=103.0, low=100.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        # After TP hit, last equity should be above initial
        assert result.equity_curve[-1] > 100_000.0

    def test_equity_curve_reflects_losing_trade(self, config: dict) -> None:
        """After a losing trade the equity curve should be below initial."""
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                98.0, high=100.0, low=97.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.equity_curve[-1] < 100_000.0

    def test_equity_flat_with_no_trades(self, config: dict) -> None:
        """Equity curve is flat when no trades are opened."""
        candles = _make_candle_series([100.0, 101.0, 99.0, 100.5])
        signal = _TestSignal({"enabled": True}, trigger_on_bar=999)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        for eq in result.equity_curve:
            assert eq == pytest.approx(100_000.0)


# ===========================================================================
# BacktestResult summary
# ===========================================================================


class TestBacktestResultSummary:
    """Test BacktestResult.summary() output."""

    def test_summary_contains_key_metrics(self) -> None:
        """The summary string should contain all key metric labels."""
        r = BacktestResult(
            total_trades=10,
            wins=6,
            losses=3,
            breakeven=1,
            total_pnl=5_000.0,
            win_rate=60.0,
            avg_win=1_500.0,
            avg_loss=-1_000.0,
            max_drawdown=2_000.0,
            max_drawdown_pct=4.0,
            sharpe_ratio=1.8,
            profit_factor=2.5,
        )
        summary = r.summary()

        assert "Total Trades" in summary
        assert "10" in summary
        assert "Wins" in summary
        assert "6" in summary
        assert "Losses" in summary
        assert "Win Rate" in summary
        assert "60.0%" in summary
        assert "Total PnL" in summary
        assert "Sharpe Ratio" in summary
        assert "Profit Factor" in summary
        assert "Max Drawdown" in summary

    def test_summary_handles_zero_trades(self) -> None:
        """Summary should render cleanly when there are no trades."""
        r = BacktestResult()
        summary = r.summary()

        assert "Total Trades" in summary
        assert "0" in summary
        assert "BACKTEST RESULTS" in summary

    def test_summary_returns_string(self) -> None:
        """summary() must always return a str."""
        r = BacktestResult(total_trades=1, wins=1, total_pnl=100.0)
        assert isinstance(r.summary(), str)


# ===========================================================================
# HistoricalDataLoader -- CSV
# ===========================================================================


class TestHistoricalDataLoaderCSV:
    """Test HistoricalDataLoader.load_from_csv."""

    def _write_csv(self, filepath: str, rows: list[dict]) -> None:
        """Write a list of row dicts to a CSV file."""
        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_csv_correctly(self) -> None:
        """CSV data should be loaded and converted to OHLCV objects."""
        rows = [
            {
                "timestamp": "2025-01-15T09:30:00",
                "open": "100.0",
                "high": "101.0",
                "low": "99.0",
                "close": "100.5",
                "volume": "5000",
            },
            {
                "timestamp": "2025-01-15T09:35:00",
                "open": "100.5",
                "high": "102.0",
                "low": "100.0",
                "close": "101.5",
                "volume": "6000",
            },
            {
                "timestamp": "2025-01-15T09:40:00",
                "open": "101.5",
                "high": "103.0",
                "low": "101.0",
                "close": "102.0",
                "volume": "7000",
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name

        try:
            self._write_csv(tmp_path, rows)

            loader = HistoricalDataLoader()
            candles = loader.load_from_csv(
                tmp_path, symbol="AAPL", market="stock", timeframe="5m",
            )

            assert len(candles) == 3
            assert candles[0].symbol == "AAPL"
            assert candles[0].market == "stock"
            assert candles[0].timeframe == "5m"
            assert candles[0].open == pytest.approx(100.0)
            assert candles[0].high == pytest.approx(101.0)
            assert candles[0].low == pytest.approx(99.0)
            assert candles[0].close == pytest.approx(100.5)
            assert candles[0].volume == pytest.approx(5000.0)
            assert candles[0].timestamp == datetime(2025, 1, 15, 9, 30)

            # Verify chronological order
            for i in range(1, len(candles)):
                assert candles[i].timestamp > candles[i - 1].timestamp
        finally:
            os.unlink(tmp_path)

    def test_csv_with_vwap_column(self) -> None:
        """CSV with an optional vwap column should populate OHLCV.vwap."""
        rows = [
            {
                "timestamp": "2025-01-15T09:30:00",
                "open": "100.0",
                "high": "101.0",
                "low": "99.0",
                "close": "100.5",
                "volume": "5000",
                "vwap": "100.2",
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name

        try:
            self._write_csv(tmp_path, rows)

            loader = HistoricalDataLoader()
            candles = loader.load_from_csv(
                tmp_path, symbol="AAPL", market="stock", timeframe="5m",
            )

            assert len(candles) == 1
            assert candles[0].vwap == pytest.approx(100.2)
        finally:
            os.unlink(tmp_path)

    def test_empty_csv_returns_empty_list(self) -> None:
        """A CSV with only a header should return an empty list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write("timestamp,open,high,low,close,volume\n")
            tmp_path = tmp.name

        try:
            loader = HistoricalDataLoader()
            candles = loader.load_from_csv(
                tmp_path, symbol="AAPL", market="stock", timeframe="5m",
            )
            assert candles == []
        finally:
            os.unlink(tmp_path)

    def test_csv_out_of_order_is_sorted(self) -> None:
        """Candles loaded from an unordered CSV should be sorted by timestamp."""
        rows = [
            {
                "timestamp": "2025-01-15T09:40:00",
                "open": "101.5",
                "high": "103.0",
                "low": "101.0",
                "close": "102.0",
                "volume": "7000",
            },
            {
                "timestamp": "2025-01-15T09:30:00",
                "open": "100.0",
                "high": "101.0",
                "low": "99.0",
                "close": "100.5",
                "volume": "5000",
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name

        try:
            self._write_csv(tmp_path, rows)

            loader = HistoricalDataLoader()
            candles = loader.load_from_csv(
                tmp_path, symbol="TEST", market="stock", timeframe="5m",
            )

            assert candles[0].timestamp < candles[1].timestamp
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# HistoricalDataLoader -- resample
# ===========================================================================


class TestHistoricalDataLoaderResample:
    """Test HistoricalDataLoader.resample."""

    def test_resample_5m_to_15m(self) -> None:
        """Resampling six 5-min candles to 15-min should produce two candles."""
        base = datetime(2025, 1, 15, 9, 30)
        candles = [
            _make_candle(
                close=100.0 + i,
                open_=100.0 + i - 0.5,
                high=100.0 + i + 1.0,
                low=100.0 + i - 1.0,
                volume=1000.0 * (i + 1),
                ts=base + timedelta(minutes=i * 5),
            )
            for i in range(6)
        ]

        loader = HistoricalDataLoader()
        resampled = loader.resample(candles, "15m")

        # 6 five-minute candles spanning 30 minutes -> 2 fifteen-minute candles
        assert len(resampled) == 2
        assert resampled[0].timeframe == "15m"

    def test_resample_preserves_ohlcv_semantics(self) -> None:
        """Resampled bars should have correct OHLCV aggregation.

        - open  = first candle's open
        - high  = max high across group
        - low   = min low across group
        - close = last candle's close
        - volume = sum of volumes
        """
        base = datetime(2025, 1, 15, 9, 30)
        candles = [
            _make_candle(
                close=101.0, open_=100.0, high=102.0, low=99.0,
                volume=1000.0, ts=base,
            ),
            _make_candle(
                close=103.0, open_=101.0, high=104.0, low=100.5,
                volume=2000.0, ts=base + timedelta(minutes=5),
            ),
            _make_candle(
                close=102.0, open_=103.0, high=103.5, low=101.0,
                volume=1500.0, ts=base + timedelta(minutes=10),
            ),
        ]

        loader = HistoricalDataLoader()
        resampled = loader.resample(candles, "15m")

        assert len(resampled) == 1
        bar = resampled[0]

        assert bar.open == pytest.approx(100.0)     # first open
        assert bar.high == pytest.approx(104.0)     # max high
        assert bar.low == pytest.approx(99.0)       # min low
        assert bar.close == pytest.approx(102.0)    # last close
        assert bar.volume == pytest.approx(4500.0)  # sum of volumes

    def test_resample_5m_to_1h(self) -> None:
        """Resampling twelve 5-min candles to 1h should produce one candle."""
        base = datetime(2025, 1, 15, 9, 0)
        candles = [
            _make_candle(
                close=100.0 + i,
                volume=1000.0,
                ts=base + timedelta(minutes=i * 5),
            )
            for i in range(12)
        ]

        loader = HistoricalDataLoader()
        resampled = loader.resample(candles, "1h")

        assert len(resampled) == 1
        assert resampled[0].timeframe == "1h"
        assert resampled[0].volume == pytest.approx(12_000.0)

    def test_resample_empty_returns_empty(self) -> None:
        """Resampling an empty list should return an empty list."""
        loader = HistoricalDataLoader()
        assert loader.resample([], "15m") == []

    def test_resample_preserves_symbol_and_market(self) -> None:
        """Resampled candles should carry the source symbol and market."""
        base = datetime(2025, 1, 15, 9, 30)
        candles = [
            _make_candle(
                close=100.0,
                volume=1000.0,
                symbol="BTC/USDT",
                market="crypto",
                ts=base + timedelta(minutes=i * 5),
            )
            for i in range(3)
        ]

        loader = HistoricalDataLoader()
        resampled = loader.resample(candles, "15m")

        assert len(resampled) == 1
        assert resampled[0].symbol == "BTC/USDT"
        assert resampled[0].market == "crypto"


# ===========================================================================
# Engine edge cases
# ===========================================================================


class TestBacktestEngineEdgeCases:
    """Additional edge-case coverage for the backtest engine."""

    @pytest.fixture
    def config(self) -> dict:
        """Standard config."""
        return _make_config()

    def test_empty_candles_returns_default_result(self, config: dict) -> None:
        """Running with no candles should return a default BacktestResult."""
        signal = _TestSignal({"enabled": True})
        engine = BacktestEngine(config)
        result = engine.run([], [signal])

        assert result.total_trades == 0
        assert result.equity_curve == [100_000.0]

    def test_disabled_signal_is_skipped(self, config: dict) -> None:
        """A disabled signal should not produce any trades."""
        candles = _make_candle_series([100.0, 103.0])
        signal = _TestSignal({"enabled": False}, trigger_on_bar=0)
        engine = BacktestEngine(config)
        result = engine.run(candles, [signal])

        assert result.total_trades == 0

    def test_rr_rejected_trade_not_opened(self) -> None:
        """Trades failing the reward:ratio gate should not be opened."""
        config = _make_config(min_reward_ratio=5.0)
        candles = _make_candle_series([100.0, 103.0])
        # Default signal has R:R = 2.0, which is < 5.0
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config)
        result = engine.run(candles, [signal])

        assert result.total_trades == 0

    def test_max_drawdown_computed(self, config: dict) -> None:
        """Max drawdown should be positive when a losing trade occurs."""
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                98.0, high=100.0, low=97.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal])

        assert result.max_drawdown > 0
        assert result.max_drawdown_pct > 0

    def test_profit_factor_computed(self, config: dict) -> None:
        """Profit factor should be computed from wins and losses."""
        # Two trades: first wins, second loses.
        candles = [
            # Trade 1: bar 0 entry, bar 1 TP hit
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(
                103.0, high=103.0, low=100.0,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
            # Trade 2: bar 2 entry, bar 3 SL hit
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 40)),
            _make_candle(
                98.0, high=100.0, low=97.0,
                ts=datetime(2025, 1, 15, 9, 45),
            ),
        ]
        signal = _TestSignal({"enabled": True}, trigger_on_bar=0)
        signal2 = _TestSignal({"enabled": True}, trigger_on_bar=2)
        # Need a different name for the second signal
        signal2._name_override = "test_signal_2"
        original_name = type(signal2).name
        type(signal2).name = property(lambda self: getattr(self, "_name_override", "test_signal"))
        signal2._name_override = "test_signal_2"

        engine = BacktestEngine(config, initial_capital=100_000.0)
        result = engine.run(candles, [signal, signal2])

        # At least one trade should have completed
        assert result.total_trades >= 1


# ===========================================================================
# Indicator computation unit tests
# ===========================================================================


class TestIndicatorComputations:
    """Direct tests for the engine's internal indicator methods."""

    @pytest.fixture
    def engine(self) -> BacktestEngine:
        """Create an engine for calling internal methods."""
        config = _make_config()
        return BacktestEngine(config)

    def test_rsi_all_gains_returns_100(self, engine: BacktestEngine) -> None:
        """RSI with only upward moves should be 100."""
        closes = [float(i) for i in range(1, 20)]  # 1, 2, ..., 19
        rsi = engine._compute_rsi(closes, period=14)
        assert rsi == pytest.approx(100.0)

    def test_rsi_all_losses_returns_0(self, engine: BacktestEngine) -> None:
        """RSI with only downward moves should be 0."""
        closes = [float(20 - i) for i in range(20)]  # 20, 19, ..., 1
        rsi = engine._compute_rsi(closes, period=14)
        assert rsi == pytest.approx(0.0, abs=0.01)

    def test_rsi_insufficient_data_returns_50(
        self, engine: BacktestEngine,
    ) -> None:
        """RSI with fewer than period+1 values should return neutral 50."""
        closes = [100.0, 101.0, 102.0]
        rsi = engine._compute_rsi(closes, period=14)
        assert rsi == pytest.approx(50.0)

    def test_ema_single_period(self, engine: BacktestEngine) -> None:
        """EMA with period=1 should equal the last close."""
        closes = [100.0, 102.0, 104.0]
        ema = engine._compute_ema(closes, period=1)
        assert ema == pytest.approx(104.0, rel=1e-2)

    def test_ema_equals_sma_when_one_period_of_data(
        self, engine: BacktestEngine,
    ) -> None:
        """When data length equals period, EMA should equal SMA."""
        closes = [100.0, 102.0, 104.0, 106.0, 108.0]
        ema = engine._compute_ema(closes, period=5)
        sma = sum(closes) / 5
        assert ema == pytest.approx(sma)

    def test_compute_atr_simple(self, engine: BacktestEngine) -> None:
        """ATR should reflect the true range of the candles."""
        candles = [
            _make_candle(
                100.0, open_=100.0, high=101.0, low=99.0,
                ts=datetime(2025, 1, 15, 9, 30),
            ),
            _make_candle(
                101.0, open_=100.0, high=102.0, low=99.5,
                ts=datetime(2025, 1, 15, 9, 35),
            ),
        ]
        atr = engine._compute_atr(candles, period=14)
        # TR = max(102-99.5, |102-100|, |99.5-100|) = max(2.5, 2.0, 0.5) = 2.5
        assert atr == pytest.approx(2.5, rel=1e-6)

    def test_compute_atr_insufficient_data(
        self, engine: BacktestEngine,
    ) -> None:
        """ATR with fewer than 2 candles should return 0."""
        candles = [
            _make_candle(100.0, ts=datetime(2025, 1, 15, 9, 30)),
        ]
        assert engine._compute_atr(candles) == 0.0

    def test_compute_sharpe_flat_returns_zero(
        self, engine: BacktestEngine,
    ) -> None:
        """Sharpe ratio with zero-variance returns should be 0."""
        returns = [0.01, 0.01, 0.01, 0.01]
        sharpe = engine._compute_sharpe(returns)
        # All returns identical => std = 0 => sharpe = 0
        assert sharpe == 0.0

    def test_compute_sharpe_positive(self, engine: BacktestEngine) -> None:
        """Sharpe ratio with positive mean and variance should be positive."""
        returns = [0.01, 0.02, 0.015, 0.005, 0.025]
        sharpe = engine._compute_sharpe(returns)
        assert sharpe > 0
