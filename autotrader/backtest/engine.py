"""Backtest engine for replaying historical data through signal strategies.

The engine iterates over OHLCV candles, evaluates configured signal
strategies, manages simulated positions with stop-loss / take-profit
exits, and computes comprehensive performance statistics.

The engine is **standalone** -- it requires no API keys, no network
access, and no broker connection.  All that is needed is a list of
:class:`~journal.models.OHLCV` candles and one or more
:class:`~signals.base_signal.BaseSignal` instances.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from journal.models import OHLCV, Trade
from signals.base_signal import BaseSignal, SignalResult, SignalDirection
from risk.position_sizer import PositionSizer
from risk.reward_ratio import RewardRatioGate


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Results of a backtest run.

    Stores trade-level detail, aggregate statistics, and the full equity
    curve so the caller can analyse or plot the run.
    """

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable formatted summary string."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Total Trades:     {self.total_trades}",
            f"Wins:             {self.wins}",
            f"Losses:           {self.losses}",
            f"Breakeven:        {self.breakeven}",
            f"Win Rate:         {self.win_rate:.1f}%",
            f"Total PnL:        ${self.total_pnl:,.2f}",
            f"Avg Win:          ${self.avg_win:,.2f}",
            f"Avg Loss:         ${self.avg_loss:,.2f}",
            f"Max Drawdown:     ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.1f}%)",
            f"Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"Profit Factor:    {self.profit_factor:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal position tracker
# ---------------------------------------------------------------------------


@dataclass
class _OpenPosition:
    """Internal representation of an open position during a backtest run."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    strategy: str
    entry_timestamp: datetime


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Replay historical data through signal strategies and measure performance.

    The engine processes candles one-by-one in chronological order.  On
    every bar it:

    1. Checks whether any open position has hit its stop-loss or
       take-profit.
    2. Evaluates each signal strategy, computing the required indicators
       (RSI, EMA, Bollinger Bands, VWAP) internally.
    3. If a signal fires, validates the risk:reward ratio, sizes the
       position, and opens a simulated trade.
    4. Records the equity curve and tracks drawdown.

    At the end of the run, any remaining open positions are closed at
    the last candle's close price and full statistics are returned in a
    :class:`BacktestResult`.
    """

    def __init__(self, config: dict, initial_capital: float = 100_000.0) -> None:
        """Initialise with trading config and starting capital.

        Args:
            config: Full trading configuration dictionary (must contain
                at least a ``risk`` key with the fields expected by
                :class:`~risk.position_sizer.PositionSizer` and a
                ``min_reward_ratio`` entry).
            initial_capital: Starting account equity in dollars.
        """
        self._config = config
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._position_sizer = PositionSizer(config["risk"])
        self._reward_gate = RewardRatioGate(config["risk"]["min_reward_ratio"])

        # Per-run state (reset on each call to ``run``)
        self._open_positions: list[_OpenPosition] = []
        self._closed_trades: list[dict] = []
        self._equity_curve: list[float] = []
        self._peak_equity: float = initial_capital

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        candles: list[OHLCV],
        signals: list[BaseSignal],
        candles_1h: list[OHLCV] | None = None,
    ) -> BacktestResult:
        """Run backtest over candle data with given signals.

        For each candle the engine:

        1. Checks if any open positions hit SL or TP.
        2. Evaluates each signal strategy.
        3. If a signal is triggered, validates R:R, sizes position, and
           records the trade.
        4. Tracks the equity curve and drawdown.

        Args:
            candles: Primary-timeframe candles (e.g. 5 m), sorted
                chronologically.
            signals: List of signal strategies to evaluate.
            candles_1h: Optional 1-hour candles for multi-timeframe
                signals such as :class:`RSIMomentumSignal` or
                :class:`EMACrossSignal`.

        Returns:
            A :class:`BacktestResult` with full statistics and trade list.
        """
        log = logger.bind(module="backtest_engine")

        # Reset per-run state
        self._capital = self._initial_capital
        self._open_positions = []
        self._closed_trades = []
        self._equity_curve = [self._initial_capital]
        self._peak_equity = self._initial_capital

        if not candles:
            log.warning("no_candles_provided")
            return BacktestResult(equity_curve=[self._initial_capital])

        max_positions: int = (
            self._config.get("trading", {}).get("max_positions", 999)
        )

        for i, candle in enumerate(candles):
            # 1. Check exits on open positions
            self._check_exits(candle)

            # 2. Evaluate signals
            candles_so_far = candles[: i + 1]

            # Determine available 1h candles up to current time
            candles_1h_so_far: list[OHLCV] | None = None
            if candles_1h:
                candles_1h_so_far = [
                    c for c in candles_1h if c.timestamp <= candle.timestamp
                ]

            if len(self._open_positions) < max_positions:
                active_strategies = {
                    pos.strategy for pos in self._open_positions
                }

                for signal in signals:
                    if not signal.is_enabled():
                        continue

                    # One position per strategy at a time
                    if signal.name in active_strategies:
                        continue

                    result = self._evaluate_signal(
                        signal, candle, candles_so_far, candles_1h_so_far,
                    )

                    if not result.triggered:
                        continue

                    # Validate R:R
                    if not self._reward_gate.check(
                        result.entry_price,
                        result.stop_loss,
                        result.take_profit,
                    ):
                        log.debug(
                            "trade_rejected_rr",
                            signal=signal.name,
                            entry=result.entry_price,
                            sl=result.stop_loss,
                            tp=result.take_profit,
                        )
                        continue

                    # Compute ATR for position sizing
                    atr = self._compute_atr(candles_so_far)

                    # Size position
                    qty = self._position_sizer.calculate_size(
                        self._capital, result.entry_price, atr,
                    )

                    # Open position
                    pos = _OpenPosition(
                        symbol=result.symbol or candle.symbol,
                        direction=result.direction.value,
                        entry_price=result.entry_price,
                        quantity=qty,
                        stop_loss=result.stop_loss,
                        take_profit=result.take_profit,
                        strategy=result.strategy_name,
                        entry_timestamp=candle.timestamp,
                    )
                    self._open_positions.append(pos)
                    active_strategies.add(signal.name)

                    log.info(
                        "position_opened",
                        signal=signal.name,
                        direction=pos.direction,
                        entry=pos.entry_price,
                        qty=pos.quantity,
                        sl=pos.stop_loss,
                        tp=pos.take_profit,
                    )

            # 3. Update equity curve
            unrealized = self._compute_unrealized_pnl(candle)
            current_equity = self._capital + unrealized
            self._equity_curve.append(current_equity)

            if current_equity > self._peak_equity:
                self._peak_equity = current_equity

        # Force-close any remaining open positions at last candle's close
        if self._open_positions and candles:
            last_candle = candles[-1]
            for pos in self._open_positions[:]:
                self._close_position(
                    pos, last_candle.close, last_candle.timestamp,
                )
            self._open_positions.clear()

        return self._build_result()

    # ------------------------------------------------------------------ #
    # Signal dispatch
    # ------------------------------------------------------------------ #

    def _evaluate_signal(
        self,
        signal: BaseSignal,
        candle: OHLCV,
        candles_so_far: list[OHLCV],
        candles_1h_so_far: list[OHLCV] | None,
    ) -> SignalResult:
        """Route evaluation to the signal's specialised method.

        The engine imports each concrete signal class and uses
        ``isinstance`` to determine which specialised evaluation method
        to call.  For unknown signal types the generic
        :meth:`BaseSignal.evaluate` is used as a fallback.

        Args:
            signal: The signal strategy to evaluate.
            candle: The current candle being processed.
            candles_so_far: All candles up to and including *candle*.
            candles_1h_so_far: Available 1-hour candles (may be ``None``).

        Returns:
            A :class:`SignalResult`.
        """
        # Lazy imports to avoid hard dependency at module level
        from signals.rsi_momentum import RSIMomentumSignal
        from signals.ema_cross import EMACrossSignal
        from signals.ema_pullback import EMAPullbackSignal
        from signals.bollinger_fade import BollingerFadeSignal
        from signals.vwap_reversion import VWAPReversionSignal
        from signals.first_candle import FirstCandleSignal

        closes = [c.close for c in candles_so_far]
        no_trigger = SignalResult(triggered=False, strategy_name=signal.name)

        # ---- RSI Momentum ------------------------------------------------
        if isinstance(signal, RSIMomentumSignal):
            rsi_period: int = signal.config.get("rsi_period", 14)
            if len(closes) < rsi_period + 1:
                return no_trigger

            rsi_5m = self._compute_rsi(closes, rsi_period)

            rsi_1h = 50.0  # neutral default
            if candles_1h_so_far and len(candles_1h_so_far) > rsi_period:
                closes_1h = [c.close for c in candles_1h_so_far]
                rsi_1h = self._compute_rsi(closes_1h, rsi_period)

            return signal.evaluate_from_rsi(
                symbol=candle.symbol,
                rsi_5m=rsi_5m,
                rsi_1h=rsi_1h,
                current_price=candle.close,
            )

        # ---- EMA Cross ---------------------------------------------------
        if isinstance(signal, EMACrossSignal):
            fast_period: int = signal.config.get("fast_period", 50)
            slow_period: int = signal.config.get("slow_period", 200)

            # Use 1h candles if configured and available
            ema_timeframe = signal.config.get("timeframe", "5m")
            if ema_timeframe in ("1h", "1H") and candles_1h_so_far:
                ema_closes = [c.close for c in candles_1h_so_far]
            else:
                ema_closes = closes

            if len(ema_closes) < slow_period + 1:
                return no_trigger

            prev_ema_closes = ema_closes[:-1]
            prev_fast = self._compute_ema(prev_ema_closes, fast_period)
            prev_slow = self._compute_ema(prev_ema_closes, slow_period)
            curr_fast = self._compute_ema(ema_closes, fast_period)
            curr_slow = self._compute_ema(ema_closes, slow_period)

            rsi = (
                self._compute_rsi(closes, 14)
                if len(closes) > 14
                else 50.0
            )

            return signal.evaluate_from_emas(
                symbol=candle.symbol,
                prev_fast_ema=prev_fast,
                prev_slow_ema=prev_slow,
                curr_fast_ema=curr_fast,
                curr_slow_ema=curr_slow,
                rsi=rsi,
                current_price=candle.close,
            )

        # ---- Bollinger Fade ----------------------------------------------
        if isinstance(signal, BollingerFadeSignal):
            bb_period: int = signal.config.get("bb_period", 20)
            bb_std: float = signal.config.get("bb_std", 2.0)

            if len(closes) < bb_period or len(candles_so_far) < 2:
                return no_trigger

            window = closes[-bb_period:]
            middle = sum(window) / bb_period
            variance = sum((x - middle) ** 2 for x in window) / bb_period
            std_dev = math.sqrt(variance)
            upper = middle + bb_std * std_dev
            lower = middle - bb_std * std_dev

            rsi = (
                self._compute_rsi(closes, 14)
                if len(closes) > 14
                else 50.0
            )
            prev_candle = candles_so_far[-2]

            return signal.evaluate_from_bands(
                symbol=candle.symbol,
                current_price=candle.close,
                lower_band=lower,
                upper_band=upper,
                middle_band=middle,
                rsi=rsi,
                prev_candle=prev_candle,
            )

        # ---- VWAP Reversion ----------------------------------------------
        if isinstance(signal, VWAPReversionSignal):
            momentum_candles: int = signal.config.get("momentum_candles", 3)
            if len(candles_so_far) < momentum_candles:
                return no_trigger

            vwap = self._compute_vwap(candles_so_far)

            # Approximate session timing
            market_open_min = 9 * 60 + 30   # 09:30
            market_close_min = 16 * 60       # 16:00
            candle_min = candle.timestamp.hour * 60 + candle.timestamp.minute
            minutes_since_open = max(0, candle_min - market_open_min)
            minutes_before_close = max(0, market_close_min - candle_min)

            return signal.evaluate_from_vwap(
                symbol=candle.symbol,
                current_price=candle.close,
                vwap=vwap,
                recent_candles=candles_so_far[-momentum_candles:],
                minutes_since_open=minutes_since_open,
                minutes_before_close=minutes_before_close,
            )

        # ---- First Candle ------------------------------------------------
        if isinstance(signal, FirstCandleSignal):
            if len(candles_so_far) < 2:
                return no_trigger

            first_candle = candles_so_far[0]
            volumes = [c.volume for c in candles_so_far]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0.0

            market_open_min = 9 * 60 + 30
            candle_min = candle.timestamp.hour * 60 + candle.timestamp.minute
            minutes_since_open = max(0, candle_min - market_open_min)

            return signal.evaluate_with_context(
                symbol=candle.symbol,
                first_candle=first_candle,
                current_candle=candle,
                avg_volume=avg_volume,
                minutes_since_open=minutes_since_open,
            )

        # ---- EMA Pullback ------------------------------------------------
        if isinstance(signal, EMAPullbackSignal):
            fast_period: int = signal.config.get("fast_ema", 20)
            slow_period: int = signal.config.get("slow_ema", 50)

            ema_closes = (
                [c.close for c in candles_1h_so_far]
                if candles_1h_so_far and len(candles_1h_so_far) >= slow_period
                else closes
            )
            if len(ema_closes) < slow_period:
                return no_trigger

            ema_fast = self._compute_ema(ema_closes, fast_period)
            ema_slow = self._compute_ema(ema_closes, slow_period)

            if signal.config.get("volume_confirmation", True):
                ema_candles = candles_1h_so_far if candles_1h_so_far and len(candles_1h_so_far) >= slow_period else candles_so_far
                volumes = [c.volume for c in ema_candles]
                if len(volumes) >= 20:
                    avg_vol = sum(volumes[-20:]) / 20
                    if volumes[-1] <= avg_vol:
                        return no_trigger

            rsi = self._compute_rsi(closes, 14) if len(closes) > 14 else 50.0

            return signal.evaluate_pullback(
                symbol=candle.symbol,
                current_price=candle.close,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rsi=rsi,
            )

        # ---- Fallback: generic evaluate ----------------------------------
        return signal.evaluate(
            candle.symbol, candles_so_far, candle.close, candle.market,
        )

    # ------------------------------------------------------------------ #
    # Exit management
    # ------------------------------------------------------------------ #

    def _check_exits(self, candle: OHLCV) -> None:
        """Check if any open positions hit SL/TP on *candle*.

        When both stop-loss and take-profit could theoretically be hit
        within the same candle the engine assumes the **stop-loss was
        hit first** (conservative approach).

        Args:
            candle: The current OHLCV bar.
        """
        for pos in self._open_positions[:]:
            sl_hit = False
            tp_hit = False

            if pos.direction == "long":
                sl_hit = candle.low <= pos.stop_loss
                tp_hit = candle.high >= pos.take_profit
            else:
                sl_hit = candle.high >= pos.stop_loss
                tp_hit = candle.low <= pos.take_profit

            if sl_hit and tp_hit:
                # Conservative: assume SL was hit first
                self._close_position(pos, pos.stop_loss, candle.timestamp)
                self._open_positions.remove(pos)
            elif sl_hit:
                self._close_position(pos, pos.stop_loss, candle.timestamp)
                self._open_positions.remove(pos)
            elif tp_hit:
                self._close_position(pos, pos.take_profit, candle.timestamp)
                self._open_positions.remove(pos)

    def _close_position(
        self,
        pos: _OpenPosition,
        exit_price: float,
        exit_timestamp: datetime,
    ) -> None:
        """Close a position and record the trade.

        Args:
            pos: The open position to close.
            exit_price: The price at which the position is exited.
            exit_timestamp: Timestamp of the exit.
        """
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        notional = pos.entry_price * pos.quantity
        pnl_pct = pnl / notional if notional != 0 else 0.0

        if pnl > 0:
            outcome = "win"
        elif pnl < 0:
            outcome = "loss"
        else:
            outcome = "breakeven"

        trade_record: dict = {
            "symbol": pos.symbol,
            "direction": pos.direction,
            "strategy": pos.strategy,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
            "entry_timestamp": pos.entry_timestamp.isoformat(),
            "exit_timestamp": exit_timestamp.isoformat(),
        }

        self._closed_trades.append(trade_record)
        self._capital += pnl

        logger.info(
            "position_closed",
            symbol=pos.symbol,
            direction=pos.direction,
            entry=pos.entry_price,
            exit=exit_price,
            pnl=round(pnl, 2),
            outcome=outcome,
        )

    def _compute_unrealized_pnl(self, candle: OHLCV) -> float:
        """Return total mark-to-market PnL for all open positions.

        Args:
            candle: The candle whose close price is used for marking.

        Returns:
            Combined unrealized PnL across all open positions.
        """
        total = 0.0
        for pos in self._open_positions:
            if pos.direction == "long":
                total += (candle.close - pos.entry_price) * pos.quantity
            else:
                total += (pos.entry_price - candle.close) * pos.quantity
        return total

    # ------------------------------------------------------------------ #
    # Indicator computations
    # ------------------------------------------------------------------ #

    def _compute_rsi(self, closes: list[float], period: int = 14) -> float:
        """Compute the Relative Strength Index from close prices.

        Uses Wilder's smoothing method: the first average is a simple
        mean, and subsequent averages are exponentially smoothed with
        ``alpha = 1 / period``.

        Args:
            closes: List of close prices, oldest first.
            period: RSI look-back period (default 14).

        Returns:
            RSI value between 0 and 100.  Returns 50.0 (neutral) when
            there are not enough data points.
        """
        if len(closes) < period + 1:
            return 50.0

        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        gains = [max(0.0, c) for c in changes]
        losses = [max(0.0, -c) for c in changes]

        # Initial averages (SMA over first *period* values)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder's smoothing for remaining values
        for idx in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
            avg_loss = (avg_loss * (period - 1) + losses[idx]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_ema(self, closes: list[float], period: int) -> float:
        """Compute the Exponential Moving Average for the latest value.

        The first *period* values are seeded with a simple moving
        average; subsequent values use the standard EMA formula with
        multiplier ``2 / (period + 1)``.

        Args:
            closes: List of close prices, oldest first.
            period: EMA look-back period.

        Returns:
            Current EMA value.  If fewer than *period* data points are
            available the last close is returned as a fallback.
        """
        if not closes:
            return 0.0
        if len(closes) < period:
            return closes[-1]

        multiplier = 2.0 / (period + 1)
        ema = sum(closes[:period]) / period

        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _compute_atr(
        self, candles: list[OHLCV], period: int = 14,
    ) -> float:
        """Compute the Average True Range.

        Args:
            candles: OHLCV candles, oldest first.
            period: ATR look-back period (default 14).

        Returns:
            Average True Range value.  Returns ``0.0`` when fewer than
            two candles are available.
        """
        if len(candles) < 2:
            return 0.0

        true_ranges: list[float] = []
        for idx in range(1, len(candles)):
            high_low = candles[idx].high - candles[idx].low
            high_close = abs(candles[idx].high - candles[idx - 1].close)
            low_close = abs(candles[idx].low - candles[idx - 1].close)
            true_ranges.append(max(high_low, high_close, low_close))

        if not true_ranges:
            return 0.0

        recent = (
            true_ranges[-period:]
            if len(true_ranges) >= period
            else true_ranges
        )
        return sum(recent) / len(recent)

    def _compute_vwap(self, candles: list[OHLCV]) -> float:
        """Compute the Volume Weighted Average Price.

        Args:
            candles: OHLCV candles over which to compute VWAP.

        Returns:
            VWAP value.  If total volume is zero the last close price
            is returned.
        """
        cum_vol = 0.0
        cum_tp_vol = 0.0

        for c in candles:
            typical_price = (c.high + c.low + c.close) / 3.0
            cum_tp_vol += typical_price * c.volume
            cum_vol += c.volume

        if cum_vol == 0:
            return candles[-1].close if candles else 0.0

        return cum_tp_vol / cum_vol

    # ------------------------------------------------------------------ #
    # Aggregate statistics
    # ------------------------------------------------------------------ #

    def _compute_drawdown(self) -> tuple[float, float]:
        """Compute maximum drawdown in dollars and as a percentage.

        Returns:
            A ``(max_drawdown_dollars, max_drawdown_pct)`` tuple.
        """
        if not self._equity_curve:
            return 0.0, 0.0

        peak = self._equity_curve[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for equity in self._equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0

        return max_dd, max_dd_pct

    def _compute_sharpe(self, returns: list[float]) -> float:
        """Compute an annualised Sharpe ratio from per-bar returns.

        Assumes 5-minute bars with approximately 78 bars per trading
        day and 252 trading days per year, giving an annualisation
        factor of ``sqrt(78 * 252)``.

        Args:
            returns: List of fractional per-bar returns.

        Returns:
            Annualised Sharpe ratio.  Returns ``0.0`` when there are
            fewer than two observations or when standard deviation is
            zero.
        """
        if len(returns) < 2:
            return 0.0

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (
            len(returns) - 1
        )
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return 0.0

        # Annualise assuming 5-min bars
        annualisation = math.sqrt(252 * 78)
        return (mean_ret / std_dev) * annualisation

    def _build_result(self) -> BacktestResult:
        """Compile all closed trades and equity curve into a :class:`BacktestResult`."""
        result = BacktestResult()
        result.trades = self._closed_trades
        result.equity_curve = self._equity_curve
        result.total_trades = len(self._closed_trades)

        if result.total_trades == 0:
            return result

        wins = [t for t in self._closed_trades if t["outcome"] == "win"]
        losses = [t for t in self._closed_trades if t["outcome"] == "loss"]
        breakevens = [
            t for t in self._closed_trades if t["outcome"] == "breakeven"
        ]

        result.wins = len(wins)
        result.losses = len(losses)
        result.breakeven = len(breakevens)
        result.total_pnl = sum(t["pnl"] for t in self._closed_trades)
        result.win_rate = (
            (result.wins / result.total_trades * 100.0)
            if result.total_trades > 0
            else 0.0
        )

        result.avg_win = (
            sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
        )
        result.avg_loss = (
            sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
        )

        # Drawdown
        result.max_drawdown, result.max_drawdown_pct = self._compute_drawdown()

        # Sharpe ratio from equity-curve returns
        if len(self._equity_curve) > 1:
            eq = self._equity_curve
            returns = [
                (eq[i] - eq[i - 1]) / eq[i - 1]
                for i in range(1, len(eq))
                if eq[i - 1] != 0
            ]
            result.sharpe_ratio = self._compute_sharpe(returns)

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            result.profit_factor = float("inf")
        else:
            result.profit_factor = 0.0

        return result
