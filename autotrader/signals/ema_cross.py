"""EMA Golden / Death Cross signal strategy.

Generates signals when a fast exponential moving average crosses a slow
exponential moving average, confirmed by the Relative Strength Index.

A *golden cross* (fast crosses above slow while RSI > 50) triggers a LONG
signal.  A *death cross* (fast crosses below slow while RSI < 50) triggers
a SHORT signal.

Config keys used:
    enabled      (bool)  : Whether this signal is active.
    fast_ema     (int)   : Fast EMA period (default 50).
    slow_ema     (int)   : Slow EMA period (default 200).
    stop_loss_pct (float) : Stop-loss distance as a fraction of entry (default 0.015).
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class EMACrossSignal(BaseSignal):
    """EMA crossover signal with RSI confirmation."""

    @property
    def name(self) -> str:
        """Return the unique name of this signal strategy."""
        return "ema_cross"

    # ------------------------------------------------------------------
    # Abstract base method (required by BaseSignal)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate entry-point.

        The real logic lives in :meth:`evaluate_from_emas`.  This method
        satisfies the abstract interface and always returns a
        non-triggered result.

        Args:
            symbol: Trading symbol.
            candles: Recent OHLCV bars.
            current_price: Latest market price.
            market: ``'stock'`` or ``'crypto'``.

        Returns:
            A :class:`SignalResult` that is never triggered.
        """
        return SignalResult(triggered=False, strategy_name=self.name)

    # ------------------------------------------------------------------
    # Specialised evaluation
    # ------------------------------------------------------------------

    def evaluate_from_emas(
        self,
        symbol: str,
        prev_fast_ema: float,
        prev_slow_ema: float,
        curr_fast_ema: float,
        curr_slow_ema: float,
        rsi: float,
        current_price: float,
        atr: float = 0.0,
    ) -> SignalResult:
        """Evaluate an EMA crossover signal.

        Args:
            symbol: Trading symbol (e.g. ``'AAPL'`` or ``'BTC/USDT'``).
            prev_fast_ema: Fast EMA value on the previous bar.
            prev_slow_ema: Slow EMA value on the previous bar.
            curr_fast_ema: Fast EMA value on the current bar.
            curr_slow_ema: Slow EMA value on the current bar.
            rsi: Current RSI value (0-100).
            current_price: Most recent market price.

        Returns:
            A :class:`SignalResult` indicating whether a crossover trade
            should be taken.
        """
        # --- Guard: disabled ------------------------------------------------
        if not self.is_enabled():
            logger.debug(
                "signal_disabled | signal={signal} symbol={symbol}",
                signal=self.name, symbol=symbol,
            )
            return SignalResult(triggered=False, strategy_name=self.name)

        # --- Config values ---------------------------------------------------
        fast_ema: int = self.config.get("fast_ema", 50)
        slow_ema: int = self.config.get("slow_ema", 200)
        stop_loss_pct: float = self.config.get("stop_loss_pct", 0.015)
        atr_stop_mult: float = self.config.get("atr_stop_multiplier", 1.5)

        # --- Cross detection -------------------------------------------------
        golden_cross = prev_fast_ema < prev_slow_ema and curr_fast_ema > curr_slow_ema
        death_cross = prev_fast_ema > prev_slow_ema and curr_fast_ema < curr_slow_ema

        direction: SignalDirection | None = None
        reason = ""

        if golden_cross and rsi > 50:
            direction = SignalDirection.LONG
            reason = (
                f"Golden cross detected (fast EMA {curr_fast_ema:.4f} "
                f"crossed above slow EMA {curr_slow_ema:.4f}), RSI={rsi:.1f}"
            )
        elif death_cross and rsi < 50:
            direction = SignalDirection.SHORT
            reason = (
                f"Death cross detected (fast EMA {curr_fast_ema:.4f} "
                f"crossed below slow EMA {curr_slow_ema:.4f}), RSI={rsi:.1f}"
            )

        if direction is None:
            if golden_cross or death_cross:
                logger.debug(
                    "cross_without_rsi_confirmation | signal={signal} "
                    "symbol={symbol} golden_cross={golden_cross} "
                    "death_cross={death_cross} rsi={rsi}",
                    signal=self.name, symbol=symbol,
                    golden_cross=golden_cross, death_cross=death_cross,
                    rsi=rsi,
                )
            else:
                logger.debug(
                    "no_cross_detected | signal={signal} symbol={symbol}",
                    signal=self.name, symbol=symbol,
                )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="No confirmed EMA cross",
            )

        # --- Swing bias check ------------------------------------------------
        if not self.check_swing_bias(symbol, direction):
            logger.info(
                "swing_bias_blocked | signal={signal} symbol={symbol} "
                "direction={direction}",
                signal=self.name, symbol=symbol,
                direction=direction.value,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="Blocked by swing bias",
            )

        # --- Price levels ----------------------------------------------------
        entry_price = current_price
        stop_distance = atr * atr_stop_mult if atr > 0 else entry_price * stop_loss_pct

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 2)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 2)

        logger.info(
            "signal_triggered | signal={signal} symbol={symbol} "
            "direction={direction} entry={entry} stop_loss={stop_loss} "
            "take_profit={take_profit} fast_ema={fast_ema} slow_ema={slow_ema} "
            "prev_fast_ema={prev_fast_ema} prev_slow_ema={prev_slow_ema} "
            "curr_fast_ema={curr_fast_ema} curr_slow_ema={curr_slow_ema} rsi={rsi}",
            signal=self.name, symbol=symbol,
            direction=direction.value, entry=entry_price,
            stop_loss=stop_loss, take_profit=take_profit,
            fast_ema=fast_ema, slow_ema=slow_ema,
            prev_fast_ema=prev_fast_ema, prev_slow_ema=prev_slow_ema,
            curr_fast_ema=curr_fast_ema, curr_slow_ema=curr_slow_ema,
            rsi=rsi,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.65,
            strategy_name=self.name,
            reason=reason,
        )
