"""Multi-timeframe RSI Momentum signal strategy.

Generates signals when the Relative Strength Index on both a short
timeframe (5-minute) and a longer timeframe (1-hour) agree on an
extreme condition.

A BUY is triggered when RSI is oversold on *both* timeframes.  A SELL
is triggered when RSI is overbought on *both* timeframes.  If the
1-hour RSI sits in the neutral zone the strategy abstains.

Config keys used:
    enabled           (bool)  : Whether this signal is active.
    rsi_oversold      (float) : 5-min RSI oversold threshold (default 30).
    rsi_overbought    (float) : 5-min RSI overbought threshold (default 70).
    rsi_neutral_low   (float) : 1-hour RSI neutral-zone lower bound (default 40).
    rsi_neutral_high  (float) : 1-hour RSI neutral-zone upper bound (default 60).
    stop_loss_pct     (float) : Stop-loss distance as a fraction of entry
                                (default 0.015).
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class RSIMomentumSignal(BaseSignal):
    """Multi-timeframe RSI momentum signal."""

    @property
    def name(self) -> str:
        """Return the unique name of this signal strategy."""
        return "rsi_momentum"

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

        The real logic lives in :meth:`evaluate_from_rsi`.  This method
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

    def evaluate_from_rsi(
        self,
        symbol: str,
        rsi_5m: float,
        rsi_1h: float,
        current_price: float,
    ) -> SignalResult:
        """Evaluate a multi-timeframe RSI signal.

        Args:
            symbol: Trading symbol (e.g. ``'AAPL'`` or ``'ETH/USDT'``).
            rsi_5m: RSI value computed on 5-minute bars (0-100).
            rsi_1h: RSI value computed on 1-hour bars (0-100).
            current_price: Most recent market price.

        Returns:
            A :class:`SignalResult` indicating whether an RSI-based
            trade should be taken.
        """
        # --- Guard: disabled ------------------------------------------------
        if not self.is_enabled():
            logger.debug(
                "signal_disabled | signal={signal} symbol={symbol}",
                signal=self.name, symbol=symbol,
            )
            return SignalResult(triggered=False, strategy_name=self.name)

        # --- Config values ---------------------------------------------------
        rsi_oversold: float = self.config.get("rsi_oversold", 30.0)
        rsi_overbought: float = self.config.get("rsi_overbought", 70.0)
        rsi_neutral_low: float = self.config.get("rsi_neutral_low", 40.0)
        rsi_neutral_high: float = self.config.get("rsi_neutral_high", 60.0)
        stop_loss_pct: float = self.config.get("stop_loss_pct", 0.015)

        # --- Guard: 1-hour RSI in neutral zone -------------------------------
        if rsi_neutral_low <= rsi_1h <= rsi_neutral_high:
            logger.debug(
                "rsi_1h_neutral_zone | signal={signal} symbol={symbol} "
                "rsi_1h={rsi_1h} rsi_neutral_low={rsi_neutral_low} "
                "rsi_neutral_high={rsi_neutral_high}",
                signal=self.name, symbol=symbol,
                rsi_1h=rsi_1h,
                rsi_neutral_low=rsi_neutral_low,
                rsi_neutral_high=rsi_neutral_high,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="1-hour RSI in neutral zone",
            )

        # --- Signal detection ------------------------------------------------
        direction: SignalDirection | None = None
        reason = ""

        if rsi_5m < rsi_oversold and rsi_1h < rsi_neutral_low:
            direction = SignalDirection.LONG
            reason = (
                f"Multi-TF oversold: RSI-5m={rsi_5m:.1f} < {rsi_oversold}, "
                f"RSI-1h={rsi_1h:.1f} < {rsi_neutral_low}"
            )
        elif rsi_5m > rsi_overbought and rsi_1h > rsi_neutral_high:
            direction = SignalDirection.SHORT
            reason = (
                f"Multi-TF overbought: RSI-5m={rsi_5m:.1f} > {rsi_overbought}, "
                f"RSI-1h={rsi_1h:.1f} > {rsi_neutral_high}"
            )

        if direction is None:
            logger.debug(
                "no_rsi_extreme | signal={signal} symbol={symbol} "
                "rsi_5m={rsi_5m} rsi_1h={rsi_1h}",
                signal=self.name, symbol=symbol,
                rsi_5m=rsi_5m, rsi_1h=rsi_1h,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="RSI values not at dual-timeframe extremes",
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
        stop_distance = entry_price * stop_loss_pct

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 2)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 2)

        logger.info(
            "signal_triggered | signal={signal} symbol={symbol} "
            "direction={direction} entry={entry} stop_loss={stop_loss} "
            "take_profit={take_profit} rsi_5m={rsi_5m} rsi_1h={rsi_1h}",
            signal=self.name, symbol=symbol,
            direction=direction.value, entry=entry_price,
            stop_loss=stop_loss, take_profit=take_profit,
            rsi_5m=rsi_5m, rsi_1h=rsi_1h,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.6,
            strategy_name=self.name,
            reason=reason,
        )
