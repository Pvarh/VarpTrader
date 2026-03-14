"""Opening Range Breakout signal strategy -- 60-minute ORB.

Monitors the 60-minute opening range (9:30-10:30 EST) and generates
signals when price breaks above the high or below the low of that
range with confirming volume.  The breakout must occur within the
valid window (60-210 minutes after open, i.e. 10:30 to 13:00 EST).

An additional volatility guard rejects the setup when the ORB range
exceeds twice the 20-day Average True Range.

Config keys used:
    enabled            (bool)  : Whether this signal is active.
    volume_multiplier  (float) : Required volume multiple above 20-day
                                 average volume (default 1.5).
    orb_window_minutes (int)   : Length of the opening-range window in
                                 minutes (default 60).
    valid_until_hour   (int)   : Last hour (EST, 24-h) during which a
                                 breakout is valid (default 13).
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class FirstCandleSignal(BaseSignal):
    """Opening Range Breakout based on the session's 60-minute opening range."""

    @property
    def name(self) -> str:
        """Return the unique name of this signal strategy."""
        return "first_candle"

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

        The real logic lives in :meth:`evaluate_with_context` which
        accepts the ORB high/low and context data explicitly.  This
        method satisfies the abstract interface and always returns a
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
    # Specialised evaluation -- 60-minute ORB
    # ------------------------------------------------------------------

    def evaluate_with_context(
        self,
        symbol: str,
        orb_high: float,
        orb_low: float,
        current_candle: OHLCV,
        avg_volume: float,
        minutes_since_open: int,
        atr_20d: float,
    ) -> SignalResult:
        """Evaluate a 60-minute Opening Range Breakout signal.

        Args:
            symbol: Trading symbol (e.g. ``'AAPL'``).
            orb_high: High of the 60-minute opening range (9:30-10:30 EST).
            orb_low: Low of the 60-minute opening range.
            current_candle: The most recent 5-minute OHLCV bar.
            avg_volume: 20-day average volume baseline.
            minutes_since_open: Minutes elapsed since market open.
            atr_20d: 20-day Average True Range (used for volatility guard).

        Returns:
            A :class:`SignalResult` indicating whether a breakout trade
            should be taken, and if so, the direction, entry, stop-loss,
            and take-profit levels.
        """
        # --- Guard: disabled ------------------------------------------------
        if not self.is_enabled():
            logger.debug(
                "signal_disabled | signal={signal} symbol={symbol}",
                signal=self.name, symbol=symbol,
            )
            return SignalResult(triggered=False, strategy_name=self.name)

        # --- Config values ---------------------------------------------------
        volume_multiplier: float = self.config.get("volume_multiplier", 1.5)

        # --- Guard: ORB range too wide (too volatile) ------------------------
        orb_range = orb_high - orb_low
        if orb_range > 2 * atr_20d:
            logger.debug(
                "orb_range_too_wide | signal={signal} symbol={symbol} "
                "orb_range={orb_range} atr_limit={atr_limit}",
                signal=self.name, symbol=symbol,
                orb_range=orb_range, atr_limit=2 * atr_20d,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="ORB range too wide relative to ATR",
            )

        # --- Guard: valid window 60-210 minutes (10:30 - 13:00 EST) ----------
        if minutes_since_open < 60 or minutes_since_open > 210:
            logger.debug(
                "outside_valid_window | signal={signal} symbol={symbol} "
                "minutes_since_open={minutes_since_open}",
                signal=self.name, symbol=symbol,
                minutes_since_open=minutes_since_open,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="Outside valid breakout window (60-210 min)",
            )

        # --- Volume confirmation ---------------------------------------------
        volume_threshold = avg_volume * volume_multiplier
        volume_confirmed = current_candle.volume > volume_threshold

        if not volume_confirmed:
            logger.debug(
                "volume_not_confirmed | signal={signal} symbol={symbol} "
                "current_volume={current_volume} threshold={threshold}",
                signal=self.name, symbol=symbol,
                current_volume=current_candle.volume,
                threshold=volume_threshold,
            )

        # --- Breakout detection ----------------------------------------------
        entry_price = current_candle.close
        direction: SignalDirection | None = None
        reason = ""

        if current_candle.close > orb_high and volume_confirmed:
            direction = SignalDirection.LONG
            reason = (
                f"Breakout above 60-min ORB high "
                f"({orb_high:.4f}) with volume confirmation"
            )
        elif current_candle.close < orb_low and volume_confirmed:
            direction = SignalDirection.SHORT
            reason = (
                f"Breakout below 60-min ORB low "
                f"({orb_low:.4f}) with volume confirmation"
            )

        if direction is None:
            logger.debug(
                "no_breakout_detected | signal={signal} symbol={symbol}",
                signal=self.name, symbol=symbol,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="No breakout or insufficient volume",
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
        if direction == SignalDirection.LONG:
            stop_loss = orb_low
            stop_distance = entry_price - orb_low
            take_profit = entry_price + (stop_distance * 2)
        else:
            stop_loss = orb_high
            stop_distance = orb_high - entry_price
            take_profit = entry_price - (stop_distance * 2)

        logger.info(
            "signal_triggered | signal={signal} symbol={symbol} "
            "direction={direction} entry={entry} stop_loss={stop_loss} "
            "take_profit={take_profit} orb_high={orb_high} orb_low={orb_low} "
            "current_volume={current_volume} avg_volume={avg_volume} "
            "minutes_since_open={minutes_since_open}",
            signal=self.name, symbol=symbol,
            direction=direction.value, entry=entry_price,
            stop_loss=stop_loss, take_profit=take_profit,
            orb_high=orb_high, orb_low=orb_low,
            current_volume=current_candle.volume,
            avg_volume=avg_volume,
            minutes_since_open=minutes_since_open,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.7,
            strategy_name=self.name,
            reason=reason,
        )
