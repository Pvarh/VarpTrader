"""Bollinger Band Fade signal strategy.

Generates mean-reversion signals when price touches or pierces a
Bollinger Band while the RSI confirms an extreme reading and the
previous candle shows exhaustion in the prevailing direction.

A BUY is triggered at the lower band when RSI is low and the prior
candle was bearish (exhaustion sell-off).  A SELL is triggered at
the upper band when RSI is high and the prior candle was bullish
(exhaustion rally).  The profit target is the middle band (20-period
moving average).

Config keys used:
    enabled              (bool)  : Whether this signal is active.
    rsi_threshold_low    (float) : RSI threshold for lower-band buys (default 30).
    rsi_threshold_high   (float) : RSI threshold for upper-band sells (default 70).
    stop_beyond_band_pct (float) : Stop-loss placed this fraction beyond the
                                   touched band (default 0.005).
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class BollingerFadeSignal(BaseSignal):
    """Bollinger Band fade (mean-reversion) signal."""

    @property
    def name(self) -> str:
        """Return the unique name of this signal strategy."""
        return "bollinger_fade"

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

        The real logic lives in :meth:`evaluate_from_bands`.  This
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
    # Specialised evaluation
    # ------------------------------------------------------------------

    def evaluate_from_bands(
        self,
        symbol: str,
        current_price: float,
        lower_band: float,
        upper_band: float,
        middle_band: float,
        rsi: float,
        prev_candle: OHLCV,
    ) -> SignalResult:
        """Evaluate a Bollinger Band fade signal.

        Args:
            symbol: Trading symbol (e.g. ``'AAPL'`` or ``'SOL/USDT'``).
            current_price: Most recent market price.
            lower_band: Current lower Bollinger Band value.
            upper_band: Current upper Bollinger Band value.
            middle_band: Current middle Bollinger Band (20-MA) value.
            rsi: Current RSI value (0-100).
            prev_candle: The immediately preceding OHLCV bar, used to
                confirm exhaustion (bearish candle at lower band,
                bullish candle at upper band).

        Returns:
            A :class:`SignalResult` indicating whether a fade trade
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
        rsi_threshold_low: float = self.config.get("rsi_threshold_low", 30.0)
        rsi_threshold_high: float = self.config.get("rsi_threshold_high", 70.0)
        stop_beyond_band_pct: float = self.config.get("stop_beyond_band_pct", 0.005)

        # --- Candle-body direction -------------------------------------------
        prev_bearish = prev_candle.close < prev_candle.open
        prev_bullish = prev_candle.close > prev_candle.open

        # --- Signal detection ------------------------------------------------
        direction: SignalDirection | None = None
        reason = ""

        if current_price <= lower_band and rsi < rsi_threshold_low and prev_bearish:
            direction = SignalDirection.LONG
            reason = (
                f"Price ({current_price:.4f}) at/below lower band "
                f"({lower_band:.4f}), RSI={rsi:.1f} < {rsi_threshold_low}, "
                f"prev candle bearish (exhaustion)"
            )
        elif current_price >= upper_band and rsi > rsi_threshold_high and prev_bullish:
            direction = SignalDirection.SHORT
            reason = (
                f"Price ({current_price:.4f}) at/above upper band "
                f"({upper_band:.4f}), RSI={rsi:.1f} > {rsi_threshold_high}, "
                f"prev candle bullish (exhaustion)"
            )

        if direction is None:
            logger.debug(
                "no_fade_signal | signal={signal} symbol={symbol} "
                "price={price} lower_band={lower_band} upper_band={upper_band} "
                "rsi={rsi} prev_bearish={prev_bearish} prev_bullish={prev_bullish}",
                signal=self.name, symbol=symbol,
                price=current_price, lower_band=lower_band,
                upper_band=upper_band, rsi=rsi,
                prev_bearish=prev_bearish, prev_bullish=prev_bullish,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="No Bollinger Band fade setup",
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

        if direction == SignalDirection.LONG:
            stop_loss = entry_price * (1 - stop_beyond_band_pct)
            take_profit = middle_band
        else:
            stop_loss = entry_price * (1 + stop_beyond_band_pct)
            take_profit = middle_band

        logger.info(
            "signal_triggered | signal={signal} symbol={symbol} "
            "direction={direction} entry={entry} stop_loss={stop_loss} "
            "take_profit={take_profit} lower_band={lower_band} "
            "upper_band={upper_band} middle_band={middle_band} rsi={rsi} "
            "prev_candle_open={prev_candle_open} "
            "prev_candle_close={prev_candle_close}",
            signal=self.name, symbol=symbol,
            direction=direction.value, entry=entry_price,
            stop_loss=stop_loss, take_profit=take_profit,
            lower_band=lower_band, upper_band=upper_band,
            middle_band=middle_band, rsi=rsi,
            prev_candle_open=prev_candle.open,
            prev_candle_close=prev_candle.close,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.55,
            strategy_name=self.name,
            reason=reason,
        )
