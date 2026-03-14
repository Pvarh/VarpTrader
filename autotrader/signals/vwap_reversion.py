"""VWAP Mean Reversion signal strategy.

Generates signals when price deviates significantly from the Volume
Weighted Average Price and momentum candles confirm a reversion move
back toward VWAP.

A BUY is triggered when price falls below VWAP by more than the
configured deviation percentage and recent candles show bullish
momentum.  A SELL is triggered symmetrically above VWAP with bearish
momentum confirmation.

The strategy only fires during active market hours: within the first
180 minutes of open *or* within the last 60 minutes before close.

Config keys used:
    enabled            (bool)  : Whether this signal is active.
    vwap_deviation_pct (float) : Required deviation from VWAP as a
                                 percentage (e.g. 0.3 means 0.3%).
                                 Divided by 100 internally (default 0.3).
    momentum_candles   (int)   : Number of recent candles that must confirm
                                 direction (default 3).
    stop_loss_pct      (float) : Stop-loss distance as a fraction of entry
                                 (default 0.01).
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class VWAPReversionSignal(BaseSignal):
    """VWAP mean-reversion signal with candle-momentum confirmation."""

    @property
    def name(self) -> str:
        """Return the unique name of this signal strategy."""
        return "vwap_reversion"

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

        The real logic lives in :meth:`evaluate_from_vwap`.  This method
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

    def evaluate_from_vwap(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        recent_candles: list[OHLCV],
        minutes_since_open: int,
        minutes_before_close: int,
    ) -> SignalResult:
        """Evaluate a VWAP mean-reversion signal.

        Args:
            symbol: Trading symbol (e.g. ``'AAPL'``).
            current_price: Most recent market price.
            vwap: Current session VWAP value.
            recent_candles: Recent OHLCV bars used for momentum check,
                most recent last.
            minutes_since_open: Minutes elapsed since market open.
            minutes_before_close: Minutes remaining until market close.

        Returns:
            A :class:`SignalResult` indicating whether a reversion trade
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
        vwap_deviation_pct: float = self.config.get("vwap_deviation_pct", 0.3)
        deviation_pct = vwap_deviation_pct / 100  # 0.3% -> 0.003
        momentum_candles: int = self.config.get("momentum_candles", 3)
        stop_loss_pct: float = self.config.get("stop_loss_pct", 0.01)

        # --- Guard: session timing -------------------------------------------
        in_early_window = minutes_since_open <= 180
        in_late_window = minutes_before_close <= 60

        if not (in_early_window or in_late_window):
            logger.debug(
                "outside_valid_session_window | signal={signal} symbol={symbol} "
                "minutes_since_open={minutes_since_open} "
                "minutes_before_close={minutes_before_close}",
                signal=self.name, symbol=symbol,
                minutes_since_open=minutes_since_open,
                minutes_before_close=minutes_before_close,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="Outside valid session window",
            )

        # --- Guard: not enough candles for momentum check --------------------
        if len(recent_candles) < momentum_candles:
            logger.debug(
                "insufficient_candles | signal={signal} symbol={symbol} "
                "available={available} required={required}",
                signal=self.name, symbol=symbol,
                available=len(recent_candles), required=momentum_candles,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="Not enough candles for momentum check",
            )

        # --- Deviation & momentum detection ----------------------------------
        lower_band = vwap * (1 - deviation_pct)
        upper_band = vwap * (1 + deviation_pct)

        tail_candles = recent_candles[-momentum_candles:]
        all_bullish = all(c.close > c.open for c in tail_candles)
        all_bearish = all(c.close < c.open for c in tail_candles)

        direction: SignalDirection | None = None
        reason = ""

        if current_price < lower_band and all_bullish:
            direction = SignalDirection.LONG
            reason = (
                f"Price ({current_price:.4f}) below VWAP lower band "
                f"({lower_band:.4f}) with {momentum_candles} bullish candles"
            )
        elif current_price > upper_band and all_bearish:
            direction = SignalDirection.SHORT
            reason = (
                f"Price ({current_price:.4f}) above VWAP upper band "
                f"({upper_band:.4f}) with {momentum_candles} bearish candles"
            )

        if direction is None:
            logger.debug(
                "no_reversion_signal | signal={signal} symbol={symbol} "
                "price={price} vwap={vwap} lower_band={lower_band} "
                "upper_band={upper_band} all_bullish={all_bullish} "
                "all_bearish={all_bearish}",
                signal=self.name, symbol=symbol,
                price=current_price, vwap=vwap,
                lower_band=lower_band, upper_band=upper_band,
                all_bullish=all_bullish, all_bearish=all_bearish,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="No VWAP reversion setup",
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
            take_profit = vwap  # target the VWAP itself
        else:
            stop_loss = entry_price + stop_distance
            take_profit = vwap  # target the VWAP itself

        logger.info(
            "signal_triggered | signal={signal} symbol={symbol} "
            "direction={direction} entry={entry} stop_loss={stop_loss} "
            "take_profit={take_profit} vwap={vwap} "
            "deviation_pct={deviation_pct} momentum_candles={momentum_candles} "
            "minutes_since_open={minutes_since_open} "
            "minutes_before_close={minutes_before_close}",
            signal=self.name, symbol=symbol,
            direction=direction.value, entry=entry_price,
            stop_loss=stop_loss, take_profit=take_profit,
            vwap=vwap, deviation_pct=deviation_pct,
            momentum_candles=momentum_candles,
            minutes_since_open=minutes_since_open,
            minutes_before_close=minutes_before_close,
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
