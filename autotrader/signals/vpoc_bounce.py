"""VPOC Bounce signal — trade bounces off the Volume Point of Control."""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult
from signals.indicators import Indicators


class VPOCBounceSignal(BaseSignal):
    """Detect price bouncing off the session's highest-volume price level.

    Builds a volume profile from session candles, identifies the POC,
    and triggers when price is near POC with directional bounce candles.
    """

    @property
    def name(self) -> str:
        return "vpoc_bounce"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used directly. See evaluate_from_profile()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_from_profile(
        self,
        symbol: str,
        current_price: float,
        session_candles: list[OHLCV],
        recent_candles: list[OHLCV],
        market: str,
    ) -> SignalResult:
        """Evaluate VPOC bounce from session volume profile."""
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        num_bins = self.config.get("num_bins", 20)
        proximity_pct = self.config.get("proximity_pct", 0.002)
        bounce_candles = self.config.get("bounce_candles", 2)
        min_poc_vol = self.config.get("min_poc_volume_pct", 0.15)
        sl_pct = self.config.get("stop_loss_pct", 0.01)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        if not session_candles or len(recent_candles) < bounce_candles:
            return no_signal

        # Build volume profile
        poc_price, vah, val = Indicators.volume_profile(session_candles, num_bins=num_bins)
        if poc_price == 0.0:
            return no_signal

        # Check POC strength
        total_vol = sum(c.volume for c in session_candles)
        if total_vol == 0:
            return no_signal

        # Compute POC bin volume
        highs = [c.high for c in session_candles]
        lows = [c.low for c in session_candles]
        price_high = max(highs)
        price_low = min(lows)
        if price_high == price_low:
            return no_signal

        bin_width = (price_high - price_low) / num_bins
        poc_bin_idx = min(int((poc_price - price_low) / bin_width), num_bins - 1)
        poc_bin_vol = 0.0
        for c in session_candles:
            typical = (c.high + c.low + c.close) / 3
            idx = min(int((typical - price_low) / bin_width), num_bins - 1)
            if idx == poc_bin_idx:
                poc_bin_vol += c.volume

        if poc_bin_vol / total_vol < min_poc_vol:
            logger.debug(
                "vpoc_weak_poc | symbol={} poc_vol_pct={:.2f} min={}",
                symbol, poc_bin_vol / total_vol, min_poc_vol,
            )
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"POC too weak: {poc_bin_vol / total_vol:.2%} < {min_poc_vol:.0%}",
            )

        # Check proximity to POC
        distance_pct = abs(current_price - poc_price) / poc_price
        if distance_pct > proximity_pct:
            return no_signal

        # Detect bounce direction
        tail = recent_candles[-bounce_candles:]
        all_bullish = all(c.close > c.open for c in tail)
        all_bearish = all(c.close < c.open for c in tail)

        direction = None
        if all_bullish and current_price >= poc_price:
            direction = SignalDirection.LONG
        elif all_bearish and current_price <= poc_price:
            direction = SignalDirection.SHORT
        else:
            return no_signal

        # Swing bias check
        if self.check_swing_bias(symbol, direction):
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"Swing bias blocks {direction.value}",
            )

        # Price levels
        entry_price = current_price
        sl_distance = entry_price * sl_pct

        if direction == SignalDirection.LONG:
            stop_loss = poc_price - sl_distance
            take_profit = entry_price + (entry_price - stop_loss) * rr_ratio
        else:
            stop_loss = poc_price + sl_distance
            take_profit = entry_price - (stop_loss - entry_price) * rr_ratio

        logger.info(
            "vpoc_bounce_triggered | symbol={} dir={} entry={:.2f} sl={:.2f} tp={:.2f} "
            "poc={:.2f} poc_vol_pct={:.2%}",
            symbol, direction.value, entry_price, stop_loss, take_profit,
            poc_price, poc_bin_vol / total_vol,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.60,
            strategy_name=self.name,
            reason=f"VPOC bounce {direction.value}: price near POC {poc_price:.2f}, "
                   f"{'bullish' if direction == SignalDirection.LONG else 'bearish'} "
                   f"rejection candles",
        )
