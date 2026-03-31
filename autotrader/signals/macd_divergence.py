"""MACD Divergence signal — classic divergence with MACD cross confirmation."""

from __future__ import annotations

import math

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult
from signals.indicators import Indicators


class MACDDivergenceSignal(BaseSignal):
    """Detect price/MACD histogram divergence confirmed by a MACD line cross.

    Bullish: price makes lower low, histogram makes higher low, MACD crosses above signal.
    Bearish: price makes higher high, histogram makes lower high, MACD crosses below signal.
    """

    @property
    def name(self) -> str:
        return "macd_divergence"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used directly. See evaluate_from_macd()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_from_macd(
        self,
        symbol: str,
        current_price: float,
        candles: list[OHLCV],
        market: str,
    ) -> SignalResult:
        """Evaluate MACD divergence with cross confirmation."""
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        fast = self.config.get("fast_period", 12)
        slow = self.config.get("slow_period", 26)
        sig_period = self.config.get("signal_period", 9)
        lookback = self.config.get("divergence_lookback", 30)
        min_dist = self.config.get("min_swing_distance", 5)
        sl_pct = self.config.get("stop_loss_pct", 0.015)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        min_bars = slow + sig_period + lookback
        if len(candles) < min_bars:
            return no_signal

        closes = [c.close for c in candles]
        macd_line, signal_line, histogram = Indicators.macd(closes, fast, slow, sig_period)

        # Work within the lookback window
        window_start = len(candles) - lookback

        # Find swing lows and swing highs in price
        swing_lows = self._find_swing_lows(candles, window_start)
        swing_highs = self._find_swing_highs(candles, window_start)

        direction = None
        reason = ""

        # Check bullish divergence
        if len(swing_lows) >= 2:
            prev_sw, curr_sw = swing_lows[-2], swing_lows[-1]
            if curr_sw[0] - prev_sw[0] >= min_dist:
                price_lower_low = curr_sw[1] < prev_sw[1]
                prev_hist = histogram[prev_sw[0]]
                curr_hist = histogram[curr_sw[0]]
                if not math.isnan(prev_hist) and not math.isnan(curr_hist):
                    hist_higher_low = curr_hist > prev_hist
                    if price_lower_low and hist_higher_low:
                        if self._bullish_cross(macd_line, signal_line):
                            direction = SignalDirection.LONG
                            reason = (
                                f"Bullish divergence: price low {curr_sw[1]:.2f} < "
                                f"{prev_sw[1]:.2f} but histogram rising, MACD cross up"
                            )

        # Check bearish divergence
        if direction is None and len(swing_highs) >= 2:
            prev_sw, curr_sw = swing_highs[-2], swing_highs[-1]
            if curr_sw[0] - prev_sw[0] >= min_dist:
                price_higher_high = curr_sw[1] > prev_sw[1]
                prev_hist = histogram[prev_sw[0]]
                curr_hist = histogram[curr_sw[0]]
                if not math.isnan(prev_hist) and not math.isnan(curr_hist):
                    hist_lower_high = curr_hist < prev_hist
                    if price_higher_high and hist_lower_high:
                        if self._bearish_cross(macd_line, signal_line):
                            direction = SignalDirection.SHORT
                            reason = (
                                f"Bearish divergence: price high {curr_sw[1]:.2f} > "
                                f"{prev_sw[1]:.2f} but histogram falling, MACD cross down"
                            )

        if direction is None:
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
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * rr_ratio
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * rr_ratio

        logger.info(
            "macd_divergence_triggered | symbol={} dir={} entry={:.2f} sl={:.2f} tp={:.2f} "
            "reason={}",
            symbol, direction.value, entry_price, stop_loss, take_profit, reason,
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

    @staticmethod
    def _find_swing_lows(
        candles: list[OHLCV], start: int, margin: int = 2,
    ) -> list[tuple[int, float]]:
        """Find swing lows (local minima) in candle lows."""
        swings = []
        for i in range(max(start, margin), len(candles) - margin):
            low = candles[i].low
            is_pivot = all(
                candles[i - j].low >= low for j in range(1, margin + 1)
            ) and all(
                candles[i + j].low >= low for j in range(1, margin + 1)
            )
            if is_pivot:
                swings.append((i, low))
        return swings

    @staticmethod
    def _find_swing_highs(
        candles: list[OHLCV], start: int, margin: int = 2,
    ) -> list[tuple[int, float]]:
        """Find swing highs (local maxima) in candle highs."""
        swings = []
        for i in range(max(start, margin), len(candles) - margin):
            high = candles[i].high
            is_pivot = all(
                candles[i - j].high <= high for j in range(1, margin + 1)
            ) and all(
                candles[i + j].high <= high for j in range(1, margin + 1)
            )
            if is_pivot:
                swings.append((i, high))
        return swings

    @staticmethod
    def _bullish_cross(macd_line: list[float], signal_line: list[float]) -> bool:
        """Check if MACD crossed above signal line within last 3 bars."""
        for i in range(-3, 0):
            try:
                prev_m, prev_s = macd_line[i - 1], signal_line[i - 1]
                curr_m, curr_s = macd_line[i], signal_line[i]
                if math.isnan(prev_m) or math.isnan(prev_s):
                    continue
                if math.isnan(curr_m) or math.isnan(curr_s):
                    continue
                if prev_m <= prev_s and curr_m > curr_s:
                    return True
            except IndexError:
                continue
        return False

    @staticmethod
    def _bearish_cross(macd_line: list[float], signal_line: list[float]) -> bool:
        """Check if MACD crossed below signal line within last 3 bars."""
        for i in range(-3, 0):
            try:
                prev_m, prev_s = macd_line[i - 1], signal_line[i - 1]
                curr_m, curr_s = macd_line[i], signal_line[i]
                if math.isnan(prev_m) or math.isnan(prev_s):
                    continue
                if math.isnan(curr_m) or math.isnan(curr_s):
                    continue
                if prev_m >= prev_s and curr_m < curr_s:
                    return True
            except IndexError:
                continue
        return False
