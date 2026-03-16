"""Market regime detection using EMA crossover and ADX.

Classifies the current market regime for a symbol as one of:
  - ``"trending_up"``   -- EMA20 > EMA50 and ADX > threshold
  - ``"trending_down"`` -- EMA20 < EMA50 and ADX > threshold
  - ``"ranging"``       -- ADX <= threshold (regardless of EMA alignment)

Used by the signal pipeline to filter out mean-reversion shorts during
uptrends and mean-reversion longs during downtrends.
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.indicators import Indicators


class RegimeDetector:
    """EMA + ADX market regime classifier."""

    def __init__(self, adx_threshold: float = 25.0) -> None:
        self._adx_threshold = adx_threshold

    def detect(self, candles: list[OHLCV]) -> str:
        """Classify the current market regime.

        Parameters
        ----------
        candles:
            OHLCV bars (oldest first).  Needs at least 50 bars for
            meaningful EMA values and ~28 bars for ADX warm-up.

        Returns
        -------
        str
            One of ``"trending_up"``, ``"trending_down"``, ``"ranging"``.
        """
        if not candles or len(candles) < 50:
            return "ranging"

        closes = [c.close for c in candles]

        ema20 = Indicators.ema(closes, 20)
        ema50 = Indicators.ema(closes, 50)
        adx = Indicators.adx(candles, 14)

        curr_ema20 = ema20[-1] if ema20 else 0.0
        curr_ema50 = ema50[-1] if ema50 else 0.0
        curr_adx = adx[-1] if adx else 0.0

        # Guard against NaN from insufficient data
        if curr_ema20 != curr_ema20 or curr_ema50 != curr_ema50:
            return "ranging"

        if curr_adx >= self._adx_threshold:
            if curr_ema20 > curr_ema50:
                regime = "trending_up"
            else:
                regime = "trending_down"
        else:
            regime = "ranging"

        logger.debug(
            "regime_detected | ema20={ema20} ema50={ema50} adx={adx} regime={regime}",
            ema20=round(curr_ema20, 4),
            ema50=round(curr_ema50, 4),
            adx=round(curr_adx, 2),
            regime=regime,
        )

        return regime
