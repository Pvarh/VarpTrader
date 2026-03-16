"""Daily directional bias based on higher-timeframe EMA alignment.

At the start of each trading day the module evaluates 4-hour candles to
determine whether the session should be biased ``"long"``, ``"short"``,
or ``"neutral"``.

When a bias is active the strategy pipeline suppresses signals in the
opposite direction (e.g.  bias = ``"long"`` blocks all short signals).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from loguru import logger

from journal.models import OHLCV
from signals.indicators import Indicators


class SessionBias:
    """Daily directional bias from 4h EMA alignment."""

    def __init__(self) -> None:
        self._bias: str = "neutral"
        self._bias_date: str = ""  # ISO date when bias was last set

    @property
    def bias(self) -> str:
        """Current session bias: ``'long'``, ``'short'``, or ``'neutral'``."""
        return self._bias

    def evaluate(self, candles_4h: list[OHLCV]) -> str:
        """Compute and cache the daily bias.

        Parameters
        ----------
        candles_4h:
            4-hour OHLCV bars (oldest first).  Needs at least 50 bars.

        Returns
        -------
        str
            One of ``"long"``, ``"short"``, ``"neutral"``.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Only re-evaluate once per day
        if self._bias_date == today:
            return self._bias

        if not candles_4h or len(candles_4h) < 50:
            self._bias = "neutral"
            self._bias_date = today
            return self._bias

        closes = [c.close for c in candles_4h]

        ema20 = Indicators.ema(closes, 20)
        ema50 = Indicators.ema(closes, 50)

        curr_ema20 = ema20[-1] if ema20 else 0.0
        curr_ema50 = ema50[-1] if ema50 else 0.0

        # NaN guard
        if curr_ema20 != curr_ema20 or curr_ema50 != curr_ema50:
            self._bias = "neutral"
        elif curr_ema20 > curr_ema50:
            self._bias = "long"
        elif curr_ema20 < curr_ema50:
            self._bias = "short"
        else:
            self._bias = "neutral"

        self._bias_date = today

        logger.info(
            "session_bias_set | bias={bias} ema20={ema20} ema50={ema50} date={date}",
            bias=self._bias,
            ema20=round(curr_ema20, 4),
            ema50=round(curr_ema50, 4),
            date=today,
        )

        return self._bias

    def should_block(self, direction: str) -> bool:
        """Return True if the current bias blocks *direction*.

        - bias ``"long"``  blocks ``"short"`` signals
        - bias ``"short"`` blocks ``"long"``  signals
        - bias ``"neutral"`` blocks nothing
        """
        if self._bias == "long" and direction == "short":
            return True
        if self._bias == "short" and direction == "long":
            return True
        return False
