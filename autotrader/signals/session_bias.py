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
    """Daily directional bias from 4h EMA alignment.

    Re-evaluates every 2 hours or on explicit reset.
    """

    def __init__(self) -> None:
        self._bias: str = "neutral"
        self._last_eval_time: float = 0.0  # Unix timestamp of last evaluation

    @property
    def bias(self) -> str:
        """Current session bias: ``'long'``, ``'short'``, or ``'neutral'``."""
        return self._bias

    def force_reevaluate(self) -> None:
        """Force a fresh evaluation on the next evaluate() call."""
        self._last_eval_time = 0.0

    def evaluate(self, candles_4h: list[OHLCV]) -> str:
        """Compute and cache the session bias.

        Re-evaluates every 2 hours or if force_reevaluate() was called.

        Parameters
        ----------
        candles_4h:
            4-hour OHLCV bars (oldest first).  Needs at least 50 bars.

        Returns
        -------
        str
            One of ``"long"``, ``"short"``, ``"neutral"``.
        """
        now = time.time()

        # Re-evaluate every 2 hours
        if now - self._last_eval_time < 7200:  # 7200 seconds = 2 hours
            return self._bias

        if not candles_4h or len(candles_4h) < 50:
            self._bias = "neutral"
            self._last_eval_time = now
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

        self._last_eval_time = now

        logger.info(
            "session_bias_set | bias={bias} ema20={ema20} ema50={ema50}",
            bias=self._bias,
            ema20=round(curr_ema20, 4),
            ema50=round(curr_ema50, 4),
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
