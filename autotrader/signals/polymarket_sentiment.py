"""Polymarket sentiment layer for BTC/ETH.

Queries the free Polymarket gamma API for active "up or down" prediction
markets on BTC or ETH.  When the "Up" probability exceeds a threshold
(default 65 %), short signals for that asset are blocked.

Falls back to neutral (0.5) on any network or parsing error.
"""

from __future__ import annotations

import time
from typing import Optional

import requests
from loguru import logger

_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
_CACHE_TTL = 300  # 5 minutes


class PolymarketSentiment:
    """Fetch directional sentiment from Polymarket prediction markets."""

    def __init__(self, block_short_threshold: float = 0.65) -> None:
        self._threshold = block_short_threshold
        self._cache: dict[str, tuple[float, float]] = {}  # asset -> (prob, ts)

    def _fetch_up_probability(self, asset: str) -> float:
        """Query Polymarket for the 'Up' probability of *asset*.

        Parameters
        ----------
        asset:
            ``"BTC"`` or ``"ETH"``.

        Returns
        -------
        float
            Probability between 0.0 and 1.0, or 0.5 on failure.
        """
        try:
            resp = requests.get(
                _GAMMA_URL,
                params={"closed": "false", "limit": 50},
                timeout=10,
            )
            resp.raise_for_status()
            markets = resp.json()

            for market in markets:
                question = (market.get("question") or "").lower()
                # Look for markets like "Will BTC go up or down in the next 4h?"
                if asset.lower() not in question:
                    continue
                if "up" not in question and "down" not in question:
                    continue

                # The outcomes list has probabilities
                outcomes = market.get("outcomes", [])
                outcome_prices = market.get("outcomePrices", [])

                if not outcomes or not outcome_prices:
                    continue

                for i, outcome in enumerate(outcomes):
                    if outcome.lower() == "up" and i < len(outcome_prices):
                        try:
                            prob = float(outcome_prices[i])
                            logger.info(
                                "polymarket_probability | asset={asset} "
                                "question={question} up_prob={prob}",
                                asset=asset,
                                question=market.get("question", ""),
                                prob=round(prob, 3),
                            )
                            return prob
                        except (ValueError, TypeError):
                            continue

            logger.debug("polymarket_no_market_found | asset={}", asset)
            return 0.5

        except Exception:
            logger.debug("polymarket_fetch_failed | asset={}", asset)
            return 0.5

    def get_up_probability(self, asset: str) -> float:
        """Get cached 'Up' probability for *asset*.

        Results are cached for ``_CACHE_TTL`` seconds.
        """
        asset = asset.upper()
        cached = self._cache.get(asset)
        if cached and time.time() - cached[1] < _CACHE_TTL:
            return cached[0]

        prob = self._fetch_up_probability(asset)
        self._cache[asset] = (prob, time.time())
        return prob

    def should_block_short(self, asset: str) -> bool:
        """Return True if short signals should be blocked for *asset*.

        Shorts are blocked when the Polymarket 'Up' probability
        exceeds the configured threshold (default 65%).

        Only applies to BTC and ETH.  Returns False for other assets.
        """
        asset = asset.upper()
        if asset not in ("BTC", "ETH"):
            return False

        prob = self.get_up_probability(asset)
        if prob >= self._threshold:
            logger.info(
                "polymarket_block_short | asset={asset} up_prob={prob} "
                "threshold={threshold}",
                asset=asset,
                prob=round(prob, 3),
                threshold=self._threshold,
            )
            return True
        return False
