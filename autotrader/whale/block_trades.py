"""Block trade detection via Polygon.io trades API.

Monitors recent trades for each symbol and flags when unusually large
block trades occur, which often indicate institutional activity.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from loguru import logger


class BlockTradeDetector:
    """Detects large block trades in equities via Polygon.io API.

    A *block trade* is any single trade whose share count exceeds
    ``stock_block_shares`` **or** whose notional value (price x size) exceeds
    ``stock_block_usd``.  When detected, a directional flag is set for the
    symbol that expires after ``flag_ttl_minutes`` minutes.

    Parameters
    ----------
    api_key:
        Polygon.io API key.
    stock_block_shares:
        Minimum number of shares for a trade to qualify as a block.
    stock_block_usd:
        Minimum notional value (USD) for a trade to qualify as a block.
    flag_ttl_minutes:
        How many minutes a whale flag stays active after detection.
    """

    _POLYGON_BASE = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str,
        stock_block_shares: int = 50_000,
        stock_block_usd: float = 2_000_000,
        flag_ttl_minutes: int = 15,
    ) -> None:
        self._api_key: str = api_key
        self._min_shares: int = stock_block_shares
        self._min_value: float = stock_block_usd
        self._flag_duration: int = flag_ttl_minutes * 60  # seconds
        self._buy_flags: dict[str, float] = {}   # symbol -> expiry timestamp
        self._sell_flags: dict[str, float] = {}  # symbol -> expiry timestamp
        self._api_disabled: bool = False  # set True on 401/403 to stop spamming

        logger.info(
            "block_trade_detector_initialised | stock_block_shares={stock_block_shares} stock_block_usd={stock_block_usd} flag_ttl_minutes={flag_ttl_minutes}",
            stock_block_shares=stock_block_shares,
            stock_block_usd=stock_block_usd,
            flag_ttl_minutes=flag_ttl_minutes,
        )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def poll(self, symbols: list[str]) -> None:
        """Poll Polygon.io trades for each symbol and check for blocks.

        Fetches the most recent 50 trades per symbol, evaluates each
        against the block-trade thresholds, and sets time-limited
        directional flags accordingly.

        Parameters
        ----------
        symbols:
            List of ticker symbols to scan (e.g. ``["AAPL", "TSLA"]``).
        """
        self._cleanup_expired()

        if self._api_disabled:
            return

        if not self._api_key:
            return

        for symbol in symbols:
            url = (
                f"{self._POLYGON_BASE}/v3/trades/{symbol}"
                f"?limit=50&order=desc&apiKey={self._api_key}"
            )
            try:
                response = requests.get(url, timeout=10)
                if response.status_code in (401, 403):
                    logger.warning(
                        "block_trade_api_auth_failed | status={status} -- disabling block trade polling (Polygon free tier does not support /v3/trades). Whale stock monitoring will be inactive.",
                        status=response.status_code,
                    )
                    self._api_disabled = True
                    return
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                results: list[dict[str, Any]] = data.get("results", [])

                if not results:
                    logger.debug("no_recent_trades | symbol={symbol}", symbol=symbol)
                    continue

                for trade in results:
                    size: int = int(trade.get("size", 0))
                    price: float = float(trade.get("price", 0.0))
                    notional: float = price * size
                    conditions: list[int] = trade.get("conditions", [])

                    if size < self._min_shares and notional < self._min_value:
                        continue

                    # Determine direction from Polygon trade conditions.
                    # Condition codes (SIP):
                    #   - Codes indicating the trade hit the ask (buyer
                    #     initiated): we treat as a buy block.
                    #   - Codes indicating the trade hit the bid (seller
                    #     initiated): we treat as a sell block.
                    # Without granular condition data, we conservatively
                    # flag *both* directions so downstream consumers can
                    # decide.
                    expiry = time.time() + self._flag_duration

                    # Heuristic: odd-lot conditions 15, 16 often indicate
                    # sell-side; above-ask conditions 37, 38 indicate
                    # buy-side.  If no conditions are present, flag both.
                    flagged_buy = False
                    flagged_sell = False

                    if any(c in (37, 38, 14) for c in conditions):
                        self._buy_flags[symbol] = expiry
                        flagged_buy = True
                    elif any(c in (15, 16) for c in conditions):
                        self._sell_flags[symbol] = expiry
                        flagged_sell = True
                    else:
                        # No distinguishing condition -- flag both sides
                        self._buy_flags[symbol] = expiry
                        self._sell_flags[symbol] = expiry
                        flagged_buy = True
                        flagged_sell = True

                    logger.warning(
                        "block_trade_detected | symbol={symbol} size={size} price={price} notional={notional} conditions={conditions} flagged_buy={flagged_buy} flagged_sell={flagged_sell}",
                        symbol=symbol,
                        size=size,
                        price=price,
                        notional=round(notional, 2),
                        conditions=conditions,
                        flagged_buy=flagged_buy,
                        flagged_sell=flagged_sell,
                    )

            except requests.RequestException:
                logger.exception("block_trade_poll_request_error | symbol={symbol}", symbol=symbol)
            except Exception:
                logger.exception("block_trade_poll_error | symbol={symbol}", symbol=symbol)

    # ------------------------------------------------------------------
    # Flag queries
    # ------------------------------------------------------------------
    def has_sell_flag(self, symbol: str) -> bool:
        """Check if a sell whale flag is active for *symbol*.

        An active sell flag indicates that a large sell-side block trade
        was recently observed, suggesting institutional distribution.
        Downstream strategies may choose to suppress new long entries
        while this flag is set.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).

        Returns
        -------
        bool
            ``True`` if a sell flag is active and has not expired.
        """
        self._cleanup_expired()
        return symbol in self._sell_flags

    def has_buy_flag(self, symbol: str) -> bool:
        """Check if a buy whale flag is active for *symbol*.

        An active buy flag indicates that a large buy-side block trade
        was recently observed, suggesting institutional accumulation.
        Downstream strategies may use this as confirmation for long
        entries.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).

        Returns
        -------
        bool
            ``True`` if a buy flag is active and has not expired.
        """
        self._cleanup_expired()
        return symbol in self._buy_flags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cleanup_expired(self) -> None:
        """Remove expired flags from both buy and sell dictionaries."""
        now = time.time()

        expired_buy = [s for s, exp in self._buy_flags.items() if exp <= now]
        for s in expired_buy:
            del self._buy_flags[s]
            logger.debug("buy_flag_expired | symbol={symbol}", symbol=s)

        expired_sell = [s for s, exp in self._sell_flags.items() if exp <= now]
        for s in expired_sell:
            del self._sell_flags[s]
            logger.debug("sell_flag_expired | symbol={symbol}", symbol=s)
