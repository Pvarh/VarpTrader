"""On-chain whale transaction detection via Whale Alert API.

Monitors large cryptocurrency wallet movements (exchange deposits,
withdrawals, and whale-to-whale transfers) to detect institutional
sell pressure or accumulation signals.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from loguru import logger

# Known exchange owner labels used by Whale Alert.  When the destination
# of a large transfer is one of these, we infer sell pressure.  When the
# source is one of these, we infer accumulation (withdrawal from exchange).
_EXCHANGE_OWNERS: frozenset[str] = frozenset({
    "binance",
    "coinbase",
    "kraken",
    "bitfinex",
    "huobi",
    "okex",
    "kucoin",
    "bybit",
    "gemini",
    "bitstamp",
    "ftx",
    "gate.io",
    "crypto.com",
    "bittrex",
    "poloniex",
})


class OnChainWhaleDetector:
    """Monitors Whale Alert API for large crypto wallet movements.

    Transactions where the *destination* is a known exchange are treated
    as **sell pressure** (tokens being moved onto an exchange to sell).
    Transactions where the *source* is a known exchange are treated as
    **accumulation** (tokens being withdrawn to a private wallet).

    Parameters
    ----------
    api_key:
        Whale Alert API key.
    crypto_transfer_usd:
        Minimum transaction value (USD) to consider.
    flag_ttl_minutes:
        How many minutes a directional signal stays active.
    """

    _WHALE_ALERT_BASE = "https://api.whale-alert.io/v1"

    def __init__(
        self,
        api_key: str,
        crypto_transfer_usd: float = 1_000_000,
        flag_ttl_minutes: int = 15,
    ) -> None:
        self._api_key: str = api_key
        self._min_usd: float = crypto_transfer_usd
        self._flag_duration: int = flag_ttl_minutes * 60  # seconds
        self._sell_pressure: dict[str, float] = {}   # symbol -> expiry
        self._accumulation: dict[str, float] = {}    # symbol -> expiry
        self._seen_tx: set[str] = set()
        self._seen_tx_last_clear: float = time.time()

        logger.info(
            "onchain_whale_detector_initialised | crypto_transfer_usd={crypto_transfer_usd} flag_ttl_minutes={flag_ttl_minutes}",
            crypto_transfer_usd=crypto_transfer_usd,
            flag_ttl_minutes=flag_ttl_minutes,
        )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def poll(self) -> None:
        """Poll Whale Alert API for recent large transactions.

        Fetches transactions from the last 10 minutes that exceed
        ``crypto_transfer_usd`` and classifies each as sell pressure or
        accumulation based on the exchange ownership of sender /
        receiver.
        """
        self._cleanup_expired()
        self._clear_seen_tx_if_stale()

        lookback_seconds = 600  # 10 minutes
        start_ts = int(time.time()) - lookback_seconds

        url = (
            f"{self._WHALE_ALERT_BASE}/transactions"
            f"?min_value={int(self._min_usd)}"
            f"&api_key={self._api_key}"
            f"&start={start_ts}"
        )

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            if data.get("result") != "success":
                logger.warning(
                    "whale_alert_api_unsuccessful | result={result} message={message}",
                    result=data.get("result"),
                    message=data.get("message"),
                )
                return

            transactions: list[dict[str, Any]] = data.get("transactions", [])
            if not transactions:
                logger.debug("no_whale_transactions")
                return

            expiry = time.time() + self._flag_duration

            for tx in transactions:
                symbol: str = tx.get("symbol", "").upper()
                amount_usd: float = float(tx.get("amount_usd", 0.0))
                from_owner: str = (
                    tx.get("from", {}).get("owner", "unknown").lower()
                )
                to_owner: str = (
                    tx.get("to", {}).get("owner", "unknown").lower()
                )
                tx_hash: str = tx.get("hash", "n/a")

                if not symbol:
                    continue

                if tx_hash in self._seen_tx:
                    continue
                self._seen_tx.add(tx_hash)

                if to_owner in _EXCHANGE_OWNERS:
                    # Moving tokens *to* an exchange -> sell pressure
                    self._sell_pressure[symbol] = expiry
                    logger.warning(
                        "whale_sell_pressure_detected | symbol={symbol} amount_usd={amount_usd} from_owner={from_owner} to_owner={to_owner} tx_hash={tx_hash}",
                        symbol=symbol,
                        amount_usd=round(amount_usd, 2),
                        from_owner=from_owner,
                        to_owner=to_owner,
                        tx_hash=tx_hash,
                    )
                elif from_owner in _EXCHANGE_OWNERS:
                    # Withdrawing tokens *from* an exchange -> accumulation
                    self._accumulation[symbol] = expiry
                    logger.warning(
                        "whale_accumulation_detected | symbol={symbol} amount_usd={amount_usd} from_owner={from_owner} to_owner={to_owner} tx_hash={tx_hash}",
                        symbol=symbol,
                        amount_usd=round(amount_usd, 2),
                        from_owner=from_owner,
                        to_owner=to_owner,
                        tx_hash=tx_hash,
                    )
                else:
                    # Whale-to-whale (unknown wallets) -- informational only
                    logger.info(
                        "whale_transfer_observed | symbol={symbol} amount_usd={amount_usd} from_owner={from_owner} to_owner={to_owner} tx_hash={tx_hash}",
                        symbol=symbol,
                        amount_usd=round(amount_usd, 2),
                        from_owner=from_owner,
                        to_owner=to_owner,
                        tx_hash=tx_hash,
                    )

        except requests.RequestException:
            logger.exception("whale_alert_poll_request_error")
        except Exception:
            logger.exception("whale_alert_poll_error")

    # ------------------------------------------------------------------
    # Signal queries
    # ------------------------------------------------------------------
    def has_sell_pressure(self, symbol: str) -> bool:
        """Check if sell pressure is active for *symbol*.

        Sell pressure is flagged when a large on-chain transfer was
        recently sent **to** a known exchange, implying the whale
        intends to sell.

        Parameters
        ----------
        symbol:
            Cryptocurrency symbol in uppercase (e.g. ``"BTC"``).

        Returns
        -------
        bool
            ``True`` if a sell-pressure flag is active and not expired.
        """
        self._cleanup_expired()
        return symbol.upper() in self._sell_pressure

    def has_accumulation_signal(self, symbol: str) -> bool:
        """Check if an accumulation signal is active for *symbol*.

        Accumulation is flagged when a large on-chain transfer was
        recently sent **from** a known exchange to a private wallet,
        implying the whale is accumulating the asset.

        Parameters
        ----------
        symbol:
            Cryptocurrency symbol in uppercase (e.g. ``"BTC"``).

        Returns
        -------
        bool
            ``True`` if an accumulation flag is active and not expired.
        """
        self._cleanup_expired()
        return symbol.upper() in self._accumulation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cleanup_expired(self) -> None:
        """Remove expired flags from both signal dictionaries."""
        now = time.time()

        expired_sell = [
            s for s, exp in self._sell_pressure.items() if exp <= now
        ]
        for s in expired_sell:
            del self._sell_pressure[s]
            logger.debug("sell_pressure_expired | symbol={symbol}", symbol=s)

        expired_acc = [
            s for s, exp in self._accumulation.items() if exp <= now
        ]
        for s in expired_acc:
            del self._accumulation[s]
            logger.debug("accumulation_expired | symbol={symbol}", symbol=s)

    def _clear_seen_tx_if_stale(self) -> None:
        """Clear the seen tx_hash set every 24 hours to prevent memory growth."""
        now = time.time()
        if now - self._seen_tx_last_clear >= 86_400:
            self._seen_tx.clear()
            self._seen_tx_last_clear = now
            logger.debug("seen_tx_hashes_cleared")
