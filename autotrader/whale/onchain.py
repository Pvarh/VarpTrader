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
        min_sell_events: int = 3,
        max_block_minutes: int = 30,
        monitored_symbols: set[str] | None = None,
    ) -> None:
        self._api_key: str = api_key
        self._min_usd: float = crypto_transfer_usd
        self._flag_duration: int = flag_ttl_minutes * 60  # seconds
        self._min_sell_events: int = max(1, int(min_sell_events))
        self._max_block_duration: int = max_block_minutes * 60
        self._monitored_symbols: set[str] | None = (
            {symbol.upper() for symbol in monitored_symbols}
            if monitored_symbols
            else None
        )
        self._sell_pressure_events: dict[str, list[float]] = {}
        self._sell_pressure_active_since: dict[str, float] = {}
        self._accumulation_events: dict[str, list[float]] = {}
        self._accumulation: dict[str, float] = {}    # symbol -> expiry
        self._seen_tx: set[str] = set()
        self._seen_tx_last_clear: float = time.time()

        logger.info(
            "onchain_whale_detector_initialised | crypto_transfer_usd={crypto_transfer_usd} flag_ttl_minutes={flag_ttl_minutes} min_sell_events={min_sell_events} max_block_minutes={max_block_minutes} monitored_symbols={monitored_symbols}",
            crypto_transfer_usd=crypto_transfer_usd,
            flag_ttl_minutes=flag_ttl_minutes,
            min_sell_events=self._min_sell_events,
            max_block_minutes=max_block_minutes,
            monitored_symbols=sorted(self._monitored_symbols) if self._monitored_symbols else "all",
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

            now_ts = time.time()
            expiry = now_ts + self._flag_duration

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

                if self._monitored_symbols and symbol not in self._monitored_symbols:
                    continue

                if tx_hash in self._seen_tx:
                    continue
                self._seen_tx.add(tx_hash)

                from_exchange = from_owner in _EXCHANGE_OWNERS
                to_exchange = to_owner in _EXCHANGE_OWNERS

                if from_exchange and to_exchange:
                    logger.debug(
                        "whale_exchange_internal_transfer_ignored | symbol={symbol} amount_usd={amount_usd} from_owner={from_owner} to_owner={to_owner} tx_hash={tx_hash}",
                        symbol=symbol,
                        amount_usd=round(amount_usd, 2),
                        from_owner=from_owner,
                        to_owner=to_owner,
                        tx_hash=tx_hash,
                    )
                    continue

                if to_exchange:
                    # Moving tokens *to* an exchange -> possible sell pressure.
                    events = self._sell_pressure_events.setdefault(symbol, [])
                    events.append(now_ts)
                    self._sell_pressure_events[symbol] = [
                        event_ts
                        for event_ts in events
                        if now_ts - event_ts <= self._flag_duration
                    ]
                    if len(self._sell_pressure_events[symbol]) >= self._min_sell_events:
                        self._sell_pressure_active_since.setdefault(symbol, now_ts)
                    logger.warning(
                        "whale_sell_pressure_detected | symbol={symbol} amount_usd={amount_usd} from_owner={from_owner} to_owner={to_owner} tx_hash={tx_hash}",
                        symbol=symbol,
                        amount_usd=round(amount_usd, 2),
                        from_owner=from_owner,
                        to_owner=to_owner,
                        tx_hash=tx_hash,
                    )
                elif from_exchange:
                    # Withdrawing tokens *from* an exchange -> accumulation
                    self._accumulation[symbol] = expiry
                    acc_events = self._accumulation_events.setdefault(symbol, [])
                    acc_events.append(now_ts)
                    self._accumulation_events[symbol] = [
                        event_ts
                        for event_ts in acc_events
                        if now_ts - event_ts <= self._flag_duration
                    ]
                    self._sell_pressure_events.pop(symbol, None)
                    self._sell_pressure_active_since.pop(symbol, None)
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
        key = symbol.upper()
        events = self._sell_pressure_events.get(key, [])
        if len(events) < self._min_sell_events:
            return False

        accumulation_events = self._accumulation_events.get(key, [])
        if len(events) <= len(accumulation_events):
            logger.debug(
                "sell_pressure_neutralized_by_accumulation | symbol={symbol} sell_events={sell_events} accumulation_events={accumulation_events}",
                symbol=key,
                sell_events=len(events),
                accumulation_events=len(accumulation_events),
            )
            return False

        active_since = self._sell_pressure_active_since.get(key)
        if active_since is None:
            self._sell_pressure_active_since[key] = events[self._min_sell_events - 1]
            active_since = self._sell_pressure_active_since[key]

        if time.time() - active_since > self._max_block_duration:
            self._sell_pressure_events.pop(key, None)
            self._sell_pressure_active_since.pop(key, None)
            logger.info(
                "sell_pressure_block_expired | symbol={symbol} max_block_minutes={minutes}",
                symbol=key,
                minutes=self._max_block_duration / 60,
            )
            return False
        return True

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
        expired_sell: list[str] = []
        for symbol, events in self._sell_pressure_events.items():
            fresh = [
                event_ts
                for event_ts in events
                if now - event_ts <= self._flag_duration
            ]
            if fresh:
                self._sell_pressure_events[symbol] = fresh
            else:
                expired_sell.append(symbol)
        for symbol in expired_sell:
            self._sell_pressure_events.pop(symbol, None)
            self._sell_pressure_active_since.pop(symbol, None)
            logger.debug("sell_pressure_expired | symbol={symbol}", symbol=symbol)

        expired_acc_events: list[str] = []
        for symbol, events in self._accumulation_events.items():
            fresh = [
                event_ts
                for event_ts in events
                if now - event_ts <= self._flag_duration
            ]
            if fresh:
                self._accumulation_events[symbol] = fresh
            else:
                expired_acc_events.append(symbol)
        for symbol in expired_acc_events:
            self._accumulation_events.pop(symbol, None)

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
