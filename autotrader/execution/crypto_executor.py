"""Crypto exchange execution wrapper using CCXT.

Provides a unified interface for submitting orders, querying balances,
and managing positions across any CCXT-supported exchange.
"""

from __future__ import annotations

from typing import Any

import ccxt
from loguru import logger


class CryptoExecutor:
    """CCXT unified order placement for crypto exchanges.

    Mirrors the initialization pattern used by :class:`CryptoFeed` and
    exposes authenticated endpoints for order management.

    Parameters
    ----------
    exchange_id:
        CCXT exchange identifier (e.g. ``"binance"``, ``"bybit"``).
    api_key:
        Exchange API key for authenticated operations.
    secret:
        Exchange API secret for authenticated operations.

    Raises
    ------
    ccxt.ExchangeNotAvailable
        If *exchange_id* is not recognised by CCXT.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str | None = None,
        secret: str | None = None,
    ) -> None:
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            logger.error("unknown_exchange | exchange_id={exchange_id}", exchange_id=exchange_id)
            raise ccxt.ExchangeNotAvailable(
                f"Exchange '{exchange_id}' is not available in CCXT"
            )

        config: dict[str, Any] = {"enableRateLimit": True}
        if api_key is not None:
            config["apiKey"] = api_key
        if secret is not None:
            config["secret"] = secret

        self._exchange: ccxt.Exchange = exchange_class(config)
        self._exchange_id: str = exchange_id

        logger.info(
            "crypto_executor_initialised | exchange={exchange} authenticated={authenticated}",
            exchange=exchange_id,
            authenticated=api_key is not None,
        )

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------
    def submit_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> dict[str, Any] | None:
        """Submit a market order.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        side:
            ``"buy"`` or ``"sell"``.
        amount:
            Quantity of the base currency to trade.

        Returns
        -------
        dict | None
            CCXT unified order structure on success, or *None* on
            failure.
        """
        try:
            order: dict[str, Any] = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side.lower(),
                amount=amount,
            )
            logger.info(
                "market_order_submitted | symbol={symbol} side={side} amount={amount} order_type=market order_id={order_id} status={status}",
                symbol=symbol,
                side=side,
                amount=amount,
                order_id=order.get("id"),
                status=order.get("status"),
            )
            return order

        except ccxt.BaseError:
            logger.exception(
                "market_order_ccxt_error | symbol={symbol} side={side} amount={amount} order_type=market",
                symbol=symbol,
                side=side,
                amount=amount,
            )
            return None
        except Exception:
            logger.exception(
                "market_order_error | symbol={symbol} side={side} amount={amount} order_type=market",
                symbol=symbol,
                side=side,
                amount=amount,
            )
            return None

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> dict[str, Any] | None:
        """Submit a limit order.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        side:
            ``"buy"`` or ``"sell"``.
        amount:
            Quantity of the base currency to trade.
        price:
            Limit price for the order.

        Returns
        -------
        dict | None
            CCXT unified order structure on success, or *None* on
            failure.
        """
        try:
            order: dict[str, Any] = self._exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side.lower(),
                amount=amount,
                price=price,
            )
            logger.info(
                "limit_order_submitted | symbol={symbol} side={side} amount={amount} price={price} order_type=limit order_id={order_id} status={status}",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                order_id=order.get("id"),
                status=order.get("status"),
            )
            return order

        except ccxt.BaseError:
            logger.exception(
                "limit_order_ccxt_error | symbol={symbol} side={side} amount={amount} price={price} order_type=limit",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
            )
            return None
        except Exception:
            logger.exception(
                "limit_order_error | symbol={symbol} side={side} amount={amount} price={price} order_type=limit",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
            )
            return None

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------
    def get_balance(self) -> dict[str, Any] | None:
        """Fetch full account balance.

        Returns
        -------
        dict | None
            CCXT unified balance structure containing ``"free"``,
            ``"used"``, and ``"total"`` sub-dicts keyed by currency.
            Returns *None* on failure.
        """
        try:
            balance: dict[str, Any] = self._exchange.fetch_balance()
            logger.info(
                "balance_fetched | currencies={currencies}",
                currencies=len(balance.get("total", {})),
            )
            return balance

        except ccxt.BaseError:
            logger.exception("get_balance_ccxt_error")
            return None
        except Exception:
            logger.exception("get_balance_error")
            return None

    # ------------------------------------------------------------------
    # Positions (spot balances as proxy)
    # ------------------------------------------------------------------
    _STABLECOINS: set[str] = {"USDC", "USDT", "BUSD", "BNB"}

    def get_positions(self) -> list[dict[str, Any]]:
        """Return non-stablecoin spot balances as open positions.

        Since we only do spot trading, any coin balance with
        ``total > 0`` (excluding stablecoins) represents an open
        position.

        Returns
        -------
        list[dict]
            Each dict contains ``currency``, ``free``, ``used``, and
            ``total`` keys.  Returns an empty list on error.
        """
        try:
            balance: dict[str, Any] = self._exchange.fetch_balance()
            totals = balance.get("total", {})
            frees = balance.get("free", {})
            useds = balance.get("used", {})

            positions: list[dict[str, Any]] = []
            for coin, total in totals.items():
                total_f = float(total or 0)
                if total_f > 0 and coin not in self._STABLECOINS:
                    positions.append({
                        "currency": coin,
                        "free": float(frees.get(coin, 0) or 0),
                        "used": float(useds.get(coin, 0) or 0),
                        "total": total_f,
                    })

            logger.info("spot_positions_fetched | count={count}", count=len(positions))
            return positions

        except ccxt.BaseError:
            logger.exception("get_positions_ccxt_error")
            return []
        except Exception:
            logger.exception("get_positions_error")
            return []

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------
    def close_position(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> dict[str, Any] | None:
        """Close a spot position via a market sell order.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        side:
            ``"buy"`` or ``"sell"``.
        amount:
            Quantity to trade.

        Returns
        -------
        dict | None
            Closing order details, or *None* on failure.
        """
        try:
            order: dict[str, Any] = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side.lower(),
                amount=amount,
            )
            logger.info(
                "position_closed | symbol={symbol} side={side} amount={amount} order_id={order_id} status={status}",
                symbol=symbol,
                side=side,
                amount=amount,
                order_id=order.get("id"),
                status=order.get("status"),
            )
            return order

        except ccxt.BaseError:
            logger.exception(
                "close_position_ccxt_error | symbol={symbol} side={side} amount={amount}",
                symbol=symbol,
                side=side,
                amount=amount,
            )
            return None
        except Exception:
            logger.exception(
                "close_position_error | symbol={symbol} side={side} amount={amount}",
                symbol=symbol,
                side=side,
                amount=amount,
            )
            return None
