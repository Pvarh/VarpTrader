"""Alpaca broker execution wrapper.

Provides a simplified interface around the ``alpaca-py`` SDK for
submitting stock market orders, querying account equity, and managing
open positions.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class AlpacaExecutor:
    """Alpaca REST API wrapper for stock order execution.

    Uses the official ``alpaca-py`` SDK.  Both paper and live trading
    are supported -- the ``paper`` flag controls which base URL is used.

    Parameters
    ----------
    api_key:
        Alpaca API key ID.
    secret_key:
        Alpaca API secret key.
    base_url:
        Base URL for the Alpaca API (e.g.
        ``"https://paper-api.alpaca.markets"``).
    paper:
        ``True`` for paper trading (default), ``False`` for live.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str,
        paper: bool = True,
    ) -> None:
        try:
            from alpaca.trading.client import TradingClient

            self._client: Any = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
                url_override=base_url,
            )
        except Exception:
            logger.exception("alpaca_client_init_error")
            self._client = None

        self._paper: bool = paper

        logger.info(
            "alpaca_executor_initialised | paper={paper} base_url={base_url}",
            paper=paper,
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------
    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
    ) -> dict[str, Any] | None:
        """Submit a market order.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).
        qty:
            Number of shares to trade.
        side:
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        dict | None
            Order details dictionary on success, or *None* on failure.
        """
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(request)
            result: dict[str, Any] = {
                "id": str(order.id),
                "symbol": order.symbol,
                "qty": str(order.qty),
                "side": str(order.side),
                "type": str(order.type),
                "status": str(order.status),
                "submitted_at": str(order.submitted_at),
            }
            logger.info(
                "market_order_submitted | symbol={symbol} qty={qty} side={side} order_type=market order_id={order_id}",
                symbol=symbol,
                qty=qty,
                side=side,
                order_id=result["id"],
            )
            return result

        except Exception:
            logger.exception(
                "market_order_error | symbol={symbol} qty={qty} side={side} order_type=market",
                symbol=symbol,
                qty=qty,
                side=side,
            )
            return None

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
    ) -> dict[str, Any] | None:
        """Submit a limit order.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).
        qty:
            Number of shares to trade.
        side:
            ``"buy"`` or ``"sell"``.
        limit_price:
            Maximum (buy) or minimum (sell) acceptable price.

        Returns
        -------
        dict | None
            Order details dictionary on success, or *None* on failure.
        """
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            order = self._client.submit_order(request)
            result: dict[str, Any] = {
                "id": str(order.id),
                "symbol": order.symbol,
                "qty": str(order.qty),
                "side": str(order.side),
                "type": str(order.type),
                "status": str(order.status),
                "limit_price": str(order.limit_price),
                "submitted_at": str(order.submitted_at),
            }
            logger.info(
                "limit_order_submitted | symbol={symbol} qty={qty} side={side} limit_price={limit_price} order_type=limit order_id={order_id}",
                symbol=symbol,
                qty=qty,
                side=side,
                limit_price=limit_price,
                order_id=result["id"],
            )
            return result

        except Exception:
            logger.exception(
                "limit_order_error | symbol={symbol} qty={qty} side={side} limit_price={limit_price} order_type=limit",
                symbol=symbol,
                qty=qty,
                side=side,
                limit_price=limit_price,
            )
            return None

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------
    def get_account(self) -> dict[str, Any] | None:
        """Get account equity and buying power.

        Returns
        -------
        dict | None
            Dictionary with ``equity``, ``buying_power``, ``cash``,
            and ``portfolio_value`` as floats, or *None* on failure.
        """
        try:
            account = self._client.get_account()
            result: dict[str, Any] = {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "currency": str(account.currency),
            }
            logger.info(
                "account_fetched | equity={equity} buying_power={buying_power}",
                equity=result["equity"],
                buying_power=result["buying_power"],
            )
            return result

        except Exception:
            logger.exception("get_account_error")
            return None

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions.

        Returns
        -------
        list[dict]
            A list of position dictionaries, each containing
            ``symbol``, ``qty``, ``side``, ``market_value``,
            ``avg_entry_price``, ``unrealized_pl``, and
            ``unrealized_plpc``.  Returns an empty list on error.
        """
        try:
            positions = self._client.get_all_positions()
            result: list[dict[str, Any]] = []
            for pos in positions:
                result.append({
                    "symbol": str(pos.symbol),
                    "qty": float(pos.qty),
                    "side": str(pos.side),
                    "market_value": float(pos.market_value),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                })
            logger.info("positions_fetched | count={count}", count=len(result))
            return result

        except Exception:
            logger.exception("get_positions_error")
            return []

    def close_position(self, symbol: str) -> dict[str, Any] | None:
        """Close a specific position.

        Parameters
        ----------
        symbol:
            Ticker symbol whose position should be fully closed.

        Returns
        -------
        dict | None
            Closing order details, or *None* on failure.
        """
        try:
            order = self._client.close_position(symbol)
            result: dict[str, Any] = {
                "id": str(order.id),
                "symbol": str(order.symbol),
                "status": str(order.status),
            }
            logger.info(
                "position_closed | symbol={symbol} order_id={order_id}",
                symbol=symbol,
                order_id=result["id"],
            )
            return result

        except Exception:
            logger.exception("close_position_error | symbol={symbol}", symbol=symbol)
            return None
