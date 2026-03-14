"""Lightweight event bus for connecting data streams to signal processors.

Provides a simple publish-subscribe mechanism so that data feed modules
(``StockStream``, ``CryptoStream``) can push events (price updates,
candle closes, etc.) to strategy engines and other consumers without
tight coupling.

Both synchronous and asynchronous subscribers are supported.  Synchronous
callbacks are invoked inline via ``publish()``, while async callbacks are
dispatched via ``publish_async()``.

Usage::

    bus = EventBus()

    # Synchronous subscriber
    bus.subscribe("price_update", lambda sym, price, mkt: print(f"{sym}: {price}"))

    # Async subscriber
    async def on_candle(symbol, bar, market):
        await strategy.evaluate(symbol, bar)

    bus.subscribe_async("candle_close", on_candle)

    # Publishing
    bus.publish("price_update", "AAPL", 150.25, "stock")
    await bus.publish_async("candle_close", "BTC/USDT", bar, "crypto")
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable

from loguru import logger


class EventBus:
    """Simple publish-subscribe event bus for trading events.

    Event types
    -----------
    The following event types are used by convention throughout the
    autotrader system.  Any string may be used as an event type; these
    are the standard ones:

    - ``"price_update"``: ``(symbol: str, price: float, market: str)``
    - ``"candle_close"``: ``(symbol: str, bar: OHLCV, market: str)``
    - ``"orderbook_update"``: ``(symbol: str, orderbook: dict, market: str)``
    - ``"signal_triggered"``: ``(signal_result,)``
    - ``"trade_opened"``: ``(trade: Trade,)``
    - ``"trade_closed"``: ``(trade: Trade,)``

    Thread safety
    -------------
    This implementation is **not** thread-safe.  It is designed to be
    used within a single asyncio event loop.  If cross-thread delivery
    is required, consider wrapping ``publish`` calls with
    ``loop.call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self._async_subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

        logger.info("event_bus_initialised")

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------
    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register a synchronous callback for an event type.

        Parameters
        ----------
        event_type:
            Event name (e.g. ``"price_update"``).
        callback:
            A synchronous callable.  Will receive the positional
            arguments passed to ``publish()``.
        """
        self._subscribers[event_type].append(callback)
        logger.debug(
            "subscriber_registered | event_type={event_type} mode=sync total={total}",
            event_type=event_type,
            total=len(self._subscribers[event_type]),
        )

    def subscribe_async(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register an async callback for an event type.

        Parameters
        ----------
        event_type:
            Event name (e.g. ``"candle_close"``).
        callback:
            An async callable (coroutine function).  Will receive the
            positional arguments passed to ``publish_async()``.
        """
        if not asyncio.iscoroutinefunction(callback):
            logger.warning(
                "non_async_callback_registered_as_async | event_type={event_type} detail={detail}",
                event_type=event_type,
                detail=(
                    "The callback is not a coroutine function. "
                    "It will be wrapped but consider using subscribe() instead."
                ),
            )

        self._async_subscribers[event_type].append(callback)
        logger.debug(
            "subscriber_registered | event_type={event_type} mode=async total={total}",
            event_type=event_type,
            total=len(self._async_subscribers[event_type]),
        )

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Remove a callback from an event type.

        Searches both synchronous and asynchronous subscriber lists.
        If the callback is not found, the call is silently ignored.

        Parameters
        ----------
        event_type:
            Event name the callback was registered under.
        callback:
            The exact callable that was previously registered.
        """
        removed = False

        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                removed = True
            except ValueError:
                pass

        if event_type in self._async_subscribers:
            try:
                self._async_subscribers[event_type].remove(callback)
                removed = True
            except ValueError:
                pass

        if removed:
            logger.debug("subscriber_removed | event_type={event_type}", event_type=event_type)
        else:
            logger.debug("subscriber_not_found | event_type={event_type}", event_type=event_type)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish(self, event_type: str, *args: Any) -> None:
        """Publish event to all synchronous subscribers.

        Each registered callback is invoked with ``*args``.  If a
        callback raises an exception it is logged and the remaining
        subscribers still receive the event.

        Parameters
        ----------
        event_type:
            Event name (e.g. ``"price_update"``).
        *args:
            Positional arguments forwarded to each callback.
        """
        callbacks = self._subscribers.get(event_type, [])
        if not callbacks:
            return

        for callback in callbacks:
            try:
                callback(*args)
            except Exception:
                logger.exception(
                    "sync_subscriber_error | event_type={event_type} callback={callback_name}",
                    event_type=event_type,
                    callback_name=getattr(callback, "__name__", repr(callback)),
                )

    async def publish_async(self, event_type: str, *args: Any) -> None:
        """Publish event to all async subscribers.

        Async callbacks are awaited sequentially.  Synchronous callbacks
        registered under the same event type are also invoked (before
        the async ones) so that a single ``publish_async`` call reaches
        every subscriber.

        If a callback raises an exception it is logged and the remaining
        subscribers still receive the event.

        Parameters
        ----------
        event_type:
            Event name (e.g. ``"candle_close"``).
        *args:
            Positional arguments forwarded to each callback.
        """
        # Deliver to sync subscribers first
        sync_callbacks = self._subscribers.get(event_type, [])
        for callback in sync_callbacks:
            try:
                callback(*args)
            except Exception:
                logger.exception(
                    "sync_subscriber_error_in_async_publish | event_type={event_type} callback={callback_name}",
                    event_type=event_type,
                    callback_name=getattr(callback, "__name__", repr(callback)),
                )

        # Deliver to async subscribers
        async_callbacks = self._async_subscribers.get(event_type, [])
        for callback in async_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    # Wrapped sync callback that was registered via subscribe_async
                    callback(*args)
            except Exception:
                logger.exception(
                    "async_subscriber_error | event_type={event_type} callback={callback_name}",
                    event_type=event_type,
                    callback_name=getattr(callback, "__name__", repr(callback)),
                )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def subscriber_count(self, event_type: str) -> int:
        """Return the total number of subscribers for an event type.

        Parameters
        ----------
        event_type:
            Event name to query.

        Returns
        -------
        int
            Combined count of synchronous and asynchronous subscribers.
        """
        sync_count = len(self._subscribers.get(event_type, []))
        async_count = len(self._async_subscribers.get(event_type, []))
        return sync_count + async_count

    def clear(self, event_type: str | None = None) -> None:
        """Remove all subscribers.

        Parameters
        ----------
        event_type:
            If provided, only subscribers for this event type are
            removed.  When *None*, **all** subscribers across all event
            types are cleared.
        """
        if event_type is not None:
            self._subscribers.pop(event_type, None)
            self._async_subscribers.pop(event_type, None)
            logger.info("subscribers_cleared | event_type={event_type}", event_type=event_type)
        else:
            self._subscribers.clear()
            self._async_subscribers.clear()
            logger.info("all_subscribers_cleared")
