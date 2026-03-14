"""Trade lifecycle state machine ensuring proper state transitions.

Manages a trade from initial signal detection through validation,
execution, monitoring, and eventual close.  Every state change is
validated against an explicit transition map and recorded for a full
audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable

from loguru import logger


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class TradeState(Enum):
    """Valid states in the trade lifecycle."""

    SIGNAL = "signal"           # Signal detected, not yet validated
    VALIDATED = "validated"     # Passed risk checks
    SUBMITTED = "submitted"    # Order sent to exchange
    FILLED = "filled"          # Order filled, position open
    MONITORING = "monitoring"  # Actively monitoring for exit
    CLOSING = "closing"        # Exit order submitted
    CLOSED = "closed"          # Exited, PnL realized
    REJECTED = "rejected"      # Failed validation
    CANCELLED = "cancelled"    # Cancelled before fill
    FAILED = "failed"          # Execution error


# ---------------------------------------------------------------------------
# Transition map
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict[TradeState, set[TradeState]] = {
    TradeState.SIGNAL:     {TradeState.VALIDATED, TradeState.REJECTED},
    TradeState.VALIDATED:  {TradeState.SUBMITTED, TradeState.REJECTED},
    TradeState.SUBMITTED:  {TradeState.FILLED, TradeState.CANCELLED, TradeState.FAILED},
    TradeState.FILLED:     {TradeState.MONITORING},
    TradeState.MONITORING: {TradeState.CLOSING, TradeState.CLOSED},
    TradeState.CLOSING:    {TradeState.CLOSED, TradeState.FAILED},
    TradeState.REJECTED:   set(),   # terminal
    TradeState.CANCELLED:  set(),   # terminal
    TradeState.CLOSED:     set(),   # terminal
    TradeState.FAILED:     {TradeState.SUBMITTED},  # can retry
}

# States that have no outgoing transitions (or only retry).
_TERMINAL_STATES: frozenset[TradeState] = frozenset({
    TradeState.REJECTED,
    TradeState.CANCELLED,
    TradeState.CLOSED,
})

# States where the trade requires active attention.
_ACTIVE_STATES: frozenset[TradeState] = frozenset({
    TradeState.SUBMITTED,
    TradeState.FILLED,
    TradeState.MONITORING,
    TradeState.CLOSING,
})


# ---------------------------------------------------------------------------
# Trade context
# ---------------------------------------------------------------------------

@dataclass
class TradeContext:
    """All data associated with a trade through its lifecycle.

    Attributes are updated as the trade progresses through states.
    ``state_history`` records every transition with a UTC timestamp.
    """

    trade_id: Optional[int] = None
    symbol: str = ""
    market: str = ""
    strategy: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    state: TradeState = TradeState.SIGNAL
    state_history: list[tuple[str, str]] = field(default_factory=list)
    rejection_reason: str = ""
    order_id: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    outcome: str = ""
    whale_flag: int = 0
    error_message: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def elapsed_seconds(self) -> float:
        """Seconds elapsed since the trade was created.

        Returns:
            Number of seconds between ``created_at`` and the current
            UTC time.
        """
        created = datetime.fromisoformat(self.created_at)
        now = datetime.now(timezone.utc)
        return (now - created).total_seconds()


# ---------------------------------------------------------------------------
# Single-trade lifecycle
# ---------------------------------------------------------------------------

class TradeLifecycle:
    """Manages state transitions for a single trade.

    Enforces that trades follow valid state transitions and logs all
    state changes for an audit trail.

    Parameters
    ----------
    context:
        The :class:`TradeContext` representing this trade.
    """

    def __init__(self, context: TradeContext) -> None:
        self._context = context
        self._on_state_change: list[
            Callable[[TradeState, TradeState, TradeContext], None]
        ] = []
        self._record_state(context.state)

    # -- properties ---------------------------------------------------------

    @property
    def context(self) -> TradeContext:
        """Get the current trade context."""
        return self._context

    @property
    def state(self) -> TradeState:
        """Get the current trade state."""
        return self._context.state

    @property
    def is_terminal(self) -> bool:
        """Check if the trade is in a terminal state (no more transitions)."""
        return self._context.state in _TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        """Check if the trade requires active monitoring."""
        return self._context.state in _ACTIVE_STATES

    # -- callbacks ----------------------------------------------------------

    def on_state_change(
        self,
        callback: Callable[[TradeState, TradeState, TradeContext], None],
    ) -> None:
        """Register a callback for state transitions.

        The callback receives ``(old_state, new_state, context)`` each
        time a valid transition occurs.

        Parameters
        ----------
        callback:
            A callable with the signature
            ``(TradeState, TradeState, TradeContext) -> None``.
        """
        self._on_state_change.append(callback)

    # -- transitions --------------------------------------------------------

    def transition_to(self, new_state: TradeState, **kwargs: object) -> bool:
        """Attempt to transition to a new state.

        Any additional keyword arguments are applied directly to the
        :class:`TradeContext` (e.g. ``order_id``, ``exit_price``,
        ``rejection_reason``, ``error_message``).

        Parameters
        ----------
        new_state:
            The target :class:`TradeState`.
        **kwargs:
            Key/value pairs to set on the trade context before
            recording the transition.

        Returns
        -------
        bool
            ``True`` if the transition succeeded, ``False`` if it was
            invalid.  Invalid transitions are logged but never raise.
        """
        old_state = self._context.state

        if not self._validate_transition(old_state, new_state):
            logger.warning(
                "invalid_state_transition | trade_id={trade_id} symbol={symbol} from_state={from_state} to_state={to_state}",
                trade_id=self._context.trade_id,
                symbol=self._context.symbol,
                from_state=old_state.value,
                to_state=new_state.value,
            )
            return False

        # Apply extra context fields.
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)
            else:
                logger.warning(
                    "unknown_context_field | field={field} trade_id={trade_id}",
                    field=key,
                    trade_id=self._context.trade_id,
                )

        self._context.state = new_state
        self._record_state(new_state)

        logger.info(
            "trade_state_transition | trade_id={trade_id} symbol={symbol} from_state={from_state} to_state={to_state}",
            trade_id=self._context.trade_id,
            symbol=self._context.symbol,
            from_state=old_state.value,
            to_state=new_state.value,
        )

        # Fire callbacks.
        for cb in self._on_state_change:
            try:
                cb(old_state, new_state, self._context)
            except Exception:
                logger.exception(
                    "state_change_callback_error | trade_id={trade_id} from_state={from_state} to_state={to_state}",
                    trade_id=self._context.trade_id,
                    from_state=old_state.value,
                    to_state=new_state.value,
                )

        return True

    # -- internals ----------------------------------------------------------

    def _record_state(self, state: TradeState) -> None:
        """Record a state change in the context's history list.

        Parameters
        ----------
        state:
            The state to record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        self._context.state_history.append((timestamp, state.value))

    def _validate_transition(
        self, from_state: TradeState, to_state: TradeState
    ) -> bool:
        """Check whether a transition is permitted.

        Parameters
        ----------
        from_state:
            Current state.
        to_state:
            Desired target state.

        Returns
        -------
        bool
            ``True`` when the transition is listed in
            :data:`VALID_TRANSITIONS`.
        """
        allowed = VALID_TRANSITIONS.get(from_state, set())
        return to_state in allowed


# ---------------------------------------------------------------------------
# Multi-trade manager
# ---------------------------------------------------------------------------

class TradeManager:
    """Manages multiple :class:`TradeLifecycle` instances.

    Active trades are stored in a dictionary keyed by a string
    identifier (symbol, trade-id, or a caller-chosen key).  Terminal
    trades are moved to a completed list via :meth:`cleanup_completed`.
    """

    def __init__(self) -> None:
        self._active_trades: dict[str, TradeLifecycle] = {}
        self._completed_trades: list[TradeLifecycle] = []

    def create_trade(self, context: TradeContext) -> TradeLifecycle:
        """Create and register a new trade lifecycle.

        The trade is keyed by ``context.symbol``.  If a ``trade_id`` is
        set it is included in the key to allow multiple trades on the
        same symbol.

        Parameters
        ----------
        context:
            The :class:`TradeContext` for the new trade.

        Returns
        -------
        TradeLifecycle
            The newly created lifecycle instance.
        """
        lifecycle = TradeLifecycle(context)

        key = self._make_key(context)
        self._active_trades[key] = lifecycle

        logger.info(
            "trade_created | key={key} symbol={symbol} strategy={strategy}",
            key=key,
            symbol=context.symbol,
            strategy=context.strategy,
        )
        return lifecycle

    def get_active_trades(self) -> list[TradeLifecycle]:
        """Get all non-terminal trades.

        Returns
        -------
        list[TradeLifecycle]
            Trades that have not yet reached a terminal state.
        """
        return [
            t for t in self._active_trades.values() if not t.is_terminal
        ]

    def get_trade(self, key: str) -> Optional[TradeLifecycle]:
        """Look up a trade by its key.

        Parameters
        ----------
        key:
            Symbol or ``symbol_trade-id`` string used when the trade
            was created.

        Returns
        -------
        Optional[TradeLifecycle]
            The matching lifecycle, or ``None`` if not found.
        """
        return self._active_trades.get(key)

    def get_monitoring_trades(self) -> list[TradeLifecycle]:
        """Get trades currently in the MONITORING state.

        Returns
        -------
        list[TradeLifecycle]
            All trades whose current state is
            :attr:`TradeState.MONITORING`.
        """
        return [
            t
            for t in self._active_trades.values()
            if t.state == TradeState.MONITORING
        ]

    def cleanup_completed(self) -> int:
        """Move terminal trades from the active map to the completed list.

        Returns
        -------
        int
            Number of trades moved.
        """
        terminal_keys = [
            k for k, t in self._active_trades.items() if t.is_terminal
        ]
        for key in terminal_keys:
            self._completed_trades.append(self._active_trades.pop(key))

        if terminal_keys:
            logger.info(
                "trades_cleaned_up | count={count} keys={keys}",
                count=len(terminal_keys),
                keys=terminal_keys,
            )
        return len(terminal_keys)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _make_key(context: TradeContext) -> str:
        """Derive a dictionary key from a trade context.

        Uses ``symbol`` alone when ``trade_id`` is ``None``, otherwise
        combines them to support multiple concurrent trades on the same
        symbol.
        """
        if context.trade_id is not None:
            return f"{context.symbol}_{context.trade_id}"
        return context.symbol
