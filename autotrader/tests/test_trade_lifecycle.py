"""Tests for the trade lifecycle state machine."""

import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, ".")

from execution.trade_lifecycle import (
    TradeContext,
    TradeLifecycle,
    TradeManager,
    TradeState,
    VALID_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_context(**overrides) -> TradeContext:
    """Create a TradeContext with sensible defaults."""
    defaults = dict(
        trade_id=1,
        symbol="AAPL",
        market="stock",
        strategy="first_candle",
        direction="long",
        entry_price=150.0,
        quantity=10.0,
        stop_loss=148.5,
        take_profit=153.0,
    )
    defaults.update(overrides)
    return TradeContext(**defaults)


def advance_to(lifecycle: TradeLifecycle, *states: TradeState) -> None:
    """Walk a lifecycle through a sequence of states, asserting each succeeds."""
    for state in states:
        assert lifecycle.transition_to(state), (
            f"Expected transition to {state.value} to succeed, "
            f"but it failed from {lifecycle.state.value}"
        )


# ===========================================================================
# Happy-path transition chain
# ===========================================================================

class TestValidTransitions:
    """Full happy-path through the trade lifecycle."""

    def test_full_lifecycle_signal_to_closed(self) -> None:
        """SIGNAL -> VALIDATED -> SUBMITTED -> FILLED -> MONITORING -> CLOSING -> CLOSED."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)

        assert lc.state == TradeState.SIGNAL

        advance_to(
            lc,
            TradeState.VALIDATED,
            TradeState.SUBMITTED,
            TradeState.FILLED,
            TradeState.MONITORING,
            TradeState.CLOSING,
            TradeState.CLOSED,
        )

        assert lc.state == TradeState.CLOSED
        assert lc.is_terminal is True
        assert lc.is_active is False

    def test_monitoring_to_closed_directly(self) -> None:
        """Trades may go from MONITORING straight to CLOSED (e.g. stop hit)."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(
            lc,
            TradeState.VALIDATED,
            TradeState.SUBMITTED,
            TradeState.FILLED,
            TradeState.MONITORING,
            TradeState.CLOSED,
        )
        assert lc.state == TradeState.CLOSED


# ===========================================================================
# Invalid transitions
# ===========================================================================

class TestInvalidTransitions:
    """Transitions that skip states must be rejected."""

    def test_signal_to_filled_is_invalid(self) -> None:
        """Cannot skip VALIDATED and SUBMITTED stages."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        result = lc.transition_to(TradeState.FILLED)
        assert result is False
        assert lc.state == TradeState.SIGNAL

    def test_signal_to_monitoring_is_invalid(self) -> None:
        """Cannot jump from SIGNAL directly to MONITORING."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        assert lc.transition_to(TradeState.MONITORING) is False

    def test_validated_to_filled_is_invalid(self) -> None:
        """Must go through SUBMITTED between VALIDATED and FILLED."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED)
        assert lc.transition_to(TradeState.FILLED) is False

    def test_filled_to_closed_is_invalid(self) -> None:
        """FILLED must go to MONITORING, not directly to CLOSED."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED, TradeState.SUBMITTED, TradeState.FILLED)
        assert lc.transition_to(TradeState.CLOSED) is False


# ===========================================================================
# Terminal states
# ===========================================================================

class TestTerminalStates:
    """Terminal states must not allow any outgoing transition."""

    @pytest.mark.parametrize(
        "terminal_path,terminal_state",
        [
            ([TradeState.REJECTED], TradeState.REJECTED),
            (
                [TradeState.VALIDATED, TradeState.SUBMITTED, TradeState.CANCELLED],
                TradeState.CANCELLED,
            ),
            (
                [
                    TradeState.VALIDATED,
                    TradeState.SUBMITTED,
                    TradeState.FILLED,
                    TradeState.MONITORING,
                    TradeState.CLOSING,
                    TradeState.CLOSED,
                ],
                TradeState.CLOSED,
            ),
        ],
        ids=["rejected", "cancelled", "closed"],
    )
    def test_terminal_state_blocks_all_transitions(
        self, terminal_path: list[TradeState], terminal_state: TradeState
    ) -> None:
        """No state is reachable from a terminal state."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, *terminal_path)
        assert lc.is_terminal is True

        for target in TradeState:
            assert lc.transition_to(target) is False, (
                f"Should not be able to transition from {terminal_state.value} "
                f"to {target.value}"
            )


# ===========================================================================
# State history
# ===========================================================================

class TestStateHistory:
    """State history must faithfully record every transition."""

    def test_initial_state_is_recorded(self) -> None:
        """Creating a lifecycle records the initial SIGNAL state."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        assert len(ctx.state_history) == 1
        assert ctx.state_history[0][1] == TradeState.SIGNAL.value

    def test_transitions_accumulate_in_history(self) -> None:
        """Each successful transition appends to the history."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED, TradeState.SUBMITTED, TradeState.FILLED)

        expected_states = ["signal", "validated", "submitted", "filled"]
        recorded_states = [entry[1] for entry in ctx.state_history]
        assert recorded_states == expected_states

    def test_failed_transition_does_not_add_history(self) -> None:
        """Invalid transitions must not appear in history."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        lc.transition_to(TradeState.FILLED)  # invalid

        assert len(ctx.state_history) == 1
        assert ctx.state_history[0][1] == TradeState.SIGNAL.value

    def test_history_entries_have_iso_timestamps(self) -> None:
        """Each history entry carries a parseable ISO-format timestamp."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED)

        for ts_str, _ in ctx.state_history:
            dt = datetime.fromisoformat(ts_str)
            assert dt.tzinfo is not None  # must be timezone-aware


# ===========================================================================
# Context updates via kwargs
# ===========================================================================

class TestKwargsUpdateContext:
    """Keyword arguments passed to transition_to update the context."""

    def test_order_id_set_on_submitted(self) -> None:
        """Setting order_id when transitioning to SUBMITTED."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED)
        lc.transition_to(TradeState.SUBMITTED, order_id="ORD-123")
        assert ctx.order_id == "ORD-123"

    def test_exit_price_and_pnl_set_on_closed(self) -> None:
        """PnL fields populated when the trade closes."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(
            lc,
            TradeState.VALIDATED,
            TradeState.SUBMITTED,
            TradeState.FILLED,
            TradeState.MONITORING,
            TradeState.CLOSING,
        )
        lc.transition_to(
            TradeState.CLOSED,
            exit_price=152.0,
            pnl=20.0,
            pnl_pct=0.0133,
            outcome="win",
        )
        assert ctx.exit_price == 152.0
        assert ctx.pnl == 20.0
        assert ctx.pnl_pct == pytest.approx(0.0133)
        assert ctx.outcome == "win"

    def test_rejection_reason_set_on_rejected(self) -> None:
        """A rejection reason is captured in context."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        lc.transition_to(
            TradeState.REJECTED,
            rejection_reason="Kill switch active",
        )
        assert ctx.rejection_reason == "Kill switch active"

    def test_error_message_set_on_failed(self) -> None:
        """An error message is recorded when transitioning to FAILED."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED, TradeState.SUBMITTED)
        lc.transition_to(
            TradeState.FAILED,
            error_message="Connection timeout",
        )
        assert ctx.error_message == "Connection timeout"

    def test_unknown_kwarg_is_ignored(self) -> None:
        """Unknown fields in kwargs do not raise; they are logged."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        result = lc.transition_to(
            TradeState.VALIDATED, nonexistent_field="value"
        )
        assert result is True
        assert not hasattr(ctx, "nonexistent_field")


# ===========================================================================
# Rejection flow
# ===========================================================================

class TestRejectionFlow:
    """Signal -> REJECTED is a valid one-step terminal path."""

    def test_signal_to_rejected(self) -> None:
        """Trade can be rejected immediately from SIGNAL state."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        result = lc.transition_to(
            TradeState.REJECTED,
            rejection_reason="Max positions reached",
        )
        assert result is True
        assert lc.state == TradeState.REJECTED
        assert lc.is_terminal is True
        assert ctx.rejection_reason == "Max positions reached"

    def test_validated_to_rejected(self) -> None:
        """Trade can also be rejected after validation (late check failure)."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED)
        result = lc.transition_to(
            TradeState.REJECTED,
            rejection_reason="Duplicate position",
        )
        assert result is True
        assert lc.state == TradeState.REJECTED


# ===========================================================================
# Retry flow
# ===========================================================================

class TestRetryFlow:
    """FAILED -> SUBMITTED is the only valid retry path."""

    def test_failed_to_submitted_retry(self) -> None:
        """A failed submission can be retried."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED, TradeState.SUBMITTED)
        lc.transition_to(TradeState.FAILED, error_message="Timeout")
        assert lc.state == TradeState.FAILED

        result = lc.transition_to(TradeState.SUBMITTED, order_id="ORD-RETRY")
        assert result is True
        assert lc.state == TradeState.SUBMITTED
        assert ctx.order_id == "ORD-RETRY"

    def test_failed_cannot_jump_to_filled(self) -> None:
        """FAILED -> FILLED is not allowed; must go through SUBMITTED."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED, TradeState.SUBMITTED)
        lc.transition_to(TradeState.FAILED)

        assert lc.transition_to(TradeState.FILLED) is False

    def test_failed_is_not_terminal(self) -> None:
        """FAILED allows a retry, so it is not a terminal state."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)
        advance_to(lc, TradeState.VALIDATED, TradeState.SUBMITTED)
        lc.transition_to(TradeState.FAILED)

        assert lc.is_terminal is False


# ===========================================================================
# TradeManager
# ===========================================================================

class TestTradeManager:
    """Tests for the multi-trade manager."""

    def test_create_trade(self) -> None:
        """Creating a trade registers it and returns a lifecycle."""
        mgr = TradeManager()
        ctx = make_context(symbol="AAPL")
        lc = mgr.create_trade(ctx)

        assert isinstance(lc, TradeLifecycle)
        assert lc.state == TradeState.SIGNAL

    def test_get_trade_by_symbol(self) -> None:
        """Look up trade by symbol key."""
        mgr = TradeManager()
        ctx = make_context(symbol="TSLA", trade_id=None)
        mgr.create_trade(ctx)

        retrieved = mgr.get_trade("TSLA")
        assert retrieved is not None
        assert retrieved.context.symbol == "TSLA"

    def test_get_trade_by_symbol_and_id(self) -> None:
        """Look up trade by combined symbol_trade-id key."""
        mgr = TradeManager()
        ctx = make_context(symbol="AAPL", trade_id=42)
        mgr.create_trade(ctx)

        assert mgr.get_trade("AAPL_42") is not None
        assert mgr.get_trade("AAPL") is None

    def test_get_trade_returns_none_for_missing(self) -> None:
        """Returns None when the key does not match any trade."""
        mgr = TradeManager()
        assert mgr.get_trade("NOPE") is None

    def test_get_active_trades_excludes_terminal(self) -> None:
        """Only non-terminal trades are returned by get_active_trades."""
        mgr = TradeManager()
        ctx_active = make_context(symbol="AAPL", trade_id=1)
        ctx_rejected = make_context(symbol="GOOG", trade_id=2)

        lc_active = mgr.create_trade(ctx_active)
        lc_rejected = mgr.create_trade(ctx_rejected)

        advance_to(lc_active, TradeState.VALIDATED)
        lc_rejected.transition_to(TradeState.REJECTED)

        active = mgr.get_active_trades()
        assert len(active) == 1
        assert active[0].context.symbol == "AAPL"

    def test_get_monitoring_trades(self) -> None:
        """Only MONITORING trades are returned."""
        mgr = TradeManager()
        ctx1 = make_context(symbol="AAPL", trade_id=1)
        ctx2 = make_context(symbol="GOOG", trade_id=2)

        lc1 = mgr.create_trade(ctx1)
        lc2 = mgr.create_trade(ctx2)

        advance_to(
            lc1,
            TradeState.VALIDATED,
            TradeState.SUBMITTED,
            TradeState.FILLED,
            TradeState.MONITORING,
        )
        advance_to(lc2, TradeState.VALIDATED, TradeState.SUBMITTED)

        monitoring = mgr.get_monitoring_trades()
        assert len(monitoring) == 1
        assert monitoring[0].context.symbol == "AAPL"

    def test_cleanup_completed(self) -> None:
        """Terminal trades are moved from active to completed."""
        mgr = TradeManager()
        ctx1 = make_context(symbol="AAPL", trade_id=1)
        ctx2 = make_context(symbol="GOOG", trade_id=2)

        lc1 = mgr.create_trade(ctx1)
        mgr.create_trade(ctx2)

        lc1.transition_to(TradeState.REJECTED)

        count = mgr.cleanup_completed()
        assert count == 1
        assert len(mgr._completed_trades) == 1
        assert mgr._completed_trades[0].context.symbol == "AAPL"
        assert mgr.get_trade("AAPL_1") is None  # removed from active

    def test_cleanup_returns_zero_when_none_terminal(self) -> None:
        """No-op when there are no terminal trades."""
        mgr = TradeManager()
        ctx = make_context(symbol="AAPL", trade_id=1)
        mgr.create_trade(ctx)

        assert mgr.cleanup_completed() == 0


# ===========================================================================
# Callbacks
# ===========================================================================

class TestCallbacks:
    """State-change callbacks fire with the correct arguments."""

    def test_callback_receives_old_new_context(self) -> None:
        """Callback is invoked with (old_state, new_state, context)."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)

        captured: list[tuple[TradeState, TradeState, TradeContext]] = []
        lc.on_state_change(lambda old, new, c: captured.append((old, new, c)))

        lc.transition_to(TradeState.VALIDATED)

        assert len(captured) == 1
        old, new, context = captured[0]
        assert old == TradeState.SIGNAL
        assert new == TradeState.VALIDATED
        assert context is ctx

    def test_callback_fires_for_each_transition(self) -> None:
        """The callback fires once per successful transition."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)

        call_count = 0

        def counter(old: TradeState, new: TradeState, c: TradeContext) -> None:
            nonlocal call_count
            call_count += 1

        lc.on_state_change(counter)
        advance_to(lc, TradeState.VALIDATED, TradeState.SUBMITTED, TradeState.FILLED)

        assert call_count == 3

    def test_callback_not_fired_on_invalid_transition(self) -> None:
        """Invalid transitions must not invoke callbacks."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)

        called = False

        def flag(old: TradeState, new: TradeState, c: TradeContext) -> None:
            nonlocal called
            called = True

        lc.on_state_change(flag)
        lc.transition_to(TradeState.FILLED)  # invalid from SIGNAL

        assert called is False

    def test_multiple_callbacks_all_fire(self) -> None:
        """All registered callbacks are invoked on transition."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)

        results: list[str] = []
        lc.on_state_change(lambda o, n, c: results.append("cb1"))
        lc.on_state_change(lambda o, n, c: results.append("cb2"))

        lc.transition_to(TradeState.VALIDATED)
        assert results == ["cb1", "cb2"]

    def test_callback_exception_does_not_break_transition(self) -> None:
        """A failing callback must not prevent the state change."""
        ctx = make_context()
        lc = TradeLifecycle(ctx)

        def bad_callback(o: TradeState, n: TradeState, c: TradeContext) -> None:
            raise RuntimeError("boom")

        safe_called = False

        def safe_callback(o: TradeState, n: TradeState, c: TradeContext) -> None:
            nonlocal safe_called
            safe_called = True

        lc.on_state_change(bad_callback)
        lc.on_state_change(safe_callback)

        result = lc.transition_to(TradeState.VALIDATED)
        assert result is True
        assert lc.state == TradeState.VALIDATED
        assert safe_called is True


# ===========================================================================
# TradeContext helpers
# ===========================================================================

class TestTradeContext:
    """Tests for TradeContext utility methods."""

    def test_elapsed_seconds_is_non_negative(self) -> None:
        """Elapsed seconds should be >= 0 for a freshly-created context."""
        ctx = make_context()
        assert ctx.elapsed_seconds() >= 0.0

    def test_elapsed_seconds_increases(self) -> None:
        """Contexts created with an older timestamp report more elapsed time."""
        ctx = make_context(created_at="2020-01-01T00:00:00+00:00")
        assert ctx.elapsed_seconds() > 0.0

    def test_default_state_is_signal(self) -> None:
        """A fresh context starts in SIGNAL state."""
        ctx = TradeContext()
        assert ctx.state == TradeState.SIGNAL


# ===========================================================================
# Transition map completeness
# ===========================================================================

class TestTransitionMap:
    """The transition map must cover every TradeState."""

    def test_every_state_has_an_entry(self) -> None:
        """VALID_TRANSITIONS should contain a key for every TradeState."""
        for state in TradeState:
            assert state in VALID_TRANSITIONS, (
                f"{state.value} is missing from VALID_TRANSITIONS"
            )
