"""Host-level overseer helper utilities.

Manages admission control, run limits, and prompt construction for the
overseer process that runs on the host (outside Docker).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# Default limits file location — can be patched in tests
LIMITS_FILE: Path = Path(__file__).parent / "overseer" / "host_limits.json"


@dataclass
class HostConfig:
    model: str
    trigger_reason: str
    deep: bool
    timeout_sec: int
    effort: str


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _read_limits() -> dict:
    if LIMITS_FILE.exists():
        return json.loads(LIMITS_FILE.read_text(encoding="utf-8"))
    return {}


def _write_limits(data: dict) -> None:
    LIMITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LIMITS_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _build_preview(report: str, max_lines: int = 20) -> str:
    """Return up to *max_lines* non-blank lines from *report*."""
    lines = [ln for ln in report.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


def _build_prompt(context: str, cfg: HostConfig) -> str:
    """Build the overseer prompt string for the given *context* and *cfg*."""
    trigger = cfg.trigger_reason
    starvation_section = ""
    if trigger == "signal_starvation":
        starvation_section = (
            "\n\nThis is a signal-starvation incident. "
            "Focus ONLY on the starvation problem. "
            "Identify which filter is over-blocking and loosen it conservatively."
        )

    return (
        f"Trigger reason: {trigger}\n\n"
        f"{context}"
        f"{starvation_section}"
    )


def _admission_guard(cfg: HostConfig) -> Tuple[bool, str]:
    """Return (allowed, reason).  Blocks runs that violate cooldown or daily limits."""
    limits = _read_limits()
    now = _utc_now()

    # Low-credit lockout
    until_str = limits.get("low_credit_until")
    if until_str:
        until = datetime.fromisoformat(until_str)
        if now < until:
            return False, f"low-credit lockout until {until_str}"

    # Per-trigger cooldown
    # "signal_starvation" → OVERSEER_STARVATION_COOLDOWN_SEC (strip "signal_" prefix)
    trigger_slug = cfg.trigger_reason.replace("signal_", "").upper()
    cooldown_env_key = f"OVERSEER_{trigger_slug}_COOLDOWN_SEC"
    cooldown_sec = int(os.environ.get(cooldown_env_key, 0))
    if cooldown_sec > 0:
        last_started: dict = limits.get("last_started", {})
        last_str = last_started.get(cfg.trigger_reason)
        if last_str:
            last = datetime.fromisoformat(last_str)
            elapsed = (now - last).total_seconds()
            if elapsed < cooldown_sec:
                remaining = int(cooldown_sec - elapsed)
                return False, f"cooldown: {remaining}s remaining for {cfg.trigger_reason}"

    return True, "ok"


def _record_run_result(cfg: HostConfig, status: str, output: str) -> None:
    """Persist run result; set low-credit lockout when credits are exhausted."""
    limits = _read_limits()
    now = _utc_now()

    # Track last_started per trigger
    last_started = limits.setdefault("last_started", {})
    last_started[cfg.trigger_reason] = now.isoformat()

    # Daily counts
    today = now.date().isoformat()
    daily = limits.setdefault("daily_counts", {})
    day_counts = daily.setdefault(today, {"total": 0, "signal_starvation": 0, "nightly": 0, "deep": 0})
    day_counts["total"] = day_counts.get("total", 0) + 1
    key = cfg.trigger_reason if cfg.trigger_reason in day_counts else "total"
    if cfg.trigger_reason in day_counts:
        day_counts[cfg.trigger_reason] = day_counts.get(cfg.trigger_reason, 0) + 1

    # Low-credit lockout
    if status == "failed" and "credit balance is too low" in output.lower():
        lockout_sec = int(os.environ.get("OVERSEER_LOW_CREDIT_LOCKOUT_SEC", 3600))
        until = datetime.fromtimestamp(now.timestamp() + lockout_sec, tz=timezone.utc)
        limits["low_credit_until"] = until.isoformat()
        limits["low_credit_reason"] = "credit balance is too low"
        limits["low_credit_trigger_reason"] = cfg.trigger_reason

    _write_limits(limits)
