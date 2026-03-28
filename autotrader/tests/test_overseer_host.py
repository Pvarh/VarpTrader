"""Tests for host overseer helper utilities."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from overseer_host import (
    HostConfig,
    _admission_guard,
    _build_preview,
    _build_prompt,
    _read_limits,
    _record_run_result,
)


def test_build_prompt_includes_trigger_and_context() -> None:
    cfg = HostConfig(
        model="claude-sonnet-4-6",
        trigger_reason="signal_starvation",
        deep=False,
        timeout_sec=300,
        effort="low",
    )

    prompt = _build_prompt("CTX", cfg)

    assert "Trigger reason: signal_starvation" in prompt
    assert "CTX" in prompt
    assert "signal-starvation incident" in prompt
    assert "Focus ONLY on the starvation problem." in prompt


def test_build_preview_skips_blank_lines_and_limits_output() -> None:
    report = "\n\nLine 1\n\nLine 2\nLine 3\nLine 4\n"

    preview = _build_preview(report, max_lines=2)

    assert preview == "Line 1\nLine 2"


def test_admission_guard_blocks_recent_starvation_retry(tmp_path: Path) -> None:
    cfg = HostConfig(
        model="claude-sonnet-4-6",
        trigger_reason="signal_starvation",
        deep=False,
        timeout_sec=300,
        effort="low",
    )
    limits_path = tmp_path / "limits.json"
    limits_path.write_text(
        json.dumps({
            "last_started": {"signal_starvation": "2026-03-22T10:00:00+00:00"},
            "daily_counts": {"2026-03-22": {"total": 1, "signal_starvation": 1, "nightly": 0, "deep": 0}},
        }),
        encoding="utf-8",
    )

    with patch("overseer_host.LIMITS_FILE", limits_path), patch(
        "overseer_host._utc_now",
        return_value=__import__("datetime").datetime(2026, 3, 22, 12, 0, tzinfo=__import__("datetime").timezone.utc),
    ), patch.dict(os.environ, {"OVERSEER_STARVATION_COOLDOWN_SEC": "21600"}, clear=False):
        allowed, reason = _admission_guard(cfg)

    assert allowed is False
    assert "cooldown" in reason


def test_record_run_result_sets_low_credit_lockout(tmp_path: Path) -> None:
    cfg = HostConfig(
        model="claude-sonnet-4-6",
        trigger_reason="nightly",
        deep=False,
        timeout_sec=300,
        effort="low",
    )
    limits_path = tmp_path / "limits.json"

    with patch("overseer_host.LIMITS_FILE", limits_path), patch(
        "overseer_host._utc_now",
        return_value=__import__("datetime").datetime(2026, 3, 22, 10, 0, tzinfo=__import__("datetime").timezone.utc),
    ), patch.dict(os.environ, {"OVERSEER_LOW_CREDIT_LOCKOUT_SEC": "7200"}, clear=False):
        _record_run_result(cfg, "failed", '[Overseer Error] {"result":"Credit balance is too low"}')
        limits = _read_limits()

    assert limits["low_credit_reason"] == "credit balance is too low"
    assert limits["low_credit_trigger_reason"] == "nightly"
    assert limits["low_credit_until"].startswith("2026-03-22T12:00:00")
