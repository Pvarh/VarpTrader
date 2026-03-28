"""Tests for nightly Claude advisor helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.llm_advisor import LLMAdvisor


def test_build_user_prompt_includes_performance_and_bounds() -> None:
    prompt = LLMAdvisor.build_user_prompt(
        {"total_trades": 40, "total_pnl": -100.0},
        {"stop_loss_pct": 0.015},
    )

    assert "total_trades" in prompt
    assert "stop_loss_pct" in prompt
    assert "Allowed parameter bounds" in prompt


def test_parse_response_accepts_markdown_fenced_json() -> None:
    parsed = LLMAdvisor.parse_response(
        '```json\n{"stop_loss_pct": 0.01, "rsi_oversold": 25}\n```'
    )

    assert parsed == {"stop_loss_pct": 0.01, "rsi_oversold": 25.0}


def test_parse_response_filters_unknown_keys() -> None:
    parsed = LLMAdvisor.parse_response(
        '{"stop_loss_pct": 0.01, "unknown_param": 123}'
    )

    assert parsed == {"stop_loss_pct": 0.01}
