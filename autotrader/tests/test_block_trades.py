"""Tests for Polygon block-trade detector behavior."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whale.block_trades import BlockTradeDetector


def test_disabled_detector_skips_api_calls(monkeypatch) -> None:
    detector = BlockTradeDetector(
        api_key="test",
        enabled=False,
    )
    called = {"value": False}

    def _unexpected_get(*args, **kwargs):
        called["value"] = True
        raise AssertionError("requests.get should not be called when detector is disabled")

    monkeypatch.setattr("whale.block_trades.requests.get", _unexpected_get)

    detector.poll(["AAPL", "TSLA"])

    assert called["value"] is False
    assert detector.has_sell_flag("AAPL") is False
    assert detector.has_buy_flag("AAPL") is False
