"""Tests for Telegram alert delivery helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerts.telegram_bot import TelegramAlert


def test_send_daily_report_sends_plain_text_directly() -> None:
    alert = TelegramAlert("token", "chat")

    with patch.object(alert, "send_message", return_value=True) as mocked_send:
        assert alert.send_daily_report("*Daily Trading Report*\n\n*rsi_momentum*")

    assert mocked_send.call_count == 1
    sent_text = mocked_send.call_args.args[0]
    assert sent_text.startswith("Daily Trading Report")
    assert "rsimomentum" in sent_text
    assert mocked_send.call_args.kwargs["parse_mode"] == ""


def test_send_daily_report_does_not_duplicate_header() -> None:
    alert = TelegramAlert("token", "chat")

    with patch.object(alert, "send_message", return_value=True) as mocked_send:
        alert.send_daily_report("*Daily Trading Report*\n\nReport body")

    sent_text = mocked_send.call_args.args[0]
    assert sent_text.count("Daily Trading Report") == 1
