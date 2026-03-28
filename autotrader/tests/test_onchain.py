"""Tests for on-chain whale sell-pressure gating."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whale.onchain import OnChainWhaleDetector


def test_has_sell_pressure_requires_confirmed_cluster(monkeypatch) -> None:
    detector = OnChainWhaleDetector(
        api_key="test",
        min_sell_events=3,
        max_block_minutes=30,
        flag_ttl_minutes=15,
    )
    detector._sell_pressure_events["BTC"] = [100.0, 110.0]

    monkeypatch.setattr("whale.onchain.time.time", lambda: 120.0)

    assert detector.has_sell_pressure("BTC") is False

    detector._sell_pressure_events["BTC"].append(119.0)
    detector._sell_pressure_active_since["BTC"] = 119.0

    assert detector.has_sell_pressure("BTC") is True


def test_sell_pressure_expires_after_max_block_window(monkeypatch) -> None:
    detector = OnChainWhaleDetector(
        api_key="test",
        min_sell_events=2,
        max_block_minutes=30,
        flag_ttl_minutes=60,
    )
    detector._sell_pressure_events["BTC"] = [100.0, 101.0]
    detector._sell_pressure_active_since["BTC"] = 101.0

    monkeypatch.setattr("whale.onchain.time.time", lambda: 101.0 + (31 * 60))

    assert detector.has_sell_pressure("BTC") is False
    assert "BTC" not in detector._sell_pressure_events


def test_exchange_internal_transfer_is_ignored(monkeypatch) -> None:
    detector = OnChainWhaleDetector(
        api_key="test",
        min_sell_events=2,
        monitored_symbols={"BTC"},
    )

    class DummyResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "result": "success",
                "transactions": [
                    {
                        "symbol": "BTC",
                        "amount_usd": 5_000_000,
                        "hash": "abc123",
                        "from": {"owner": "binance"},
                        "to": {"owner": "binance"},
                    }
                ],
            }

    monkeypatch.setattr("whale.onchain.requests.get", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr("whale.onchain.time.time", lambda: 1_000.0)

    detector.poll()

    assert detector._sell_pressure_events == {}
    assert detector.has_sell_pressure("BTC") is False


def test_accumulation_events_neutralize_sell_pressure(monkeypatch) -> None:
    detector = OnChainWhaleDetector(
        api_key="test",
        min_sell_events=3,
        max_block_minutes=30,
        flag_ttl_minutes=15,
    )
    detector._sell_pressure_events["BTC"] = [100.0, 110.0, 119.0]
    detector._sell_pressure_active_since["BTC"] = 119.0
    detector._accumulation_events["BTC"] = [111.0, 118.0, 119.5]

    monkeypatch.setattr("whale.onchain.time.time", lambda: 120.0)

    assert detector.has_sell_pressure("BTC") is False
