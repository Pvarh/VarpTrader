"""Tests for the swing advisor module."""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, ".")

from analysis.swing_advisor import SwingAdvisor, WEEKLY_BIAS_PATH
from journal.db import TradeDatabase
from journal.models import SwingBias


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> TradeDatabase:
    """Create a fresh database in a temp directory."""
    db_path = str(tmp_path / "test_swing.db")
    database = TradeDatabase(db_path)
    yield database
    database.close()


@pytest.fixture
def advisor(db) -> SwingAdvisor:
    """Create a SwingAdvisor with dummy API keys."""
    return SwingAdvisor(
        anthropic_api_key="test-key-123",
        news_api_key="test-news-key-456",
        db=db,
        model="claude-sonnet-4-6",
        min_confidence=60,
    )


@pytest.fixture
def valid_bias_file(tmp_path) -> Path:
    """Create a valid weekly_bias.json in tmp_path."""
    now = datetime.now(timezone.utc)
    data = {
        "week_start": now.strftime("%Y-%m-%d"),
        "generated_at": now.isoformat(),
        "biases": {
            "AAPL": {
                "symbol": "AAPL",
                "bias": "bearish",
                "confidence": 75,
                "reason": "Weak earnings outlook and sector rotation.",
            },
            "TSLA": {
                "symbol": "TSLA",
                "bias": "bullish",
                "confidence": 80,
                "reason": "Strong delivery numbers beat expectations.",
            },
            "NVDA": {
                "symbol": "NVDA",
                "bias": "neutral",
                "confidence": 45,
                "reason": "Mixed signals from AI spending.",
            },
            "SPY": {
                "symbol": "SPY",
                "bias": "bearish",
                "confidence": 50,
                "reason": "Low confidence bearish signal.",
            },
        },
    }
    path = tmp_path / "weekly_bias.json"
    path.write_text(json.dumps(data, indent=2))
    return path


@pytest.fixture
def stale_bias_file(tmp_path) -> Path:
    """Create a weekly_bias.json that is >8 days old."""
    old_dt = datetime.now(timezone.utc) - timedelta(days=10)
    data = {
        "week_start": old_dt.strftime("%Y-%m-%d"),
        "generated_at": old_dt.isoformat(),
        "biases": {
            "AAPL": {
                "symbol": "AAPL",
                "bias": "bearish",
                "confidence": 90,
                "reason": "Stale data.",
            },
        },
    }
    path = tmp_path / "weekly_bias.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------------------
# should_block_trade tests
# ---------------------------------------------------------------------------

class TestShouldBlockTrade:
    """Tests for SwingAdvisor.should_block_trade static method."""

    def test_bearish_bias_blocks_long(self, valid_bias_file: Path) -> None:
        """Bearish bias with high confidence should block long trades."""
        blocked = SwingAdvisor.should_block_trade(
            "AAPL", "long", min_confidence=60, bias_path=valid_bias_file,
        )
        assert blocked is True

    def test_bullish_bias_blocks_short(self, valid_bias_file: Path) -> None:
        """Bullish bias with high confidence should block short trades."""
        blocked = SwingAdvisor.should_block_trade(
            "TSLA", "short", min_confidence=60, bias_path=valid_bias_file,
        )
        assert blocked is True

    def test_neutral_allows_all(self, valid_bias_file: Path) -> None:
        """Neutral bias should allow both long and short trades."""
        assert SwingAdvisor.should_block_trade(
            "NVDA", "long", min_confidence=60, bias_path=valid_bias_file,
        ) is False
        assert SwingAdvisor.should_block_trade(
            "NVDA", "short", min_confidence=60, bias_path=valid_bias_file,
        ) is False

    def test_low_confidence_allows_through(self, valid_bias_file: Path) -> None:
        """Bearish bias with confidence below threshold should allow trades."""
        # SPY has bearish bias but only 50 confidence
        blocked = SwingAdvisor.should_block_trade(
            "SPY", "long", min_confidence=60, bias_path=valid_bias_file,
        )
        assert blocked is False

    def test_missing_file_allows_through(self, tmp_path: Path) -> None:
        """Missing weekly_bias.json should allow all trades."""
        missing_path = tmp_path / "nonexistent.json"
        blocked = SwingAdvisor.should_block_trade(
            "AAPL", "long", min_confidence=60, bias_path=missing_path,
        )
        assert blocked is False

    def test_stale_file_allows_through(self, stale_bias_file: Path) -> None:
        """Stale weekly_bias.json (>8 days old) should allow all trades."""
        blocked = SwingAdvisor.should_block_trade(
            "AAPL", "long", min_confidence=60, bias_path=stale_bias_file,
        )
        assert blocked is False

    def test_bearish_allows_short(self, valid_bias_file: Path) -> None:
        """Bearish bias should NOT block short trades (same direction)."""
        blocked = SwingAdvisor.should_block_trade(
            "AAPL", "short", min_confidence=60, bias_path=valid_bias_file,
        )
        assert blocked is False

    def test_bullish_allows_long(self, valid_bias_file: Path) -> None:
        """Bullish bias should NOT block long trades (same direction)."""
        blocked = SwingAdvisor.should_block_trade(
            "TSLA", "long", min_confidence=60, bias_path=valid_bias_file,
        )
        assert blocked is False

    def test_unknown_symbol_allows_through(self, valid_bias_file: Path) -> None:
        """Symbol not present in bias file should allow trade."""
        blocked = SwingAdvisor.should_block_trade(
            "MSFT", "long", min_confidence=60, bias_path=valid_bias_file,
        )
        assert blocked is False


# ---------------------------------------------------------------------------
# _build_payload tests
# ---------------------------------------------------------------------------

class TestBuildPayload:
    """Tests for SwingAdvisor._build_payload."""

    def test_build_payload_structure(self, advisor: SwingAdvisor) -> None:
        """Payload should contain all expected keys with correct values."""
        news = ["Headline 1", "Headline 2"]
        fundamentals = {
            "pe_ratio": 25.4,
            "market_cap": 2800000000000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        price_summary = {
            "pct_change_5d": 2.5,
            "avg_volume_5d": 50000000,
            "avg_volume_20d": 45000000,
            "volume_ratio": 1.11,
        }

        payload = advisor._build_payload("AAPL", news, fundamentals, price_summary)

        assert payload["symbol"] == "AAPL"
        assert payload["news_headlines"] == ["Headline 1", "Headline 2"]
        assert payload["fundamentals"]["pe_ratio"] == 25.4
        assert payload["fundamentals"]["market_cap"] == 2800000000000
        assert payload["fundamentals"]["sector"] == "Technology"
        assert payload["fundamentals"]["industry"] == "Consumer Electronics"
        assert payload["price_summary"]["pct_change_5d"] == 2.5
        assert payload["price_summary"]["avg_volume_5d"] == 50000000
        assert payload["price_summary"]["avg_volume_20d"] == 45000000
        assert payload["price_summary"]["volume_ratio"] == 1.11

    def test_build_payload_empty_news(self, advisor: SwingAdvisor) -> None:
        """Payload should work with empty news list."""
        payload = advisor._build_payload("TSLA", [], {}, {})
        assert payload["symbol"] == "TSLA"
        assert payload["news_headlines"] == []
        assert payload["fundamentals"] == {}
        assert payload["price_summary"] == {}


# ---------------------------------------------------------------------------
# _query_claude tests
# ---------------------------------------------------------------------------

class TestQueryClaude:
    """Tests for SwingAdvisor._query_claude."""

    def test_query_claude_valid_response(self, advisor: SwingAdvisor) -> None:
        """Should parse a valid JSON array from Claude's response."""
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = json.dumps([
            {
                "symbol": "AAPL",
                "bias": "bearish",
                "confidence": 72,
                "reason": "Declining revenue guidance.",
            },
            {
                "symbol": "TSLA",
                "bias": "bullish",
                "confidence": 65,
                "reason": "Record deliveries exceeded forecasts.",
            },
        ])
        mock_response.content = [mock_content_block]

        advisor._client = MagicMock()
        advisor._client.messages.create.return_value = mock_response

        payloads = [
            {"symbol": "AAPL", "news_headlines": [], "fundamentals": {}, "price_summary": {}},
            {"symbol": "TSLA", "news_headlines": [], "fundamentals": {}, "price_summary": {}},
        ]

        biases, raw = advisor._query_claude(payloads)

        assert len(biases) == 2
        assert biases[0]["symbol"] == "AAPL"
        assert biases[0]["bias"] == "bearish"
        assert biases[0]["confidence"] == 72
        assert biases[1]["symbol"] == "TSLA"
        assert biases[1]["bias"] == "bullish"
        assert biases[1]["confidence"] == 65
        assert raw != ""

    def test_query_claude_invalid_response_fallback(self, advisor: SwingAdvisor) -> None:
        """Should return neutral fallback when Claude returns invalid JSON."""
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "I'm sorry, I cannot analyze these tickers right now."
        mock_response.content = [mock_content_block]

        advisor._client = MagicMock()
        advisor._client.messages.create.return_value = mock_response

        payloads = [
            {"symbol": "AAPL", "news_headlines": [], "fundamentals": {}, "price_summary": {}},
            {"symbol": "NVDA", "news_headlines": [], "fundamentals": {}, "price_summary": {}},
        ]

        biases, raw = advisor._query_claude(payloads)

        assert len(biases) == 2
        for b in biases:
            assert b["bias"] == "neutral"
            assert b["confidence"] == 0

    def test_query_claude_api_error_fallback(self, advisor: SwingAdvisor) -> None:
        """Should return neutral fallback on anthropic API error."""
        import anthropic as anthropic_mod

        advisor._client = MagicMock()
        advisor._client.messages.create.side_effect = anthropic_mod.APIError(
            message="Rate limited",
            request=MagicMock(),
            body=None,
        )

        payloads = [
            {"symbol": "AAPL", "news_headlines": [], "fundamentals": {}, "price_summary": {}},
        ]

        biases, raw = advisor._query_claude(payloads)

        assert len(biases) == 1
        assert biases[0]["bias"] == "neutral"
        assert biases[0]["confidence"] == 0

    def test_query_claude_markdown_fenced_response(self, advisor: SwingAdvisor) -> None:
        """Should correctly parse JSON wrapped in markdown code fences."""
        raw_json = json.dumps([
            {"symbol": "SPY", "bias": "bullish", "confidence": 68, "reason": "Uptrend intact."},
        ])
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = f"```json\n{raw_json}\n```"
        mock_response.content = [mock_content_block]

        advisor._client = MagicMock()
        advisor._client.messages.create.return_value = mock_response

        payloads = [
            {"symbol": "SPY", "news_headlines": [], "fundamentals": {}, "price_summary": {}},
        ]

        biases, raw = advisor._query_claude(payloads)

        assert len(biases) == 1
        assert biases[0]["symbol"] == "SPY"
        assert biases[0]["bias"] == "bullish"
        assert biases[0]["confidence"] == 68

    def test_query_claude_empty_content(self, advisor: SwingAdvisor) -> None:
        """Should return neutral fallback when Claude returns empty content."""
        mock_response = MagicMock()
        mock_response.content = []

        advisor._client = MagicMock()
        advisor._client.messages.create.return_value = mock_response

        payloads = [
            {"symbol": "QQQ", "news_headlines": [], "fundamentals": {}, "price_summary": {}},
        ]

        biases, raw = advisor._query_claude(payloads)

        assert len(biases) == 1
        assert biases[0]["bias"] == "neutral"


# ---------------------------------------------------------------------------
# _write_weekly_bias tests
# ---------------------------------------------------------------------------

class TestWriteWeeklyBias:
    """Tests for SwingAdvisor._write_weekly_bias."""

    def test_write_weekly_bias_creates_valid_json(
        self, advisor: SwingAdvisor, tmp_path: Path, monkeypatch
    ) -> None:
        """Should create a valid JSON file with correct structure."""
        output_path = tmp_path / "weekly_bias.json"
        monkeypatch.setattr(
            "analysis.swing_advisor.WEEKLY_BIAS_PATH", output_path,
        )

        biases = [
            {"symbol": "AAPL", "bias": "bearish", "confidence": 72, "reason": "Weak outlook."},
            {"symbol": "TSLA", "bias": "bullish", "confidence": 80, "reason": "Strong sales."},
        ]

        advisor._write_weekly_bias(biases)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "week_start" in data
        assert "generated_at" in data
        assert "biases" in data
        assert "AAPL" in data["biases"]
        assert data["biases"]["AAPL"]["bias"] == "bearish"
        assert data["biases"]["AAPL"]["confidence"] == 72
        assert "TSLA" in data["biases"]
        assert data["biases"]["TSLA"]["bias"] == "bullish"


# ---------------------------------------------------------------------------
# load_weekly_bias tests
# ---------------------------------------------------------------------------

class TestLoadWeeklyBias:
    """Tests for SwingAdvisor.load_weekly_bias."""

    def test_load_valid_file(self, valid_bias_file: Path) -> None:
        """Should load and return data from a valid, recent bias file."""
        data = SwingAdvisor.load_weekly_bias(valid_bias_file)
        assert "biases" in data
        assert "AAPL" in data["biases"]
        assert data["biases"]["AAPL"]["bias"] == "bearish"
        assert data["biases"]["TSLA"]["bias"] == "bullish"

    def test_load_stale_file(self, stale_bias_file: Path) -> None:
        """Should return empty biases for a file older than 8 days."""
        data = SwingAdvisor.load_weekly_bias(stale_bias_file)
        assert data["biases"] == {}

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Should return empty biases when file does not exist."""
        missing_path = tmp_path / "nope.json"
        data = SwingAdvisor.load_weekly_bias(missing_path)
        assert data["biases"] == {}

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        """Should return empty biases for a corrupt JSON file."""
        path = tmp_path / "weekly_bias.json"
        path.write_text("{not valid json!!!")
        data = SwingAdvisor.load_weekly_bias(path)
        assert data["biases"] == {}


# ---------------------------------------------------------------------------
# run() end-to-end test
# ---------------------------------------------------------------------------

class TestRunEndToEnd:
    """End-to-end test for SwingAdvisor.run with all externals mocked."""

    def test_run_full_pipeline(self, advisor: SwingAdvisor, tmp_path: Path, monkeypatch) -> None:
        """run() should fetch data, query Claude, write file, and log to DB."""
        output_path = tmp_path / "weekly_bias.json"
        monkeypatch.setattr(
            "analysis.swing_advisor.WEEKLY_BIAS_PATH", output_path,
        )

        # Mock news fetch
        monkeypatch.setattr(
            advisor, "_fetch_news",
            lambda symbol: [f"{symbol} headline 1", f"{symbol} headline 2"],
        )

        # Mock fundamentals fetch
        monkeypatch.setattr(
            advisor, "_fetch_fundamentals",
            lambda symbol: {
                "pe_ratio": 20.0,
                "market_cap": 1000000000,
                "sector": "Tech",
                "industry": "Software",
            },
        )

        # Mock price summary fetch
        monkeypatch.setattr(
            advisor, "_fetch_price_summary",
            lambda symbol: {
                "pct_change_5d": 1.5,
                "avg_volume_5d": 5000000,
                "avg_volume_20d": 4500000,
                "volume_ratio": 1.11,
            },
        )

        # Mock Claude response
        claude_response_data = [
            {"symbol": "AAPL", "bias": "bearish", "confidence": 70, "reason": "Weak guidance."},
            {"symbol": "TSLA", "bias": "bullish", "confidence": 82, "reason": "Record deliveries."},
            {"symbol": "NVDA", "bias": "neutral", "confidence": 40, "reason": "Mixed signals."},
        ]
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = json.dumps(claude_response_data)
        mock_response.content = [mock_content_block]

        advisor._client = MagicMock()
        advisor._client.messages.create.return_value = mock_response

        # Run the pipeline
        symbols = ["AAPL", "TSLA", "NVDA"]
        summary = advisor.run(symbols)

        # Verify summary
        assert summary["symbols_analyzed"] == 3
        assert summary["biases"]["AAPL"] == "bearish"
        assert summary["biases"]["TSLA"] == "bullish"
        assert summary["biases"]["NVDA"] == "neutral"
        assert "generated_at" in summary

        # Verify weekly_bias.json was written
        assert output_path.exists()
        file_data = json.loads(output_path.read_text())
        assert "AAPL" in file_data["biases"]
        assert file_data["biases"]["AAPL"]["bias"] == "bearish"
        assert file_data["biases"]["TSLA"]["bias"] == "bullish"

        # Verify Claude was called exactly once
        advisor._client.messages.create.assert_called_once()

        # Verify DB logging: check swing_bias_log table
        week_start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        db_biases = advisor._db.get_swing_biases_for_week(week_start)
        assert len(db_biases) == 3
        symbols_logged = {row["symbol"] for row in db_biases}
        assert symbols_logged == {"AAPL", "TSLA", "NVDA"}

    def test_run_with_claude_failure_still_writes_neutral(
        self, advisor: SwingAdvisor, tmp_path: Path, monkeypatch,
    ) -> None:
        """run() should still write neutral biases when Claude API fails."""
        import anthropic as anthropic_mod

        output_path = tmp_path / "weekly_bias.json"
        monkeypatch.setattr(
            "analysis.swing_advisor.WEEKLY_BIAS_PATH", output_path,
        )

        monkeypatch.setattr(advisor, "_fetch_news", lambda symbol: [])
        monkeypatch.setattr(
            advisor, "_fetch_fundamentals",
            lambda symbol: {"pe_ratio": None, "market_cap": None, "sector": "N/A", "industry": "N/A"},
        )
        monkeypatch.setattr(
            advisor, "_fetch_price_summary",
            lambda symbol: {"pct_change_5d": None, "avg_volume_5d": None, "avg_volume_20d": None, "volume_ratio": None},
        )

        advisor._client = MagicMock()
        advisor._client.messages.create.side_effect = anthropic_mod.APIError(
            message="Service unavailable",
            request=MagicMock(),
            body=None,
        )

        summary = advisor.run(["AAPL", "TSLA"])

        assert summary["symbols_analyzed"] == 2
        assert summary["biases"]["AAPL"] == "neutral"
        assert summary["biases"]["TSLA"] == "neutral"

        assert output_path.exists()
        file_data = json.loads(output_path.read_text())
        assert file_data["biases"]["AAPL"]["bias"] == "neutral"
        assert file_data["biases"]["TSLA"]["bias"] == "neutral"


# ---------------------------------------------------------------------------
# _log_to_db tests
# ---------------------------------------------------------------------------

class TestLogToDb:
    """Tests for SwingAdvisor._log_to_db."""

    def test_log_to_db_inserts_records(self, advisor: SwingAdvisor) -> None:
        """Should insert one swing_bias_log row per bias entry."""
        biases = [
            {"symbol": "AAPL", "bias": "bearish", "confidence": 70, "reason": "Weak outlook."},
            {"symbol": "TSLA", "bias": "bullish", "confidence": 80, "reason": "Strong sales."},
        ]

        advisor._log_to_db(biases, "raw response text here")

        week_start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = advisor._db.get_swing_biases_for_week(week_start)
        assert len(rows) == 2
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["bias"] == "bearish"
        assert rows[0]["confidence"] == 70
        assert rows[0]["claude_raw_response"] == "raw response text here"
        assert rows[1]["symbol"] == "TSLA"
        assert rows[1]["bias"] == "bullish"


# ---------------------------------------------------------------------------
# _parse_claude_response edge cases
# ---------------------------------------------------------------------------

class TestParseClaude:
    """Tests for edge cases in _parse_claude_response."""

    def test_clamps_confidence(self, advisor: SwingAdvisor) -> None:
        """Confidence values outside 0-100 should be clamped."""
        payloads = [{"symbol": "X"}]
        raw = json.dumps([
            {"symbol": "X", "bias": "bullish", "confidence": 150, "reason": "Over max."},
        ])
        result = advisor._parse_claude_response(raw, payloads)
        assert result[0]["confidence"] == 100

    def test_invalid_bias_defaults_neutral(self, advisor: SwingAdvisor) -> None:
        """Invalid bias value should default to neutral."""
        payloads = [{"symbol": "X"}]
        raw = json.dumps([
            {"symbol": "X", "bias": "VERY_BULLISH", "confidence": 80, "reason": "Invalid."},
        ])
        result = advisor._parse_claude_response(raw, payloads)
        assert result[0]["bias"] == "neutral"

    def test_dict_instead_of_list_fallback(self, advisor: SwingAdvisor) -> None:
        """A dict response instead of list should fall back to neutral."""
        payloads = [{"symbol": "X"}]
        raw = json.dumps({"symbol": "X", "bias": "bullish", "confidence": 70, "reason": "Oops."})
        result = advisor._parse_claude_response(raw, payloads)
        assert len(result) == 1
        assert result[0]["bias"] == "neutral"
