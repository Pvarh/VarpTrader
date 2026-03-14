"""Weekly swing bias advisor using Claude API.

Runs every Monday at 06:00. For each ticker in the watchlist, fetches news,
fundamentals, and recent price action, then asks Claude for a directional bias.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import anthropic
import yfinance as yf
import httpx
from loguru import logger

from journal.db import TradeDatabase
from journal.models import SwingBias

WEEKLY_BIAS_PATH = Path("weekly_bias.json")


class SwingAdvisor:
    """Generates weekly directional bias per ticker using Claude."""

    SYSTEM_PROMPT = (
        "You are a swing trade analyst reviewing a portfolio watchlist. "
        "For each ticker provided, analyze the news, fundamentals, and "
        "recent price action. Output ONLY a valid JSON array with objects "
        "containing: symbol, bias (bullish/bearish/neutral), "
        "confidence (0-100), reason (max 20 words). "
        "Be conservative — only set bullish/bearish when confidence >= 60. "
        "Default to neutral when data is ambiguous."
    )

    def __init__(
        self,
        anthropic_api_key: str,
        news_api_key: str,
        db: TradeDatabase,
        model: str = "claude-sonnet-4-6",
        min_confidence: int = 60,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=anthropic_api_key)
        self._news_api_key = news_api_key
        self._db = db
        self._model = model
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, symbols: list[str]) -> dict:
        """Run the full swing advisor pipeline for all symbols.

        1. Fetch news headlines for each symbol (NewsAPI)
        2. Fetch fundamentals via yfinance .info (P/E, market cap, sector)
        3. Fetch 5-day OHLCV summary (% change, avg volume vs 20-day avg)
        4. Build structured JSON payload
        5. Send to Claude in a single API call
        6. Parse response
        7. Write to weekly_bias.json (overwrite)
        8. Log each bias to swing_bias_log table
        9. Return summary dict
        """
        logger.info("swing_advisor_run_started | symbols={symbols}", symbols=symbols)

        payloads: list[dict] = []
        for symbol in symbols:
            news = self._fetch_news(symbol)
            fundamentals = self._fetch_fundamentals(symbol)
            price_summary = self._fetch_price_summary(symbol)
            payload = self._build_payload(symbol, news, fundamentals, price_summary)
            payloads.append(payload)

        biases, raw_response = self._query_claude(payloads)

        self._write_weekly_bias(biases)
        self._log_to_db(biases, raw_response)

        summary = {
            "symbols_analyzed": len(symbols),
            "biases": {b["symbol"]: b["bias"] for b in biases},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("swing_advisor_run_complete | summary={summary}", summary=summary)
        return summary

    # ------------------------------------------------------------------
    # Data fetching helpers
    # ------------------------------------------------------------------

    def _fetch_news(self, symbol: str) -> list[str]:
        """Fetch last 7 days of news headlines from NewsAPI for a symbol.

        Returns list of headline strings. Returns empty list on failure.
        """
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": seven_days_ago,
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": self._news_api_key,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            headlines = [a["title"] for a in articles if a.get("title")]
            logger.debug(
                "news_fetched | symbol={symbol} count={count}",
                symbol=symbol, count=len(headlines),
            )
            return headlines
        except Exception as exc:
            logger.warning(
                "news_fetch_failed | symbol={symbol} error={error}",
                symbol=symbol, error=str(exc),
            )
            return []

    def _fetch_fundamentals(self, symbol: str) -> dict:
        """Fetch basic fundamentals via yfinance.

        Returns dict with pe_ratio, market_cap, sector, industry.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            fundamentals = {
                "pe_ratio": info.get("trailingPE"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
            }
            logger.debug(
                "fundamentals_fetched | symbol={symbol}",
                symbol=symbol,
            )
            return fundamentals
        except Exception as exc:
            logger.warning(
                "fundamentals_fetch_failed | symbol={symbol} error={error}",
                symbol=symbol, error=str(exc),
            )
            return {
                "pe_ratio": None,
                "market_cap": None,
                "sector": "N/A",
                "industry": "N/A",
            }

    def _fetch_price_summary(self, symbol: str) -> dict:
        """Fetch 5-day price summary.

        Returns dict with pct_change_5d, avg_volume_5d, avg_volume_20d, volume_ratio.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist_20 = ticker.history(period="1mo")
            if hist_20.empty or len(hist_20) < 5:
                logger.warning(
                    "price_summary_insufficient_data | symbol={symbol}",
                    symbol=symbol,
                )
                return {
                    "pct_change_5d": None,
                    "avg_volume_5d": None,
                    "avg_volume_20d": None,
                    "volume_ratio": None,
                }

            last_5 = hist_20.tail(5)
            last_20 = hist_20.tail(20)

            close_5d_ago = last_5["Close"].iloc[0]
            close_latest = last_5["Close"].iloc[-1]
            pct_change_5d = round(((close_latest - close_5d_ago) / close_5d_ago) * 100, 2)

            avg_volume_5d = int(last_5["Volume"].mean())
            avg_volume_20d = int(last_20["Volume"].mean())
            volume_ratio = round(avg_volume_5d / avg_volume_20d, 2) if avg_volume_20d > 0 else None

            summary = {
                "pct_change_5d": pct_change_5d,
                "avg_volume_5d": avg_volume_5d,
                "avg_volume_20d": avg_volume_20d,
                "volume_ratio": volume_ratio,
            }
            logger.debug(
                "price_summary_fetched | symbol={symbol} pct_change_5d={pct}",
                symbol=symbol, pct=pct_change_5d,
            )
            return summary
        except Exception as exc:
            logger.warning(
                "price_summary_fetch_failed | symbol={symbol} error={error}",
                symbol=symbol, error=str(exc),
            )
            return {
                "pct_change_5d": None,
                "avg_volume_5d": None,
                "avg_volume_20d": None,
                "volume_ratio": None,
            }

    # ------------------------------------------------------------------
    # Payload construction and Claude interaction
    # ------------------------------------------------------------------

    def _build_payload(
        self, symbol: str, news: list[str], fundamentals: dict, price_summary: dict
    ) -> dict:
        """Build the structured JSON payload for one symbol."""
        return {
            "symbol": symbol,
            "news_headlines": news,
            "fundamentals": fundamentals,
            "price_summary": price_summary,
        }

    def _query_claude(self, payloads: list[dict]) -> tuple[list[dict], str]:
        """Send all symbol payloads to Claude in a single API call.

        Parse response as JSON array. Return (list of bias dicts, raw_response_text).
        On failure, returns neutral biases for every symbol.
        """
        user_prompt = (
            "Analyze the following watchlist data and provide your swing bias "
            "for each ticker.\n\n"
            f"```json\n{json.dumps(payloads, indent=2, default=str)}\n```"
        )

        raw_text = ""
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if not message.content:
                logger.error("swing_advisor_empty_response")
                return self._neutral_fallback(payloads), ""

            raw_text = message.content[0].text
            logger.debug("swing_advisor_raw_response | text={text}", text=raw_text[:500])

            biases = self._parse_claude_response(raw_text, payloads)
            return biases, raw_text

        except anthropic.APIError as exc:
            logger.error(
                "swing_advisor_api_error | error={error}",
                error=str(exc),
            )
            return self._neutral_fallback(payloads), raw_text

    def _parse_claude_response(self, raw_text: str, payloads: list[dict]) -> list[dict]:
        """Parse Claude's raw text response into a list of bias dicts.

        Handles markdown code fences. Falls back to neutral on parse failure.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            start = 1
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[start:end]).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.error(
                "swing_advisor_parse_failed | raw_text={raw}",
                raw=raw_text[:500],
            )
            return self._neutral_fallback(payloads)

        if not isinstance(parsed, list):
            logger.error("swing_advisor_unexpected_type | type={t}", t=type(parsed).__name__)
            return self._neutral_fallback(payloads)

        # Validate each entry
        valid_biases = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            symbol = entry.get("symbol", "")
            bias = entry.get("bias", "neutral").lower()
            confidence = entry.get("confidence", 0)
            reason = entry.get("reason", "")

            if bias not in ("bullish", "bearish", "neutral"):
                bias = "neutral"
            if not isinstance(confidence, (int, float)):
                confidence = 0
            confidence = max(0, min(100, int(confidence)))

            valid_biases.append({
                "symbol": symbol,
                "bias": bias,
                "confidence": confidence,
                "reason": reason,
            })

        return valid_biases

    @staticmethod
    def _neutral_fallback(payloads: list[dict]) -> list[dict]:
        """Generate neutral biases for all symbols when Claude call fails."""
        return [
            {
                "symbol": p["symbol"],
                "bias": "neutral",
                "confidence": 0,
                "reason": "Unable to determine bias — defaulting to neutral.",
            }
            for p in payloads
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_weekly_bias(self, biases: list[dict]) -> None:
        """Write biases to weekly_bias.json atomically (write tmp, rename)."""
        now = datetime.now(timezone.utc)
        data = {
            "week_start": now.strftime("%Y-%m-%d"),
            "generated_at": now.isoformat(),
            "biases": {b["symbol"]: b for b in biases},
        }

        target = WEEKLY_BIAS_PATH
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent), suffix=".tmp", prefix="weekly_bias_"
            )
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename (works on POSIX; on Windows this overwrites)
            tmp = Path(tmp_path)
            tmp.replace(target)
            logger.info("weekly_bias_written | path={path}", path=str(target))
        except Exception as exc:
            logger.error(
                "weekly_bias_write_failed | error={error}",
                error=str(exc),
            )
            # Clean up temp file if rename failed
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _log_to_db(self, biases: list[dict], raw_response: str) -> None:
        """Log each bias to the swing_bias_log table via TradeDatabase."""
        week_start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for b in biases:
            swing_bias = SwingBias(
                symbol=b["symbol"],
                bias=b["bias"],
                confidence=b["confidence"],
                reason=b.get("reason", ""),
                week_start=week_start,
                claude_raw_response=raw_response,
            )
            try:
                self._db.insert_swing_bias(swing_bias)
            except Exception as exc:
                logger.error(
                    "swing_bias_db_insert_failed | symbol={symbol} error={error}",
                    symbol=b["symbol"], error=str(exc),
                )

    # ------------------------------------------------------------------
    # Static helpers for trade-gating
    # ------------------------------------------------------------------

    @staticmethod
    def load_weekly_bias(bias_path: Optional[Path] = None) -> dict:
        """Load and return the current weekly_bias.json.

        Returns empty biases dict if file missing or >8 days old.

        Args:
            bias_path: Optional override path for testing. Defaults to
                       WEEKLY_BIAS_PATH.
        """
        path = bias_path or WEEKLY_BIAS_PATH
        try:
            if not path.exists():
                logger.warning("weekly_bias_file_missing | path={path}", path=str(path))
                return {"week_start": "", "generated_at": "", "biases": {}}

            with open(path, "r") as f:
                data = json.load(f)

            generated_at = data.get("generated_at", "")
            if not generated_at:
                logger.warning("weekly_bias_no_timestamp")
                return {"week_start": "", "generated_at": "", "biases": {}}

            gen_dt = datetime.fromisoformat(generated_at)
            # Make timezone-aware if naive
            if gen_dt.tzinfo is None:
                gen_dt = gen_dt.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - gen_dt
            if age > timedelta(days=8):
                logger.warning(
                    "weekly_bias_stale | age_days={days}",
                    days=age.days,
                )
                return {"week_start": "", "generated_at": "", "biases": {}}

            return data

        except Exception as exc:
            logger.error(
                "weekly_bias_load_failed | error={error}",
                error=str(exc),
            )
            return {"week_start": "", "generated_at": "", "biases": {}}

    @staticmethod
    def should_block_trade(
        symbol: str,
        direction: str,
        min_confidence: int = 60,
        bias_path: Optional[Path] = None,
    ) -> bool:
        """Check if a trade should be blocked based on swing bias.

        Rules:
        - If bias='bearish' with confidence >= min_confidence AND direction='long': BLOCK
        - If bias='bullish' with confidence >= min_confidence AND direction='short': BLOCK
        - If bias='neutral' or missing: ALLOW
        - If weekly_bias.json is missing or >8 days old: log warning, ALLOW

        Args:
            symbol: Ticker symbol to check.
            direction: Trade direction, 'long' or 'short'.
            min_confidence: Minimum confidence threshold to block.
            bias_path: Optional override path for testing.

        Returns:
            True if the trade should be blocked, False otherwise.
        """
        data = SwingAdvisor.load_weekly_bias(bias_path)
        biases = data.get("biases", {})

        if not biases:
            logger.debug(
                "swing_bias_no_data | symbol={symbol} — allowing trade",
                symbol=symbol,
            )
            return False

        entry = biases.get(symbol)
        if entry is None:
            logger.debug(
                "swing_bias_no_entry | symbol={symbol} — allowing trade",
                symbol=symbol,
            )
            return False

        bias = entry.get("bias", "neutral").lower()
        confidence = entry.get("confidence", 0)

        if bias == "neutral":
            return False

        if confidence < min_confidence:
            logger.debug(
                "swing_bias_low_confidence | symbol={symbol} bias={bias} "
                "confidence={conf} threshold={thr} — allowing trade",
                symbol=symbol, bias=bias, conf=confidence, thr=min_confidence,
            )
            return False

        # Bearish bias blocks long trades
        if bias == "bearish" and direction.lower() == "long":
            logger.info(
                "swing_bias_blocking_trade | symbol={symbol} bias=bearish "
                "confidence={conf} direction=long",
                symbol=symbol, conf=confidence,
            )
            return True

        # Bullish bias blocks short trades
        if bias == "bullish" and direction.lower() == "short":
            logger.info(
                "swing_bias_blocking_trade | symbol={symbol} bias=bullish "
                "confidence={conf} direction=short",
                symbol=symbol, conf=confidence,
            )
            return True

        return False
