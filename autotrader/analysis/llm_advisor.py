"""Claude-backed nightly tuning advisor helpers.

This module provides prompt construction and response parsing for the
nightly analysis flow. Direct API usage is optional; on the VPS we route
the Claude call through the host login session instead.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


class LLMAdvisor:
    """Prompt/parse helper for nightly config recommendations."""

    SYSTEM_PROMPT: str = (
        "You are a quantitative trading analyst. Review this trading performance "
        "data and suggest specific, numeric config parameter adjustments. "
        "Output ONLY valid JSON matching the requested flat schema. "
        "Do not suggest changes outside the allowed bounds. "
        "Be conservative and only suggest changes backed by the data."
    )

    ALLOWED_PARAMS: set[str] = {
        "stop_loss_pct",
        "position_size_pct",
        "rsi_oversold",
        "rsi_overbought",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-6",
    ) -> None:
        """Initialize the advisor metadata.

        ``api_key`` is retained for compatibility with the existing app
        wiring, but the host-login flow does not use it.
        """
        _ = api_key
        self.model = model
        logger.info(
            "llm_advisor_initialized | model={model}",
            model=self.model,
        )

    @staticmethod
    def _round_param(name: str, value: float) -> float:
        if name in {"rsi_oversold", "rsi_overbought"}:
            return round(value)
        return round(value, 4)

    def _fallback_recommendations(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Local compatibility fallback when the host Claude path is unavailable."""
        total_trades = int(performance_data.get("total_trades", 0) or 0)
        total_wins = int(performance_data.get("total_wins", 0) or 0)
        total_pnl = float(performance_data.get("total_pnl", 0.0) or 0.0)
        overall_win_rate = (total_wins / total_trades) if total_trades else 0.0
        strategy_win_rates = performance_data.get("strategy_win_rates", {}) or {}
        avg_pnl_per_strategy = performance_data.get("avg_pnl_per_strategy", {}) or {}

        recommendations: dict[str, Any] = {}
        if total_trades >= 30 and total_pnl < 0 and overall_win_rate <= 0.35:
            recommendations["position_size_pct"] = 0.005
        if total_trades >= 30 and total_pnl < 0 and overall_win_rate <= 0.25:
            recommendations["stop_loss_pct"] = 0.01

        rsi_wr = strategy_win_rates.get("rsi_momentum")
        rsi_avg_pnl = avg_pnl_per_strategy.get("rsi_momentum")
        if (
            rsi_wr is not None
            and rsi_avg_pnl is not None
            and rsi_avg_pnl < 0
            and rsi_wr <= max(0.10, overall_win_rate - 0.10)
        ):
            recommendations["rsi_oversold"] = 25
            recommendations["rsi_overbought"] = 75

        return {
            key: self._round_param(key, float(value))
            for key, value in recommendations.items()
        }

    @classmethod
    def build_user_prompt(
        cls,
        performance_data: dict[str, Any],
        current_values: dict[str, Any] | None = None,
    ) -> str:
        """Build the Claude user prompt for nightly analysis."""
        bounds_info = {
            "stop_loss_pct": {"min": 0.005, "max": 0.05},
            "position_size_pct": {"min": 0.005, "max": 0.03},
            "rsi_oversold": {"min": 20, "max": 40},
            "rsi_overbought": {"min": 60, "max": 80},
        }
        current_values = current_values or {}
        return (
            "Here is the recent trading performance data:\n\n"
            f"```json\n{json.dumps(performance_data, indent=2, default=str)}\n```\n\n"
            "Current tunable values:\n\n"
            f"```json\n{json.dumps(current_values, indent=2, default=str)}\n```\n\n"
            "Allowed parameter bounds:\n\n"
            f"```json\n{json.dumps(bounds_info, indent=2)}\n```\n\n"
            "Suggest parameter changes as a flat JSON object with parameter names as keys and new numeric values. "
            "Only include parameters you want to change. "
            "Output ONLY the JSON object, no additional text."
        )

    @classmethod
    def build_messages(
        cls,
        performance_data: dict[str, Any],
        current_values: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Return the system and user prompt pair for Claude."""
        return cls.SYSTEM_PROMPT, cls.build_user_prompt(
            performance_data,
            current_values=current_values,
        )

    @classmethod
    def parse_response(cls, raw_text: str) -> dict[str, Any]:
        """Extract a JSON dict of recommendations from a Claude response."""
        text = raw_text.strip()
        if not text:
            logger.error("llm_response_empty")
            return {}

        if text.startswith("```"):
            lines = text.splitlines()
            start = 1
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[start:end]).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.error(
                "llm_response_parse_failed | raw_text={raw_text}",
                raw_text=raw_text[:500],
            )
            return {}

        if not isinstance(parsed, dict):
            logger.error(
                "llm_response_unexpected_type | type={type_name}",
                type_name=type(parsed).__name__,
            )
            return {}

        filtered: dict[str, Any] = {}
        for key, value in parsed.items():
            if key not in cls.ALLOWED_PARAMS:
                logger.warning(
                    "llm_recommendation_unknown_param | param={param} value={value}",
                    param=key,
                    value=value,
                )
                continue
            try:
                filtered[key] = float(value)
            except (TypeError, ValueError):
                logger.warning(
                    "llm_recommendation_non_numeric | param={param} value={value}",
                    param=key,
                    value=value,
                )

        logger.info(
            "llm_recommendations_received | recommendations={recommendations}",
            recommendations=filtered,
        )
        return filtered

    def get_recommendations(
        self,
        performance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Compatibility fallback for non-host nightly paths."""
        recommendations = self._fallback_recommendations(performance_data)
        logger.info(
            "llm_recommendations_received | mode=fallback recommendations={recommendations}",
            recommendations=recommendations,
        )
        return recommendations
