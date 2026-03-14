"""LLM-based trading advisor using the Anthropic Claude API.

Sends performance analytics to Claude and parses structured JSON
recommendations for config parameter adjustments.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
from loguru import logger


class LLMAdvisor:
    """Request config-tuning recommendations from Claude based on trade data."""

    SYSTEM_PROMPT: str = (
        "You are a quantitative trading analyst. Review this trading performance "
        "data and suggest specific, numeric config parameter adjustments. "
        "Output ONLY valid JSON matching the config.json schema. "
        "Do not suggest changes outside the allowed bounds. "
        "Be conservative -- only suggest changes when win rate difference is > 10%."
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
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key.
            model: Claude model identifier to use for recommendations.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info("llm_advisor_initialized | model={model}", model=self.model)

    def _build_user_prompt(self, performance_data: dict[str, Any]) -> str:
        """Format the performance data into a user-message string.

        Args:
            performance_data: The full report dict produced by
                :pyclass:`~analysis.analyzer.PerformanceAnalyzer`.

        Returns:
            A human-readable JSON string summarising the data, accompanied
            by the allowed parameter bounds for the model's reference.
        """
        bounds_info = {
            "stop_loss_pct": {"min": 0.005, "max": 0.05},
            "position_size_pct": {"min": 0.005, "max": 0.03},
            "rsi_oversold": {"min": 20, "max": 40},
            "rsi_overbought": {"min": 60, "max": 80},
        }
        prompt = (
            "Here is the recent trading performance data:\n\n"
            f"```json\n{json.dumps(performance_data, indent=2, default=str)}\n```\n\n"
            "Allowed parameter bounds:\n\n"
            f"```json\n{json.dumps(bounds_info, indent=2)}\n```\n\n"
            "Based on this data, suggest parameter changes as a flat JSON "
            "object with parameter names as keys and new numeric values. "
            "Only include parameters you want to change. "
            "Example: {\"stop_loss_pct\": 0.02, \"rsi_oversold\": 30}\n"
            "Output ONLY the JSON object, no additional text."
        )
        return prompt

    def _parse_response(self, raw_text: str) -> dict[str, Any]:
        """Extract a JSON dict of recommendations from the model response.

        Handles responses that may include markdown code fences or
        extraneous text surrounding the JSON payload.

        Args:
            raw_text: Raw text from the Claude API response.

        Returns:
            Parsed dict of parameter recommendations.  Returns an empty
            dict when parsing fails or the response is not valid JSON.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first line (```json or ```) and last line (```)
            start = 1
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[start:end]).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.error("llm_response_parse_failed | raw_text={raw_text}", raw_text=raw_text[:500])
            return {}

        if not isinstance(parsed, dict):
            logger.error(
                "llm_response_unexpected_type | type={type_name}",
                type_name=type(parsed).__name__,
            )
            return {}

        # Filter to only allowed parameter names
        filtered: dict[str, Any] = {}
        for key, value in parsed.items():
            if key in self.ALLOWED_PARAMS:
                try:
                    filtered[key] = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "llm_recommendation_non_numeric | param={param} value={value}",
                        param=key,
                        value=value,
                    )
            else:
                logger.warning(
                    "llm_recommendation_unknown_param | param={param} value={value}",
                    param=key,
                    value=value,
                )

        return filtered

    def get_recommendations(
        self, performance_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Send performance stats to Claude and return suggested changes.

        Args:
            performance_data: Full report dict from
                :pyfunc:`PerformanceAnalyzer.build_full_report`.

        Returns:
            Dict of parameter names to suggested numeric values.
            Returns an empty dict on API errors or unparseable responses.
        """
        user_prompt = self._build_user_prompt(performance_data)

        logger.info(
            "llm_advisor_requesting_recommendations | model={model} data_keys={data_keys}",
            model=self.model,
            data_keys=list(performance_data.keys()),
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
        except anthropic.APIError:
            logger.exception("llm_api_call_failed")
            return {}

        # Extract text from the first content block
        if not message.content:
            logger.error("llm_response_empty")
            return {}

        raw_text = message.content[0].text
        logger.debug("llm_raw_response | text={text}", text=raw_text[:500])

        recommendations = self._parse_response(raw_text)

        logger.info(
            "llm_recommendations_received | recommendations={recommendations}",
            recommendations=recommendations,
        )
        return recommendations
