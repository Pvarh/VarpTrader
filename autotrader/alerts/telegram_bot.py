"""Telegram Bot API alerting module.

Sends trade alerts, daily reports, and kill-switch notifications to a
configured Telegram chat via the Bot HTTP API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger


class TelegramAlert:
    """Telegram Bot API wrapper for sending trade alerts and reports.

    All messages are sent synchronously via ``httpx`` to the Telegram
    ``sendMessage`` endpoint.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token (from BotFather).
    chat_id:
        Target chat, group, or channel ID.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token: str = bot_token
        self._chat_id: str = chat_id
        self._base_url: str = f"https://api.telegram.org/bot{bot_token}"

        logger.info("telegram_alert_initialised | chat_id={chat_id}", chat_id=chat_id)

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------
    def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
    ) -> bool:
        """Send a text message to the configured chat.

        Parameters
        ----------
        text:
            Message body.  Supports Markdown or HTML depending on
            *parse_mode*.
        parse_mode:
            Telegram parse mode (``"Markdown"``, ``"MarkdownV2"``, or
            ``"HTML"``).

        Returns
        -------
        bool
            ``True`` if the message was delivered successfully.
        """
        url = f"{self._base_url}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

            data: dict[str, Any] = response.json()
            if not data.get("ok"):
                logger.warning(
                    "telegram_send_not_ok | chat_id={chat_id} parse_mode={parse_mode} description={description}",
                    chat_id=self._chat_id,
                    parse_mode=parse_mode,
                    description=data.get("description"),
                )
                return False

            logger.info(
                "telegram_message_sent | chat_id={chat_id} parse_mode={parse_mode}",
                chat_id=self._chat_id,
                parse_mode=parse_mode,
            )
            return True

        except httpx.HTTPStatusError:
            logger.exception(
                "telegram_send_http_error | chat_id={chat_id} parse_mode={parse_mode}",
                chat_id=self._chat_id,
                parse_mode=parse_mode,
            )
            return False
        except httpx.RequestError:
            logger.exception(
                "telegram_send_request_error | chat_id={chat_id} parse_mode={parse_mode}",
                chat_id=self._chat_id,
                parse_mode=parse_mode,
            )
            return False
        except Exception:
            logger.exception(
                "telegram_send_error | chat_id={chat_id} parse_mode={parse_mode}",
                chat_id=self._chat_id,
                parse_mode=parse_mode,
            )
            return False

    # ------------------------------------------------------------------
    # Trade alert
    # ------------------------------------------------------------------
    def send_trade_alert(self, trade_dict: dict[str, Any]) -> bool:
        """Format and send a trade entry/exit alert.

        Parameters
        ----------
        trade_dict:
            Dictionary describing the trade.  Expected keys:

            - ``action`` (str): ``"ENTRY"`` or ``"EXIT"``.
            - ``symbol`` (str): Instrument symbol.
            - ``direction`` (str): ``"LONG"`` or ``"SHORT"``.
            - ``price`` (float): Execution price.
            - ``quantity`` (int | float): Size.
            - ``strategy`` (str, optional): Strategy name.
            - ``pnl`` (float, optional): Realised P&L (exits only).

        Returns
        -------
        bool
            ``True`` if the alert was delivered successfully.
        """
        action: str = trade_dict.get("action", "TRADE")
        symbol: str = trade_dict.get("symbol", "???")
        direction: str = trade_dict.get("direction", "N/A")
        price: float = trade_dict.get("price", 0.0)
        quantity = trade_dict.get("quantity", 0)
        strategy: str = trade_dict.get("strategy", "N/A")
        pnl = trade_dict.get("pnl")

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines: list[str] = [
            f"*{action}* | {symbol}",
            f"Direction: {direction}",
            f"Price: {price}",
            f"Qty: {quantity}",
            f"Strategy: {strategy}",
        ]
        if pnl is not None:
            lines.append(f"P&L: {pnl:+.2f}")
        lines.append(f"Time: {timestamp}")

        text = "\n".join(lines)
        return self.send_message(text)

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------
    def send_daily_report(self, report_markdown: str) -> bool:
        """Send the daily analysis report.

        Parameters
        ----------
        report_markdown:
            Pre-formatted Markdown string containing the full daily
            performance report.

        Returns
        -------
        bool
            ``True`` if the report was delivered successfully.
        """
        header = "*Daily Trading Report*\n\n"
        return self.send_message(header + report_markdown)

    # ------------------------------------------------------------------
    # Kill switch alert
    # ------------------------------------------------------------------
    def send_kill_switch_alert(self) -> bool:
        """Alert that the kill switch has been triggered.

        Sends a high-urgency message indicating that all trading has
        been halted due to the daily loss limit being breached.

        Returns
        -------
        bool
            ``True`` if the alert was delivered successfully.
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        text = (
            "*KILL SWITCH ACTIVATED*\n\n"
            "Daily loss limit breached. All trading has been halted.\n"
            f"Time: {timestamp}\n\n"
            "Manual review required before resuming operations."
        )
        return self.send_message(text)
