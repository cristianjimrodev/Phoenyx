"""Telegram notification channel."""
from __future__ import annotations

import time
from collections import deque

import aiohttp
from loguru import logger

from src.notifications.base import Notifier, NotificationEvent


class TelegramNotifier(Notifier):
    """Sends notifications via Telegram Bot API.

    Rate-limited to 20 messages per minute.
    """

    MAX_PER_MINUTE = 20

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._timestamps: deque[float] = deque()

    def _is_rate_limited(self) -> bool:
        """Return True if sending another message would exceed the rate limit."""
        now = time.monotonic()
        # Discard timestamps older than 60 seconds
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()
        return len(self._timestamps) >= self.MAX_PER_MINUTE

    @staticmethod
    def _format_message(event: NotificationEvent) -> str:
        """Format the event as a Markdown message for Telegram."""
        lines: list[str] = []

        type_emoji = {
            "trade_opened": "NEW TRADE",
            "trade_closed": "TRADE CLOSED",
            "risk_blocked": "RISK BLOCK",
            "signal": "SIGNAL",
            "error": "ERROR",
            "system": "SYSTEM",
        }
        header = type_emoji.get(event.event_type, event.event_type.upper())
        lines.append(f"*[{header}] {event.symbol}*")
        lines.append(event.message)

        details = event.details
        if "side" in details:
            lines.append(f"Side: {details['side']}")
        if "volume" in details:
            lines.append(f"Volume: {details['volume']}")
        if "entry_price" in details:
            lines.append(f"Entry: {details['entry_price']}")
        if "sl" in details:
            lines.append(f"SL: {details['sl']}")
        if "tp" in details:
            lines.append(f"TP: {details['tp']}")
        if "confidence" in details:
            lines.append(f"Confidence: {details['confidence']}%")
        if "pnl" in details:
            pnl = details["pnl"]
            sign = "+" if pnl >= 0 else ""
            lines.append(f"PnL: {sign}{pnl}")
        if "reason" in details:
            lines.append(f"Reason: {details['reason']}")

        return "\n".join(lines)

    async def send(self, event: NotificationEvent) -> bool:
        """Send a Telegram message. Returns True on success, False otherwise."""
        try:
            if self._is_rate_limited():
                logger.warning("Telegram rate limit reached, skipping notification")
                return False

            text = self._format_message(event)
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self._url, json=payload) as resp:
                    if resp.status == 200:
                        self._timestamps.append(time.monotonic())
                        return True
                    body = await resp.text()
                    logger.error(
                        f"Telegram API error {resp.status}: {body}"
                    )
                    return False

        except Exception as exc:
            logger.error(f"Telegram send failed: {exc}")
            return False
