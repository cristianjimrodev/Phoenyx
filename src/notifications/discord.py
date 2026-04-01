"""Discord notification channel via webhooks."""
from __future__ import annotations

import aiohttp
from loguru import logger

from src.notifications.base import Notifier, NotificationEvent

# Embed colour palette (decimal RGB)
COLOR_GREEN = 0x2ECC71   # BUY / win
COLOR_RED = 0xE74C3C     # SELL / loss
COLOR_YELLOW = 0xF1C40F  # warning / risk_blocked


class DiscordNotifier(Notifier):
    """Sends rich-embed notifications to a Discord webhook."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    @staticmethod
    def _pick_color(event: NotificationEvent) -> int:
        """Choose embed colour based on event semantics."""
        et = event.event_type
        details = event.details

        if et == "risk_blocked" or et == "error":
            return COLOR_YELLOW if et == "risk_blocked" else COLOR_RED

        side = str(details.get("side", "")).upper()
        pnl = details.get("pnl")

        if pnl is not None:
            return COLOR_GREEN if pnl >= 0 else COLOR_RED

        if side == "BUY":
            return COLOR_GREEN
        if side == "SELL":
            return COLOR_RED

        return COLOR_YELLOW

    @staticmethod
    def _build_embed(event: NotificationEvent, color: int) -> dict:
        """Create a Discord embed dict from the event."""
        title_map = {
            "trade_opened": "New Trade Opened",
            "trade_closed": "Trade Closed",
            "risk_blocked": "Trade Blocked by Risk Manager",
            "signal": "Signal Detected",
            "error": "Error",
            "system": "System",
        }
        title = f"{title_map.get(event.event_type, event.event_type.upper())} - {event.symbol}"

        fields: list[dict] = []
        detail_labels = {
            "side": "Side",
            "volume": "Volume",
            "entry_price": "Entry Price",
            "sl": "Stop Loss",
            "tp": "Take Profit",
            "confidence": "Confidence",
            "pnl": "PnL",
            "reason": "Reason",
        }
        for key, label in detail_labels.items():
            if key in event.details:
                value = event.details[key]
                if key == "confidence":
                    value = f"{value}%"
                elif key == "pnl":
                    sign = "+" if value >= 0 else ""
                    value = f"{sign}{value}"
                fields.append({"name": label, "value": str(value), "inline": True})

        embed: dict = {
            "title": title,
            "description": event.message,
            "color": color,
            "timestamp": event.timestamp,
        }
        if fields:
            embed["fields"] = fields

        return embed

    async def send(self, event: NotificationEvent) -> bool:
        """Post a rich embed to the Discord webhook. Returns True on success."""
        try:
            color = self._pick_color(event)
            embed = self._build_embed(event, color)
            payload = {"embeds": [embed]}

            async with aiohttp.ClientSession() as session:
                async with session.post(self._webhook_url, json=payload) as resp:
                    if resp.status in (200, 204):
                        return True
                    body = await resp.text()
                    logger.error(
                        f"Discord webhook error {resp.status}: {body}"
                    )
                    return False

        except Exception as exc:
            logger.error(f"Discord send failed: {exc}")
            return False
