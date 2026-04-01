"""Notification dispatcher that fans out events to configured channels."""
from __future__ import annotations

import asyncio
import os
from typing import Sequence

from loguru import logger

from src.notifications.base import Notifier, NotificationEvent


class NotificationManager:
    """Dispatches NotificationEvents to one or more Notifier instances.

    Optionally filters events per-notifier so each channel only receives
    the event types it is subscribed to.
    """

    def __init__(
        self,
        notifiers: Sequence[Notifier] | None = None,
        event_filter: dict[Notifier, set[str]] | None = None,
    ) -> None:
        self._notifiers: list[Notifier] = list(notifiers or [])
        self._event_filter: dict[Notifier, set[str]] = dict(event_filter or {})

    def add_notifier(
        self, notifier: Notifier, events: set[str] | None = None
    ) -> None:
        """Register a notifier, optionally restricting it to certain event types."""
        self._notifiers.append(notifier)
        if events is not None:
            self._event_filter[notifier] = events

    async def notify(self, event: NotificationEvent) -> None:
        """Dispatch *event* to all registered notifiers (fire-and-forget).

        Each notifier is wrapped in its own try/except so a single failure
        does not prevent the others from receiving the event.
        """
        for notifier in self._notifiers:
            allowed = self._event_filter.get(notifier)
            if allowed is not None and event.event_type not in allowed:
                continue
            try:
                await notifier.send(event)
            except Exception as exc:
                logger.error(
                    f"Notifier {notifier.__class__.__name__} failed: {exc}"
                )

    @classmethod
    def from_config(cls, config: dict) -> "NotificationManager":
        """Build a NotificationManager from the ``notifications`` config block.

        Secrets are read from environment variables:
          - TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
          - DISCORD_WEBHOOK_URL
        """
        notif_cfg = config.get("notifications", {})
        mgr = cls()

        if not notif_cfg.get("enabled", False):
            return mgr

        # --- Telegram ---
        tg_cfg = notif_cfg.get("telegram", {})
        if tg_cfg.get("enabled", False):
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            if bot_token and chat_id:
                from src.notifications.telegram import TelegramNotifier

                notifier = TelegramNotifier(bot_token, chat_id)
                events = set(tg_cfg.get("events", []))
                mgr.add_notifier(notifier, events if events else None)
                logger.info("Telegram notifications enabled")
            else:
                logger.warning(
                    "Telegram enabled in config but TELEGRAM_BOT_TOKEN / "
                    "TELEGRAM_CHAT_ID env vars not set"
                )

        # --- Discord ---
        dc_cfg = notif_cfg.get("discord", {})
        if dc_cfg.get("enabled", False):
            webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
            if webhook_url:
                from src.notifications.discord import DiscordNotifier

                notifier = DiscordNotifier(webhook_url)
                events = set(dc_cfg.get("events", []))
                mgr.add_notifier(notifier, events if events else None)
                logger.info("Discord notifications enabled")
            else:
                logger.warning(
                    "Discord enabled in config but DISCORD_WEBHOOK_URL env var not set"
                )

        return mgr
