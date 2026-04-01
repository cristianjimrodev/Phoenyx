"""Tests for the notification system."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.notifications.base import NotificationEvent, Notifier
from src.notifications.telegram import TelegramNotifier
from src.notifications.discord import DiscordNotifier
from src.notifications.manager import NotificationManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: str = "trade_opened", symbol: str = "EURUSD", **details) -> NotificationEvent:
    return NotificationEvent(
        event_type=event_type,
        symbol=symbol,
        message="test message",
        details=details,
    )


def _mock_response(status: int = 200, body: str = "ok"):
    """Create a mock aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    return resp


def _mock_session(response):
    """Build a mock aiohttp.ClientSession that yields *response* on post()."""
    post_cm = AsyncMock()
    post_cm.__aenter__ = AsyncMock(return_value=response)
    post_cm.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.post = MagicMock(return_value=post_cm)

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return session_cm, session


# ---------------------------------------------------------------------------
# TelegramNotifier tests
# ---------------------------------------------------------------------------

class TestTelegramNotifier:

    @pytest.mark.asyncio
    async def test_send_success(self):
        notifier = TelegramNotifier(bot_token="TOKEN123", chat_id="CHAT456")
        event = _make_event(side="buy", volume=0.1, entry_price=1.1234)

        resp = _mock_response(200)
        session_cm, session = _mock_session(resp)

        with patch("src.notifications.telegram.aiohttp.ClientSession", return_value=session_cm):
            result = await notifier.send(event)

        assert result is True
        session.post.assert_called_once()
        call_args = session.post.call_args
        assert call_args[0][0] == "https://api.telegram.org/botTOKEN123/sendMessage"
        payload = call_args[1]["json"]
        assert payload["chat_id"] == "CHAT456"
        assert payload["parse_mode"] == "Markdown"
        assert "EURUSD" in payload["text"]

    @pytest.mark.asyncio
    async def test_send_api_error_returns_false(self):
        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        event = _make_event()

        resp = _mock_response(status=400, body="Bad Request")
        session_cm, session = _mock_session(resp)

        with patch("src.notifications.telegram.aiohttp.ClientSession", return_value=session_cm):
            result = await notifier.send(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_exception_returns_false(self):
        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        event = _make_event()

        with patch("src.notifications.telegram.aiohttp.ClientSession", side_effect=RuntimeError("boom")):
            result = await notifier.send(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        event = _make_event()

        resp = _mock_response(200)
        session_cm, session = _mock_session(resp)

        with patch("src.notifications.telegram.aiohttp.ClientSession", return_value=session_cm):
            # Fill up the rate limit bucket
            for _ in range(TelegramNotifier.MAX_PER_MINUTE):
                result = await notifier.send(event)
                assert result is True

            # Next message should be rate-limited
            result = await notifier.send(event)
            assert result is False

    @pytest.mark.asyncio
    async def test_format_message_includes_pnl(self):
        event = _make_event(event_type="trade_closed", pnl=42.5)
        text = TelegramNotifier._format_message(event)
        assert "+42.5" in text

    @pytest.mark.asyncio
    async def test_format_message_negative_pnl(self):
        event = _make_event(event_type="trade_closed", pnl=-15.0)
        text = TelegramNotifier._format_message(event)
        assert "-15.0" in text


# ---------------------------------------------------------------------------
# DiscordNotifier tests
# ---------------------------------------------------------------------------

class TestDiscordNotifier:

    @pytest.mark.asyncio
    async def test_send_success(self):
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/123/abc")
        event = _make_event(side="buy", volume=0.5)

        resp = _mock_response(204)
        session_cm, session = _mock_session(resp)

        with patch("src.notifications.discord.aiohttp.ClientSession", return_value=session_cm):
            result = await notifier.send(event)

        assert result is True
        session.post.assert_called_once()
        call_args = session.post.call_args
        assert call_args[0][0] == "https://discord.com/api/webhooks/123/abc"
        payload = call_args[1]["json"]
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        embed = payload["embeds"][0]
        assert "EURUSD" in embed["title"]

    @pytest.mark.asyncio
    async def test_send_webhook_error_returns_false(self):
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/x/y")
        event = _make_event()

        resp = _mock_response(status=429, body="Rate limited")
        session_cm, session = _mock_session(resp)

        with patch("src.notifications.discord.aiohttp.ClientSession", return_value=session_cm):
            result = await notifier.send(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_exception_returns_false(self):
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/x/y")
        event = _make_event()

        with patch("src.notifications.discord.aiohttp.ClientSession", side_effect=RuntimeError("fail")):
            result = await notifier.send(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_color_buy_is_green(self):
        event = _make_event(side="BUY")
        color = DiscordNotifier._pick_color(event)
        assert color == 0x2ECC71

    @pytest.mark.asyncio
    async def test_color_sell_is_red(self):
        event = _make_event(side="SELL")
        color = DiscordNotifier._pick_color(event)
        assert color == 0xE74C3C

    @pytest.mark.asyncio
    async def test_color_risk_blocked_is_yellow(self):
        event = _make_event(event_type="risk_blocked")
        color = DiscordNotifier._pick_color(event)
        assert color == 0xF1C40F

    @pytest.mark.asyncio
    async def test_color_positive_pnl_is_green(self):
        event = _make_event(event_type="trade_closed", pnl=10.0)
        color = DiscordNotifier._pick_color(event)
        assert color == 0x2ECC71

    @pytest.mark.asyncio
    async def test_color_negative_pnl_is_red(self):
        event = _make_event(event_type="trade_closed", pnl=-5.0)
        color = DiscordNotifier._pick_color(event)
        assert color == 0xE74C3C


# ---------------------------------------------------------------------------
# NotificationManager tests
# ---------------------------------------------------------------------------

class _FakeNotifier(Notifier):
    """Simple notifier for testing that records calls."""

    def __init__(self, should_fail: bool = False):
        self.events: list[NotificationEvent] = []
        self.should_fail = should_fail

    async def send(self, event: NotificationEvent) -> bool:
        if self.should_fail:
            raise RuntimeError("intentional failure")
        self.events.append(event)
        return True


class TestNotificationManager:

    @pytest.mark.asyncio
    async def test_dispatches_to_multiple_notifiers(self):
        n1 = _FakeNotifier()
        n2 = _FakeNotifier()
        mgr = NotificationManager(notifiers=[n1, n2])

        event = _make_event()
        await mgr.notify(event)

        assert len(n1.events) == 1
        assert len(n2.events) == 1
        assert n1.events[0] is event
        assert n2.events[0] is event

    @pytest.mark.asyncio
    async def test_handles_notifier_failure_gracefully(self):
        failing = _FakeNotifier(should_fail=True)
        healthy = _FakeNotifier()
        mgr = NotificationManager(notifiers=[failing, healthy])

        event = _make_event()
        # Should not raise
        await mgr.notify(event)

        # The healthy notifier should still have received the event
        assert len(healthy.events) == 1
        assert len(failing.events) == 0

    @pytest.mark.asyncio
    async def test_event_filtering_allows_matching_type(self):
        n = _FakeNotifier()
        mgr = NotificationManager()
        mgr.add_notifier(n, events={"trade_opened", "trade_closed"})

        await mgr.notify(_make_event(event_type="trade_opened"))
        assert len(n.events) == 1

    @pytest.mark.asyncio
    async def test_event_filtering_blocks_non_matching_type(self):
        n = _FakeNotifier()
        mgr = NotificationManager()
        mgr.add_notifier(n, events={"trade_opened", "trade_closed"})

        await mgr.notify(_make_event(event_type="error"))
        assert len(n.events) == 0

    @pytest.mark.asyncio
    async def test_no_filter_means_all_events(self):
        n = _FakeNotifier()
        mgr = NotificationManager()
        mgr.add_notifier(n, events=None)

        await mgr.notify(_make_event(event_type="error"))
        await mgr.notify(_make_event(event_type="trade_opened"))
        assert len(n.events) == 2

    @pytest.mark.asyncio
    async def test_from_config_disabled(self):
        cfg = {"notifications": {"enabled": False}}
        mgr = NotificationManager.from_config(cfg)
        assert len(mgr._notifiers) == 0

    @pytest.mark.asyncio
    async def test_from_config_telegram_enabled(self):
        cfg = {
            "notifications": {
                "enabled": True,
                "telegram": {
                    "enabled": True,
                    "events": ["trade_opened", "error"],
                },
            }
        }
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "cid"}):
            mgr = NotificationManager.from_config(cfg)

        assert len(mgr._notifiers) == 1
        assert isinstance(mgr._notifiers[0], TelegramNotifier)
        assert mgr._event_filter[mgr._notifiers[0]] == {"trade_opened", "error"}

    @pytest.mark.asyncio
    async def test_from_config_discord_enabled(self):
        cfg = {
            "notifications": {
                "enabled": True,
                "discord": {
                    "enabled": True,
                    "events": ["trade_closed"],
                },
            }
        }
        with patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://hooks.example.com"}):
            mgr = NotificationManager.from_config(cfg)

        assert len(mgr._notifiers) == 1
        assert isinstance(mgr._notifiers[0], DiscordNotifier)

    @pytest.mark.asyncio
    async def test_from_config_missing_env_vars(self):
        cfg = {
            "notifications": {
                "enabled": True,
                "telegram": {"enabled": True},
                "discord": {"enabled": True},
            }
        }
        with patch.dict("os.environ", {}, clear=True):
            mgr = NotificationManager.from_config(cfg)

        # No notifiers added because env vars are missing
        assert len(mgr._notifiers) == 0
