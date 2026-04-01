"""Shared state container for the dashboard."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from loguru import logger


class DashboardState:
    """Holds real-time trading state and manages WebSocket connections."""

    def __init__(self):
        self.latest_signals: dict[str, dict] = {}
        self.latest_prices: dict[str, dict] = {}
        self.account_info: dict = {}
        self.open_positions: list[dict] = []
        self.system_status: dict = {"started": datetime.now().isoformat(), "status": "running"}
        self._connections: set = set()
        self._lock = asyncio.Lock()

    async def add_connection(self, websocket) -> None:
        async with self._lock:
            self._connections.add(websocket)
        # Send initial state
        await self._send_to(websocket, "init", {
            "account": self.account_info,
            "positions": self.open_positions,
            "signals": self.latest_signals,
            "prices": self.latest_prices,
            "system": self.system_status,
        })

    async def remove_connection(self, websocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def _send_to(self, websocket, event_type: str, data: Any) -> None:
        """Send a single message to one WebSocket connection."""
        message = json.dumps({"type": event_type, "data": data, "ts": datetime.now().isoformat()})
        try:
            await websocket.send_text(message)
        except Exception:
            logger.debug("Failed to send initial state to WebSocket client")

    async def broadcast(self, event_type: str, data: dict) -> None:
        if not self._connections:
            return
        message = json.dumps({"type": event_type, "data": data, "ts": datetime.now().isoformat()})
        async with self._lock:
            stale = set()
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    stale.add(ws)
            self._connections -= stale

    def update_signal(self, symbol: str, signal_data: dict) -> None:
        self.latest_signals[symbol] = {**signal_data, "updated": datetime.now().isoformat()}
        asyncio.ensure_future(self.broadcast("signal", {"symbol": symbol, **signal_data}))

    def update_price(self, symbol: str, price_data: dict) -> None:
        self.latest_prices[symbol] = price_data
        asyncio.ensure_future(self.broadcast("price", {"symbol": symbol, **price_data}))

    def update_account(self, account_data: dict) -> None:
        self.account_info = account_data
        asyncio.ensure_future(self.broadcast("account", account_data))

    def update_positions(self, positions: list[dict]) -> None:
        self.open_positions = positions
        asyncio.ensure_future(self.broadcast("positions", {"positions": positions}))
