"""Tests for the real-time web dashboard."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.dashboard.state import DashboardState
from src.dashboard.app import create_app


# ---------------------------------------------------------------------------
# DashboardState tests
# ---------------------------------------------------------------------------

class TestDashboardState:
    """Tests for the DashboardState container."""

    def test_initial_state(self):
        state = DashboardState()
        assert state.latest_signals == {}
        assert state.latest_prices == {}
        assert state.account_info == {}
        assert state.open_positions == []
        assert state.system_status["status"] == "running"
        assert "started" in state.system_status

    def test_update_signal_stores_data(self):
        state = DashboardState()
        state.update_signal("EURUSD", {"signal": "buy", "confidence": 75})
        assert "EURUSD" in state.latest_signals
        assert state.latest_signals["EURUSD"]["signal"] == "buy"
        assert state.latest_signals["EURUSD"]["confidence"] == 75
        assert "updated" in state.latest_signals["EURUSD"]

    def test_update_price_stores_data(self):
        state = DashboardState()
        state.update_price("GBPUSD", {"bid": 1.2500, "ask": 1.2502, "spread": 2})
        assert "GBPUSD" in state.latest_prices
        assert state.latest_prices["GBPUSD"]["bid"] == 1.2500

    def test_update_account_stores_data(self):
        state = DashboardState()
        acct = {"balance": 10000, "equity": 10500, "free_margin": 9000}
        state.update_account(acct)
        assert state.account_info["balance"] == 10000
        assert state.account_info["equity"] == 10500

    def test_update_positions_stores_data(self):
        state = DashboardState()
        positions = [
            {"symbol": "EURUSD", "side": "buy", "volume": 0.1, "profit": 25.0},
            {"symbol": "GBPUSD", "side": "sell", "volume": 0.2, "profit": -10.0},
        ]
        state.update_positions(positions)
        assert len(state.open_positions) == 2
        assert state.open_positions[0]["symbol"] == "EURUSD"

    def test_multiple_signal_updates_overwrite(self):
        state = DashboardState()
        state.update_signal("EURUSD", {"signal": "buy", "confidence": 60})
        state.update_signal("EURUSD", {"signal": "sell", "confidence": 80})
        assert state.latest_signals["EURUSD"]["signal"] == "sell"
        assert state.latest_signals["EURUSD"]["confidence"] == 80


class TestDashboardStateBroadcast:
    """Tests for WebSocket broadcast behaviour."""

    @pytest.mark.asyncio
    async def test_add_and_remove_connection(self):
        state = DashboardState()
        ws = AsyncMock()
        ws.send_text = AsyncMock()

        await state.add_connection(ws)
        assert ws in state._connections

        await state.remove_connection(ws)
        assert ws not in state._connections

    @pytest.mark.asyncio
    async def test_add_connection_sends_init(self):
        state = DashboardState()
        state.account_info = {"balance": 5000}
        ws = AsyncMock()
        ws.send_text = AsyncMock()

        await state.add_connection(ws)

        ws.send_text.assert_called_once()
        msg = json.loads(ws.send_text.call_args[0][0])
        assert msg["type"] == "init"
        assert msg["data"]["account"]["balance"] == 5000

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self):
        state = DashboardState()
        ws1 = AsyncMock()
        ws1.send_text = AsyncMock()
        ws2 = AsyncMock()
        ws2.send_text = AsyncMock()

        await state.add_connection(ws1)
        await state.add_connection(ws2)

        await state.broadcast("account", {"balance": 9999})

        # Each ws got the init message + the broadcast
        assert ws1.send_text.call_count == 2
        assert ws2.send_text.call_count == 2

        last_msg1 = json.loads(ws1.send_text.call_args_list[-1][0][0])
        assert last_msg1["type"] == "account"
        assert last_msg1["data"]["balance"] == 9999

    @pytest.mark.asyncio
    async def test_broadcast_removes_stale_connections(self):
        state = DashboardState()
        good_ws = AsyncMock()
        good_ws.send_text = AsyncMock()
        bad_ws = AsyncMock()
        bad_ws.send_text = AsyncMock(side_effect=Exception("closed"))

        # Manually add to bypass the init send (which would fail for bad_ws)
        async with state._lock:
            state._connections.add(good_ws)
            state._connections.add(bad_ws)

        await state.broadcast("test", {"x": 1})

        assert good_ws in state._connections
        assert bad_ws not in state._connections

    @pytest.mark.asyncio
    async def test_broadcast_no_connections(self):
        state = DashboardState()
        # Should not raise
        await state.broadcast("test", {"x": 1})


# ---------------------------------------------------------------------------
# FastAPI app tests
# ---------------------------------------------------------------------------

class TestCreateApp:
    """Tests for the FastAPI application factory."""

    def test_create_app_returns_fastapi(self):
        from fastapi import FastAPI
        state = DashboardState()
        app = create_app(state)
        assert isinstance(app, FastAPI)
        assert app.title == "Phoenyx Trading Dashboard"

    def test_create_app_has_routes(self):
        state = DashboardState()
        app = create_app(state)
        route_paths = [r.path for r in app.routes]
        assert "/" in route_paths
        assert "/api/account" in route_paths
        assert "/api/positions" in route_paths
        assert "/api/signals" in route_paths
        assert "/api/prices" in route_paths
        assert "/api/status" in route_paths
        assert "/ws" in route_paths


class TestRESTEndpoints:
    """Test REST API endpoints using the FastAPI test client."""

    @pytest.fixture
    def state(self):
        s = DashboardState()
        s.account_info = {
            "balance": 10000, "equity": 10500,
            "free_margin": 9000, "currency": "USD",
        }
        s.open_positions = [
            {"symbol": "EURUSD", "side": "buy", "volume": 0.1, "profit": 50},
        ]
        s.latest_signals = {
            "EURUSD": {"signal": "buy", "confidence": 75, "updated": "2026-01-01T00:00:00"},
        }
        s.latest_prices = {
            "EURUSD": {"bid": 1.1000, "ask": 1.1002, "spread": 2},
        }
        return s

    @pytest.fixture
    def client(self, state):
        from starlette.testclient import TestClient
        app = create_app(state)
        return TestClient(app)

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Phoenyx Trading Dashboard" in resp.text

    def test_get_account(self, client, state):
        resp = client.get("/api/account")
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] == 10000
        assert data["equity"] == 10500

    def test_get_positions(self, client, state):
        resp = client.get("/api/positions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "EURUSD"

    def test_get_signals(self, client, state):
        resp = client.get("/api/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert "EURUSD" in data
        assert data["EURUSD"]["confidence"] == 75

    def test_get_prices(self, client, state):
        resp = client.get("/api/prices")
        assert resp.status_code == 200
        data = resp.json()
        assert data["EURUSD"]["bid"] == 1.1000

    def test_get_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "started" in data
