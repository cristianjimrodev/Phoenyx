"""FastAPI dashboard application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.dashboard.state import DashboardState

STATIC_DIR = Path(__file__).parent / "static"


def create_app(state: DashboardState) -> FastAPI:
    app = FastAPI(title="Phoenyx Trading Dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/account")
    async def get_account():
        return state.account_info

    @app.get("/api/positions")
    async def get_positions():
        return {"positions": state.open_positions}

    @app.get("/api/signals")
    async def get_signals():
        return state.latest_signals

    @app.get("/api/prices")
    async def get_prices():
        return state.latest_prices

    @app.get("/api/status")
    async def get_status():
        return state.system_status

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        await state.add_connection(websocket)
        try:
            while True:
                await websocket.receive_text()  # keep alive
        except WebSocketDisconnect:
            pass
        finally:
            await state.remove_connection(websocket)

    return app
