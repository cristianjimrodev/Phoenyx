"""FastAPI dashboard application."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.dashboard.state import DashboardState

STATIC_DIR = Path(__file__).parent / "static"


def _build_auth_dependency():
    """Return a dependency that enforces HTTP Basic auth if configured.

    Set DASHBOARD_USER and DASHBOARD_PASSWORD env vars to enable. If either is
    unset, auth is a no-op (useful for local dev).
    """
    user = os.getenv("DASHBOARD_USER")
    password = os.getenv("DASHBOARD_PASSWORD")
    if not user or not password:
        async def _noop():
            return None
        return _noop

    basic = HTTPBasic()

    def _check(creds: HTTPBasicCredentials = Depends(basic)):
        ok_user = secrets.compare_digest(creds.username, user)
        ok_pass = secrets.compare_digest(creds.password, password)
        if not (ok_user and ok_pass):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

    return _check


def create_app(state: DashboardState) -> FastAPI:
    app = FastAPI(title="Phoenyx Trading Dashboard")
    auth = _build_auth_dependency()

    @app.get("/", response_class=HTMLResponse)
    async def index(_: None = Depends(auth)):
        html_path = STATIC_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/account")
    async def get_account(_: None = Depends(auth)):
        return state.account_info

    @app.get("/api/positions")
    async def get_positions(_: None = Depends(auth)):
        return {"positions": state.open_positions}

    @app.get("/api/signals")
    async def get_signals(_: None = Depends(auth)):
        return state.latest_signals

    @app.get("/api/prices")
    async def get_prices(_: None = Depends(auth)):
        return state.latest_prices

    @app.get("/api/status")
    async def get_status(_: None = Depends(auth)):
        return state.system_status

    @app.get("/api/history")
    async def get_history(_: None = Depends(auth)):
        return {"trades": state.trade_history}

    @app.get("/api/stats")
    async def get_stats(_: None = Depends(auth)):
        return state.stats

    @app.get("/api/equity")
    async def get_equity(_: None = Depends(auth)):
        return {"points": state.equity_curve}

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
