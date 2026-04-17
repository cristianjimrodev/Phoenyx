"""Launch the Phoenyx dashboard.

Periodically reloads account + open positions from paper_state.json and trade
history/stats/equity from data/trades.db. Designed to run as a long-lived
service alongside run_daily.py (which writes the underlying files).

Usage:
    python run_dashboard.py                    # 127.0.0.1:8080
    python run_dashboard.py 0.0.0.0 8080       # bind all interfaces
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from src.dashboard.state import DashboardState
from src.dashboard.app import create_app
from src.data.trade_store import TradeStore


STATE_FILE = Path("data/paper_state.json")
REFRESH_INTERVAL = 10  # seconds
INITIAL_BALANCE = 1000.0  # EUR — matches PaperBroker default in run_daily.py


def _load_paper_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {STATE_FILE}: {e}")
        return None


def _refresh_from_paper(state: DashboardState) -> None:
    paper = _load_paper_state()
    if paper is None:
        return

    currency = paper.get("currency", "USD")
    balance = float(paper.get("balance", 0))
    trades = paper.get("trades", {})

    unrealized = sum(float(t.get("profit", 0)) for t in trades.values())
    equity = balance + unrealized

    state.account_info = {
        "balance": balance,
        "equity": equity,
        "margin": 0.0,
        "free_margin": equity,
        "currency": currency,
    }

    state.open_positions = [
        {
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "volume": float(t.get("volume", 0)),
            "open_price": float(t.get("open_price", 0)),
            "sl": float(t.get("sl", 0)),
            "tp": float(t.get("tp", 0)),
            "profit": float(t.get("profit", 0)),
            "contract_size": float(t.get("contract_size", 100000)),
        }
        for t in trades.values()
    ]


def _refresh_from_store(state: DashboardState, store: TradeStore, currency: str) -> None:
    df = store.load_trades(limit=1000)
    if df.empty:
        state.trade_history = []
        state.stats = {"total": 0, "closed": 0, "win_rate": 0, "total_pnl": 0}
        state.equity_curve = [{"t": None, "equity": INITIAL_BALANCE}]
        return

    state.trade_history = [
        {
            "id": int(row["id"]),
            "timestamp": row["timestamp"],
            "symbol": row["symbol"],
            "side": row["side"],
            "volume": float(row["volume"]),
            "entry_price": float(row["entry_price"]),
            "exit_price": float(row["exit_price"]),
            "sl": float(row["sl"]),
            "tp": float(row["tp"]),
            "confidence": float(row["confidence"]),
            "status": row["status"],
            "pnl": float(row["pnl"]),
        }
        for _, row in df.iterrows()
    ]

    state.stats = {**store.get_stats(), "currency": currency}

    closed = df[df["status"] == "closed"].sort_values("timestamp")
    if closed.empty:
        state.equity_curve = [{"t": None, "equity": INITIAL_BALANCE}]
    else:
        equity = INITIAL_BALANCE
        points = [{"t": None, "equity": equity}]
        for _, row in closed.iterrows():
            equity += float(row["pnl"])
            points.append({"t": row["timestamp"], "equity": round(equity, 2)})
        state.equity_curve = points


_TRADE_STORE: TradeStore | None = None


def _refresh_state(state: DashboardState) -> None:
    global _TRADE_STORE
    _refresh_from_paper(state)

    if _TRADE_STORE is None:
        try:
            _TRADE_STORE = TradeStore()
        except Exception as e:
            logger.warning(f"Could not open TradeStore: {e}")
            return

    try:
        _refresh_from_store(state, _TRADE_STORE,
                            state.account_info.get("currency", "USD"))
    except Exception as e:
        logger.warning(f"TradeStore refresh error: {e}")


async def _refresh_loop(state: DashboardState):
    while True:
        try:
            _refresh_state(state)
        except Exception as e:
            logger.warning(f"Dashboard refresh error: {e}")
        await asyncio.sleep(REFRESH_INTERVAL)


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    state = DashboardState()
    _refresh_state(state)

    app = create_app(state)

    @app.on_event("startup")
    async def _start_refresh():
        asyncio.create_task(_refresh_loop(state))

    logger.info(f"Phoenyx Dashboard running at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
