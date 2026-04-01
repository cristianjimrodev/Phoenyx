"""Launch the Phoenyx dashboard standalone for previewing backtest/trade data."""
from __future__ import annotations

import asyncio
import sys

import uvicorn
from loguru import logger

from src.dashboard.state import DashboardState
from src.dashboard.app import create_app
from src.data.trade_store import TradeStore


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    state = DashboardState()

    # Load trade history if available
    try:
        ts = TradeStore()
        df = ts.load_trades(limit=200)
        if not df.empty:
            trades = df.to_dict("records")
            state.update_positions([])
            logger.info(f"Loaded {len(trades)} trades from history")
        ts.close()
    except Exception:
        pass

    state.update_account({
        "balance": 11921.29,
        "equity": 11921.29,
        "free_margin": 11921.29,
        "currency": "USD",
    })

    state.system_status["status"] = "dashboard preview"

    app = create_app(state)

    print(f"\n  Phoenyx Dashboard running at http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
