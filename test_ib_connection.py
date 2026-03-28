"""Test Interactive Brokers connection and basic functionality."""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from src.broker.ib.client import IBClient
from src.utils.logging import setup_logging


async def test():
    load_dotenv()
    setup_logging(level="DEBUG")

    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "7497"))
    client_id = int(os.getenv("IB_CLIENT_ID", "1"))

    logger.info(f"Connecting to IB at {host}:{port} (clientId={client_id})")
    logger.info("Make sure TWS or IB Gateway is running!")
    print()

    client = IBClient(host=host, port=port, client_id=client_id)

    # 1. Connect
    connected = await client.connect()
    if not connected:
        logger.error("Could not connect. Check that TWS/Gateway is running and API is enabled.")
        return

    print("\n" + "=" * 50)
    print("  CONNECTION SUCCESSFUL")
    print("=" * 50)

    # 2. Account info
    try:
        account = await client.get_account_info()
        print(f"\n  Account Info:")
        print(f"    Balance:     ${account.balance:,.2f}")
        print(f"    Equity:      ${account.equity:,.2f}")
        print(f"    Margin:      ${account.margin:,.2f}")
        print(f"    Free Margin: ${account.free_margin:,.2f}")
        print(f"    Currency:    {account.currency}")
    except Exception as e:
        logger.error(f"Account info failed: {e}")

    # 3. Get historical data for EURUSD
    print(f"\n  Fetching EURUSD H1 candles...")
    try:
        df = await client.get_candles("EURUSD", "H1", 100)
        if not df.empty:
            print(f"    Received {len(df)} candles")
            print(f"    Range: {df.index[0]} to {df.index[-1]}")
            print(f"    Last close: {df['close'].iloc[-1]:.5f}")
            print(f"    High: {df['high'].max():.5f} | Low: {df['low'].min():.5f}")
        else:
            print("    No data received")
    except Exception as e:
        logger.error(f"Candles failed: {e}")

    # 4. Get AAPL stock data
    print(f"\n  Fetching AAPL D1 candles...")
    try:
        df = await client.get_candles("AAPL", "D1", 30)
        if not df.empty:
            print(f"    Received {len(df)} candles")
            print(f"    Last close: ${df['close'].iloc[-1]:.2f}")
        else:
            print("    No data received")
    except Exception as e:
        logger.error(f"AAPL candles failed: {e}")

    # 5. Open positions
    try:
        trades = await client.get_open_trades()
        print(f"\n  Open Positions: {len(trades)}")
        for t in trades:
            print(f"    {t.symbol}: {t.side.value} {t.volume} @ {t.open_price:.2f}")
    except Exception as e:
        logger.error(f"Open trades failed: {e}")

    print("\n" + "=" * 50)
    print("  TEST COMPLETE")
    print("=" * 50)

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test())
