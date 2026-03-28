"""Run backtest using real historical data from Interactive Brokers."""
from __future__ import annotations

import asyncio
import os
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.broker.ib.client import IBClient
from src.strategy.technical import TechnicalStrategy
from src.utils.logging import setup_logging
from backtest.engine import BacktestEngine


DURATION_MAP = {
    "M5":  "1 M",
    "M15": "2 M",
    "M30": "3 M",
    "H1":  "6 M",
    "H4":  "1 Y",
    "D1":  "2 Y",
}


async def run(symbol: str, timeframe: str, initial_balance: float,
              min_confidence: float):
    load_dotenv()
    setup_logging(level="INFO")

    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "7497"))
    client_id = int(os.getenv("IB_CLIENT_ID", "1"))

    client = IBClient(host=host, port=port, client_id=client_id)

    connected = await client.connect()
    if not connected:
        logger.error("Could not connect to IB. Check TWS/Gateway.")
        return

    try:
        # Fetch historical data
        duration = DURATION_MAP.get(timeframe, "6 M")
        logger.info(f"Downloading {symbol} {timeframe} data ({duration} of history)...")

        df = await client.get_candles(symbol, timeframe, 5000)

        if df.empty:
            logger.error(f"No data received for {symbol}")
            return

        logger.info(f"Received {len(df)} candles: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")

        # Run backtest
        with open("config/strategies.yaml") as f:
            strat_config = yaml.safe_load(f)

        strategy = TechnicalStrategy(strat_config.get("analysis", {}))
        engine = BacktestEngine(
            strategy=strategy,
            initial_balance=initial_balance,
            risk_per_trade=0.02,
            min_confidence=min_confidence,
        )

        result = await engine.run(df, symbol, lookback=200)

        # Print results
        final_balance = initial_balance * (1 + result.total_return_pct / 100)
        print("\n" + "=" * 60)
        print(f"  BACKTEST: {symbol} ({timeframe}) - REAL IB DATA")
        print("=" * 60)
        print(f"  Period:           {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Bars:             {len(df)}")
        print(f"  Initial Balance:  ${initial_balance:,.2f}")
        print(f"  Final Balance:    ${final_balance:,.2f}")
        print("-" * 60)
        print(f"  Total Return:     {result.total_return_pct:+.2f}%")
        print(f"  Total Trades:     {result.total_trades}")
        print(f"  Win Rate:         {result.win_rate:.1f}%")
        print(f"  Winning Trades:   {result.winning_trades}")
        print(f"  Losing Trades:    {result.losing_trades}")
        print(f"  Avg Win:          ${result.avg_win:+.2f}")
        print(f"  Avg Loss:         ${result.avg_loss:+.2f}")
        print(f"  Profit Factor:    {result.profit_factor:.2f}")
        print(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print("=" * 60)

        if result.trades:
            print(f"\n  Last 10 trades:")
            print(f"  {'Side':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Reason'}")
            print(f"  {'-' * 50}")
            for t in result.trades[-10:]:
                print(f"  {t.side:<6} {t.entry_price:>10.5f} {t.exit_price:>10.5f} "
                      f"${t.pnl:>+8.2f}  {t.reason}")

    finally:
        await client.disconnect()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "H1"
    balance = float(sys.argv[3]) if len(sys.argv) > 3 else 10000
    confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 30

    print(f"Backtest: {symbol} | {timeframe} | ${balance:,.0f} | min_conf={confidence}")
    asyncio.run(run(symbol, timeframe, balance, confidence))


if __name__ == "__main__":
    main()
