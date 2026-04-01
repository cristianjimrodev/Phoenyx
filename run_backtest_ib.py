"""Run backtest using real historical data from Interactive Brokers (multi-timeframe)."""
from __future__ import annotations

import asyncio
import os
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.broker.ib.client import IBClient
from src.strategy.technical import TechnicalStrategy
from src.reporting.report import generate_backtest_report
from src.utils.logging import setup_logging
from backtest.engine import BacktestEngine


# Duration strings sent to IB for each timeframe
DURATION_MAP = {
    "M5":  "1 M",
    "M15": "2 M",
    "M30": "3 M",
    "H1":  "10 M",
    "H4":  "1 Y",
    "D1":  "2 Y",
}


async def run(symbol: str, timeframe: str, initial_balance: float,
              min_confidence: float, months: int = 10):
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
        # Load strategy config
        with open("config/strategies.yaml") as f:
            strat_config = yaml.safe_load(f)

        analysis_cfg = strat_config.get("analysis", {})
        mtf_cfg = analysis_cfg.get("multi_timeframe", {})
        mtf_enabled = mtf_cfg.get("enabled", False)

        # Determine timeframes to download
        confirmation_tfs = []
        if mtf_enabled:
            tf_weights = mtf_cfg.get("weights", {})
            confirmation_tfs = [tf for tf in tf_weights if tf != timeframe]
            all_timeframes = [timeframe] + confirmation_tfs
        else:
            all_timeframes = [timeframe]

        logger.info(f"Timeframes to download: {all_timeframes}")

        # Duration for primary TF based on requested months
        primary_duration = f"{months} M"

        # Download data for all timeframes
        dataframes: dict[str, object] = {}
        for tf in all_timeframes:
            if tf == timeframe:
                duration = primary_duration
            else:
                # HTFs need at least the same calendar period
                duration = primary_duration

            logger.info(f"Downloading {symbol} {tf} ({duration})...")
            df = await client.get_candles(symbol, tf, 50000, duration=duration)

            if df.empty:
                logger.error(f"No data received for {symbol} {tf}")
                if tf == timeframe:
                    return
                continue

            logger.info(f"  {tf}: {len(df)} candles, {df.index[0]} → {df.index[-1]}")
            dataframes[tf] = df

        primary_df = dataframes[timeframe]
        logger.info(f"\nPrice range: {primary_df['low'].min():.5f} - {primary_df['high'].max():.5f}")

        # Create strategy and engine
        strategy = TechnicalStrategy(analysis_cfg)
        engine = BacktestEngine(
            strategy=strategy,
            initial_balance=initial_balance,
            risk_per_trade=0.02,
            min_confidence=min_confidence,
        )

        # Run backtest
        if mtf_enabled and len(dataframes) > 1:
            logger.info(f"\nRunning MTF backtest ({', '.join(dataframes.keys())})...")
            result = await engine.run_mtf(dataframes, symbol, lookback=200)
            mode = "MTF"
        else:
            logger.info(f"\nRunning single-timeframe backtest ({timeframe})...")
            result = await engine.run(primary_df, symbol, lookback=200)
            mode = "Single TF"

        # Print results
        final_balance = initial_balance * (1 + result.total_return_pct / 100)
        print("\n" + "=" * 60)
        print(f"  BACKTEST: {symbol} ({mode}: {', '.join(dataframes.keys())}) - REAL IB DATA")
        print("=" * 60)
        print(f"  Period:           {primary_df.index[0].date()} to {primary_df.index[-1].date()}")
        print(f"  Primary bars:     {len(primary_df)} ({timeframe})")
        for tf, df in dataframes.items():
            if tf != timeframe:
                print(f"  {tf} bars:          {len(df)}")
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
            print(f"  {'Side':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Conf':>6} {'Reason'}")
            print(f"  {'-' * 56}")
            for t in result.trades[-10:]:
                print(f"  {t.side:<6} {t.entry_price:>10.5f} {t.exit_price:>10.5f} "
                      f"${t.pnl:>+8.2f} {t.confidence:>5.0f}%  {t.reason}")

        # Generate HTML report with candlestick chart
        report_path = generate_backtest_report(
            result, output_path=f"reports/backtest_{symbol}_ib_mtf.html",
            price_df=primary_df,
        )
        print(f"\n  HTML report: {report_path}")

    finally:
        await client.disconnect()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "H1"
    balance = float(sys.argv[3]) if len(sys.argv) > 3 else 10000
    confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 30
    months = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    print(f"\nBacktest: {symbol} | {timeframe} | ${balance:,.0f} | "
          f"min_conf={confidence} | {months} months")
    asyncio.run(run(symbol, timeframe, balance, confidence, months))


if __name__ == "__main__":
    main()
