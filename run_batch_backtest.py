"""Run backtest on multiple symbols and print summary table."""
from __future__ import annotations

import asyncio
import os
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.broker.ib.client import IBClient
from src.strategy.topdown import TopDownStrategy, load_asset_params
from src.reporting.report import generate_backtest_report
from src.utils.logging import setup_logging
from backtest.engine import BacktestEngine


async def run_single(client: IBClient, symbol: str, analysis_cfg: dict) -> dict:
    """Run backtest for one symbol, return summary dict."""
    try:
        asset_params = load_asset_params(symbol)
        min_confidence = asset_params.get("min_confidence", 75)

        tf_cfg = asset_params.get("timeframes", {})
        trend_tf = tf_cfg.get("trend", "W1")
        confirm_tf = tf_cfg.get("confirmation", "D1")
        entry_tf = tf_cfg.get("entry", "H4")
        durations = asset_params.get("trend_durations", {"W1": "10 Y", "D1": "1 Y", "H4": "10 M"})

        tfs_to_download = {}
        for tf in dict.fromkeys([entry_tf, confirm_tf, trend_tf]):
            tfs_to_download[tf] = durations.get(tf, "10 M")

        dataframes = {}
        for tf, dur in tfs_to_download.items():
            df = await client.get_candles(symbol, tf, 50000, duration=dur)
            if not df.empty:
                dataframes[tf] = df
            elif tf == entry_tf:
                return {"symbol": symbol, "error": f"No {entry_tf} data"}

        strategy = TopDownStrategy(analysis_cfg, asset_params=asset_params)
        engine = BacktestEngine(strategy=strategy, initial_balance=10000,
                                risk_per_trade=0.02, min_confidence=min_confidence)

        ordered = {entry_tf: dataframes[entry_tf]}
        for tf in dataframes:
            if tf != entry_tf:
                ordered[tf] = dataframes[tf]

        result = await engine.run_mtf(ordered, symbol, lookback=200)

        generate_backtest_report(result, output_path=f"reports/backtest_{symbol}_topdown.html",
                                 price_df=dataframes[entry_tf])

        return {
            "symbol": symbol,
            "return": result.total_return_pct,
            "trades": result.total_trades,
            "win_rate": result.win_rate,
            "pf": result.profit_factor,
            "max_dd": result.max_drawdown_pct,
            "sharpe": result.sharpe_ratio,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)[:60]}


async def main():
    load_dotenv()
    setup_logging(level="WARNING")

    symbols = sys.argv[1:] if len(sys.argv) > 1 else [
        "SPY", "IWM", "DIA", "EWP", "EWU", "EWQ", "EWI", "FEZ", "VIXY", "ZN",
    ]

    client = IBClient(
        host=os.getenv("IB_HOST", "127.0.0.1"),
        port=int(os.getenv("IB_PORT", "7497")),
        client_id=int(os.getenv("IB_CLIENT_ID", "1")),
    )

    connected = await client.connect()
    if not connected:
        print("Could not connect to IB.")
        return

    with open("config/strategies.yaml") as f:
        strat_config = yaml.safe_load(f)
    analysis_cfg = strat_config.get("analysis", {})

    results = []
    for sym in symbols:
        print(f"  Running {sym}...", end=" ", flush=True)
        r = await run_single(client, sym, analysis_cfg)
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(f"{r['return']:+.2f}% ({r['trades']} trades)")
        results.append(r)

    await client.disconnect()

    # Print summary table
    print("\n" + "=" * 90)
    print(f"  {'Symbol':<8} {'Return':>8} {'Trades':>7} {'WinRate':>8} {'PF':>7} "
          f"{'MaxDD':>7} {'Sharpe':>8} {'AvgWin':>8} {'AvgLoss':>8}")
    print("  " + "-" * 82)

    total_return = 0
    total_trades = 0

    for r in sorted(results, key=lambda x: x.get("return", -999), reverse=True):
        if "error" in r:
            print(f"  {r['symbol']:<8} {'ERROR':>8}   {r['error']}")
            continue
        pf = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
        print(f"  {r['symbol']:<8} {r['return']:>+7.2f}% {r['trades']:>7} "
              f"{r['win_rate']:>7.1f}% {pf:>7} {r['max_dd']:>6.2f}% "
              f"{r['sharpe']:>8.2f} {r['avg_win']:>+8.2f} {r['avg_loss']:>+8.2f}")
        total_return += r["return"]
        total_trades += r["trades"]

    print("  " + "-" * 82)
    n = sum(1 for r in results if "error" not in r)
    if n > 0:
        print(f"  {'AVG':<8} {total_return/n:>+7.2f}% {total_trades:>7}")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
