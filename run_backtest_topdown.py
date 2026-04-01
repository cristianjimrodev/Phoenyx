"""Run top-down backtest (W1->D1->H4) with real IB data.

Downloads:
  - W1: up to 10 years (global trend + strong S/R)
  - D1: 1 year (intermediate confirmation)
  - H4: 10 months (entry/exit signals)

Usage:
  python run_backtest_topdown.py EURUSD 10000 55
"""
from __future__ import annotations

import asyncio
import os
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.broker.ib.client import IBClient
from src.strategy.topdown import TopDownStrategy
from src.reporting.report import generate_backtest_report
from src.utils.logging import setup_logging
from backtest.engine import BacktestEngine


async def run(symbol: str, initial_balance: float, min_confidence: float):
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

        # ── Load per-asset parameters from assets.yaml ─────────
        from src.strategy.topdown import load_asset_params
        asset_params = load_asset_params(symbol)
        if asset_params:
            logger.info(f"Loaded asset params for {symbol}")
        else:
            logger.info(f"No asset-specific params for {symbol}, using defaults")

        # Override min_confidence from asset params if present
        min_confidence = asset_params.get("min_confidence", min_confidence)

        tf_cfg = asset_params.get("timeframes", {})
        trend_tf = tf_cfg.get("trend", "W1")
        confirm_tf = tf_cfg.get("confirmation", "D1")
        entry_tf = tf_cfg.get("entry", "H4")

        durations = asset_params.get("trend_durations", {
            "W1": "10 Y", "D1": "1 Y", "H4": "10 M",
        })

        # Collect unique timeframes to download
        tfs_to_download = {}
        for tf in dict.fromkeys([entry_tf, confirm_tf, trend_tf]):  # preserves order, deduplicates
            dur = durations.get(tf, "10 M")
            tfs_to_download[tf] = dur

        logger.info(f"[{symbol}] Timeframes: {trend_tf} -> {confirm_tf} -> {entry_tf}")

        # ── Download data from IB ──────────────────────────────
        dataframes: dict[str, object] = {}

        for tf, dur in tfs_to_download.items():
            logger.info(f"Downloading {symbol} {tf} ({dur})...")
            df = await client.get_candles(symbol, tf, 50000, duration=dur)
            if not df.empty:
                logger.info(f"  {tf}: {len(df)} candles, {df.index[0]} -> {df.index[-1]}")
                dataframes[tf] = df
            else:
                logger.warning(f"  {tf}: no data received")
                if tf == entry_tf:
                    logger.error(f"  {entry_tf}: cannot run backtest without entry data")
                    return

        primary_df = dataframes[entry_tf]
        logger.info(f"\nPrice range: {primary_df['low'].min():.5f} - {primary_df['high'].max():.5f}")

        rsi_cfg = asset_params.get("rsi", {})
        stoch_cfg = asset_params.get("stochastic", {})
        atr_cfg = asset_params.get("atr", {})

        # ── Create strategy and engine ─────────────────────────
        strategy = TopDownStrategy(analysis_cfg, asset_params=asset_params)
        engine = BacktestEngine(
            strategy=strategy,
            initial_balance=initial_balance,
            risk_per_trade=0.02,
            min_confidence=min_confidence,
        )

        # ── Run MTF backtest ───────────────────────────────────
        tfs = list(dataframes.keys())
        logger.info(f"\nRunning top-down backtest ({' -> '.join(tfs)})...")
        logger.info(f"Strategy: {trend_tf} trend + {confirm_tf} confirm + {entry_tf} entry")
        logger.info(f"RSI({rsi_cfg.get('period', 14)}) OS<{rsi_cfg.get('oversold', 35)} OB>{rsi_cfg.get('overbought', 65)}")
        logger.info(f"Stoch({stoch_cfg.get('k_period', 5)},{stoch_cfg.get('d_period', 3)},{stoch_cfg.get('smooth', 3)}) OS<{stoch_cfg.get('oversold', 25)} OB>{stoch_cfg.get('overbought', 75)}")
        logger.info(f"ATR SL={atr_cfg.get('sl_multiplier', 1.5)}x TP={atr_cfg.get('tp_multiplier', 3.0)}x")
        logger.info(f"Min confidence: {min_confidence}")

        # Put entry TF first as primary (engine iterates over first key)
        ordered = {entry_tf: dataframes[entry_tf]}
        for tf in tfs:
            if tf != entry_tf and tf in dataframes:
                ordered[tf] = dataframes[tf]

        result = await engine.run_mtf(ordered, symbol, lookback=200)

        # ── Print results ──────────────────────────────────────
        final_balance = initial_balance * (1 + result.total_return_pct / 100)
        print("\n" + "=" * 64)
        print(f"  TOP-DOWN BACKTEST: {symbol} ({trend_tf} -> {confirm_tf} -> {entry_tf}) - REAL IB DATA")
        print("=" * 64)
        print(f"  Strategy:         {trend_tf} trend + {confirm_tf} confirm + {entry_tf} entry")
        print(f"  {entry_tf} period:        {primary_df.index[0].date()} to {primary_df.index[-1].date()}")
        for tf, df in dataframes.items():
            print(f"  {tf} bars:          {len(df)}")
        print(f"  Initial Balance:  ${initial_balance:,.2f}")
        print(f"  Final Balance:    ${final_balance:,.2f}")
        print("-" * 64)
        print(f"  Total Return:     {result.total_return_pct:+.2f}%")
        print(f"  Total Trades:     {result.total_trades}")
        print(f"  Win Rate:         {result.win_rate:.1f}%")
        print(f"  Winning Trades:   {result.winning_trades}")
        print(f"  Losing Trades:    {result.losing_trades}")
        print(f"  Avg Win:          ${result.avg_win:+.2f}")
        print(f"  Avg Loss:         ${result.avg_loss:+.2f}")
        pf = "inf" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
        print(f"  Profit Factor:    {pf}")
        print(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print("=" * 64)

        if result.trades:
            print(f"\n  Last 15 trades:")
            print(f"  {'Side':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Conf':>6} {'Reason'}")
            print(f"  {'-' * 56}")
            for t in result.trades[-15:]:
                print(f"  {t.side:<6} {t.entry_price:>10.5f} {t.exit_price:>10.5f} "
                      f"${t.pnl:>+8.2f} {t.confidence:>5.0f}%  {t.reason}")

        # Generate HTML report with candlestick chart
        report_path = generate_backtest_report(
            result, output_path=f"reports/backtest_{symbol}_topdown.html",
            price_df=primary_df,
        )
        print(f"\n  HTML report: {report_path}")

    finally:
        await client.disconnect()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    balance = float(sys.argv[2]) if len(sys.argv) > 2 else 10000
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 55

    print(f"\nTop-Down Backtest: {symbol} | ${balance:,.0f} | min_conf={confidence}")
    print(f"Timeframes: W1 (10yr) -> D1 (1yr) -> H4 (10mo)")
    print(f"Indicators: RSI(14), Stochastic(5,3,3)")

    asyncio.run(run(symbol, balance, confidence))


if __name__ == "__main__":
    main()
