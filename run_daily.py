"""Daily trading bot — run once each morning.

1. Connect to IB (for market data)
2. Load PaperBroker state from previous day
3. Check if any open trades hit SL/TP overnight (using yesterday's candles)
4. Evaluate all configured assets with TopDown strategy
5. Open new trades where signals are strong enough
6. Save state and disconnect

Usage:
    python run_daily.py                  # run all configured assets
    python run_daily.py EURUSD GBPJPY    # run specific assets only
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.broker.ib.client import IBClient
from src.broker.paper import PaperBroker
from src.broker.base import Side
from src.strategy.topdown import TopDownStrategy, load_asset_params
from src.risk.manager import RiskManager
from src.orders.manager import OrderManager
from src.data.trade_store import TradeStore
from src.reporting.report import generate_live_report
from src.utils.logging import setup_logging


# All tradeable assets (excluding those marked excluded in assets.yaml)
ALL_ASSETS = [
    # Forex Major
    "EURUSD", "EURGBP", "EURJPY", "EURCHF", "GBPUSD", "GBPJPY",
    "GBPCHF", "USDJPY", "USDCHF", "AUDUSD",
    # Forex Minor
    "EURAUD", "EURCAD", "EURNZD", "EURSEK", "EURNOK", "EURPLN",
    "GBPCAD", "GBPNZD", "GBPNOK", "GBPPLN", "GBPSGD",
    "USDCAD", "USDSEK", "USDNOK", "USDPLN", "USDSGD", "USDMXN",
    # Indices
    "QQQ", "SPY", "DIA", "IWM", "DAX", "EWP", "EWU", "EWQ",
    "EWI", "FEZ", "EWZ", "EWJ", "VNM", "EWH",
    # Commodities
    "NG", "ZS", "KC", "CL", "ZW", "SB", "CT", "CC", "ZL", "ZC",
    # Metals
    "GC", "SI", "HG", "ALI",
    # Bonds
    "ZN",
]


async def check_overnight_sltp(paper: PaperBroker, ib: IBClient,
                               trade_store: TradeStore | None = None):
    """Check if any open trade hit SL/TP since last run using H4 candle highs/lows."""
    open_trades = await paper.get_open_trades()
    if not open_trades:
        return

    logger.info(f"Checking {len(open_trades)} open trades for overnight SL/TP hits...")

    for trade in open_trades:
        try:
            # Get the last 6 H4 candles (~24h)
            df = await ib.get_candles(trade.symbol, "H4", 6)
            if df.empty:
                continue

            period_high = df["high"].max()
            period_low = df["low"].min()
            current_price = df["close"].iloc[-1]

            # Update latest price for open position PnL
            paper._latest_prices[trade.symbol] = current_price

            hit_sl = False
            hit_tp = False

            if trade.side == Side.BUY:
                hit_sl = trade.sl > 0 and period_low <= trade.sl
                hit_tp = trade.tp > 0 and period_high >= trade.tp
            else:
                hit_sl = trade.sl > 0 and period_high >= trade.sl
                hit_tp = trade.tp > 0 and period_low <= trade.tp

            if hit_tp:
                # Close at TP price, not current price
                paper._latest_prices[trade.symbol] = trade.tp
                logger.info(
                    f"  [TP HIT] {trade.symbol} {trade.side.value} @ {trade.open_price:.5f} "
                    f"-> TP {trade.tp:.5f} (high={period_high:.5f})"
                )
                await paper.close_trade(trade.trade_id)
                if trade_store:
                    exit_price = trade.tp
                    if trade.side == Side.BUY:
                        pnl = (exit_price - trade.open_price) * trade.volume * trade.contract_size
                    else:
                        pnl = (trade.open_price - exit_price) * trade.volume * trade.contract_size
                    trade_store.update_order(trade.trade_id, "closed", exit_price, pnl)
            elif hit_sl:
                # Close at SL price, not current price
                paper._latest_prices[trade.symbol] = trade.sl
                logger.info(
                    f"  [SL HIT] {trade.symbol} {trade.side.value} @ {trade.open_price:.5f} "
                    f"-> SL {trade.sl:.5f} (low={period_low:.5f})"
                )
                await paper.close_trade(trade.trade_id)
                if trade_store:
                    exit_price = trade.sl
                    if trade.side == Side.BUY:
                        pnl = (exit_price - trade.open_price) * trade.volume * trade.contract_size
                    else:
                        pnl = (trade.open_price - exit_price) * trade.volume * trade.contract_size
                    trade_store.update_order(trade.trade_id, "closed", exit_price, pnl)
            else:
                logger.info(
                    f"  [OPEN] {trade.symbol} {trade.side.value} @ {trade.open_price:.5f} "
                    f"SL={trade.sl:.5f} TP={trade.tp:.5f} "
                    f"current={current_price:.5f} PnL={trade.profit:+.2f}"
                )
        except Exception as e:
            logger.warning(f"  Error checking {trade.symbol}: {e}")


async def evaluate_and_open(symbol: str, paper: PaperBroker, ib: IBClient,
                            strategy: TopDownStrategy, risk_mgr: RiskManager,
                            order_mgr: OrderManager, trade_store: TradeStore):
    """Evaluate one symbol and open a trade if signal is strong."""
    try:
        asset_params = load_asset_params(symbol)
        if asset_params.get("excluded", False):
            return None

        min_confidence = asset_params.get("min_confidence", 80)

        tf_cfg = asset_params.get("timeframes", {})
        trend_tf = tf_cfg.get("trend", "W1")
        confirm_tf = tf_cfg.get("confirmation", "D1")
        entry_tf = tf_cfg.get("entry", "H4")
        durations = asset_params.get("trend_durations", {
            "W1": "10 Y", "D1": "1 Y", "H4": "10 M",
        })

        # Check if we already have an open trade on this symbol
        open_trades = await paper.get_open_trades()
        if any(t.symbol == symbol for t in open_trades):
            return None

        # Download data
        dataframes = {}
        for tf in dict.fromkeys([entry_tf, confirm_tf, trend_tf]):
            dur = durations.get(tf, "10 M")
            df = await ib.get_candles(symbol, tf, 50000, duration=dur)
            if not df.empty:
                dataframes[tf] = df
            elif tf == entry_tf:
                return None

        # Evaluate
        ordered = {entry_tf: dataframes[entry_tf]}
        for tf in dataframes:
            if tf != entry_tf:
                ordered[tf] = dataframes[tf]

        signal = await strategy.evaluate_mtf(symbol, ordered)

        if signal.signal.value == "hold" or signal.confidence < min_confidence:
            return None

        if signal.suggested_sl == 0:
            return None

        # Risk check
        account = await paper.get_account_info()
        can_trade, reason = risk_mgr.can_trade(account, open_trades)
        if not can_trade:
            logger.info(f"  {symbol}: Risk blocked - {reason}")
            return None

        # Position size
        # Position size: use default pip_value and contract_size for forex
        # For other assets, try to get symbol info from IB
        try:
            sym_info = await ib.get_symbol(symbol)
            pip_value = sym_info.pip_size
            contract_size = sym_info.contract_size
        except Exception:
            pip_value = 0.0001
            contract_size = 100000

        entry_price = dataframes[entry_tf]["close"].iloc[-1]
        volume = risk_mgr.compute_position_size(
            account, signal, pip_value=pip_value, contract_size=contract_size,
            entry_price=entry_price,
        )
        if volume <= 0:
            return None

        # Execute
        side = Side.BUY if signal.signal.value == "buy" else Side.SELL
        trade = await paper.open_trade(
            symbol, side, volume, sl=signal.suggested_sl, tp=signal.suggested_tp,
        )

        # Log to trade store
        from src.orders.manager import OrderRecord
        record = OrderRecord(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=signal.signal.value,
            volume=volume,
            entry_price=trade.open_price,
            sl=signal.suggested_sl,
            tp=signal.suggested_tp,
            confidence=signal.confidence,
            reasons=signal.details[-3:],  # last 3 details
            trade_id=trade.trade_id,
            status="opened",
            contract_size=trade.contract_size,
        )
        trade_store.save_order(record)

        return {
            "symbol": symbol,
            "side": signal.signal.value.upper(),
            "price": trade.open_price,
            "sl": signal.suggested_sl,
            "tp": signal.suggested_tp,
            "volume": volume,
            "confidence": signal.confidence,
        }

    except Exception as e:
        logger.error(f"  {symbol}: Error - {e}")
        return None


async def main():
    load_dotenv()
    setup_logging(level="INFO")

    symbols = sys.argv[1:] if len(sys.argv) > 1 else ALL_ASSETS

    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "7497"))
    client_id = int(os.getenv("IB_CLIENT_ID", "1"))

    # Load configs
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open("config/strategies.yaml") as f:
        strat_config = yaml.safe_load(f)

    # Connect IB for market data
    ib = IBClient(host=host, port=port, client_id=client_id)
    connected = await ib.connect()
    if not connected:
        logger.error("Could not connect to IB. Check TWS/Gateway.")
        return

    # Paper broker with persistent state
    paper = PaperBroker(ib, initial_balance=1000.0, currency="EUR")
    trade_store = TradeStore()
    risk_mgr = RiskManager(settings.get("risk", {}))
    order_mgr = OrderManager(paper, trade_store=trade_store)
    strategy = TopDownStrategy(strat_config.get("analysis", {}))

    try:
        account = await paper.get_account_info()
        open_trades = await paper.get_open_trades()

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        print()
        print("=" * 64)
        print(f"  PHOENYX DAILY RUN - {now}")
        print("=" * 64)
        print(f"  Balance:     {account.balance:,.2f} EUR")
        print(f"  Equity:      {account.equity:,.2f} EUR")
        print(f"  Open trades: {len(open_trades)}")
        print("-" * 64)

        # Step 1: Check overnight SL/TP on existing trades
        if open_trades:
            print(f"\n  Checking {len(open_trades)} open trades for SL/TP hits...")
            await check_overnight_sltp(paper, ib, trade_store)
            open_trades = await paper.get_open_trades()
            print(f"  Remaining open: {len(open_trades)}")

        # Step 2: Evaluate all assets and open new trades
        print(f"\n  Evaluating {len(symbols)} assets...")
        new_trades = []

        for sym in symbols:
            result = await evaluate_and_open(
                sym, paper, ib, strategy, risk_mgr, order_mgr, trade_store,
            )
            if result:
                new_trades.append(result)
                print(
                    f"    NEW {result['side']:>4} {result['symbol']:<10} "
                    f"@ {result['price']:.5f}  SL={result['sl']:.5f}  "
                    f"TP={result['tp']:.5f}  vol={result['volume']:.2f}  "
                    f"conf={result['confidence']:.0f}%"
                )

        # Step 3: Summary
        account = await paper.get_account_info()
        all_open = await paper.get_open_trades()

        print()
        print("-" * 64)
        print(f"  NEW TRADES TODAY:  {len(new_trades)}")
        print(f"  TOTAL OPEN:       {len(all_open)}")
        print(f"  BALANCE:          {account.balance:,.2f} EUR")
        print(f"  EQUITY:           {account.equity:,.2f} EUR")

        if all_open:
            print(f"\n  OPEN POSITIONS:")
            print(f"  {'Symbol':<10} {'Side':<5} {'Vol':>5} {'Entry':>10} {'SL':>10} {'TP':>10} {'PnL':>8}")
            print(f"  {'-' * 60}")
            for t in all_open:
                print(
                    f"  {t.symbol:<10} {t.side.value:<5} {t.volume:>5.2f} "
                    f"{t.open_price:>10.5f} {t.sl:>10.5f} {t.tp:>10.5f} "
                    f"{t.profit:>+8.2f}"
                )

        # Step 4: Save state
        paper.save_state()

        # Step 5: Generate daily report
        trade_df = trade_store.load_trades(limit=500)
        if not trade_df.empty:
            report_path = generate_live_report(
                trade_df, output_path="reports/daily_report.html",
                initial_balance=account.balance,
            )
            print(f"\n  Report: {report_path}")

        stats = trade_store.get_stats()
        if stats.get("total", 0) > 0:
            print(f"\n  OVERALL STATS:")
            print(f"  Total trades: {stats['total']}")
            print(f"  Win rate:     {stats.get('win_rate', 0):.1f}%")
            print(f"  Total PnL:    {stats.get('total_pnl', 0):+.2f} EUR")

        print()
        print("=" * 64)
        print(f"  Done. Next run: tomorrow morning.")
        print("=" * 64)

    finally:
        await ib.disconnect()
        trade_store.close()


if __name__ == "__main__":
    asyncio.run(main())
