"""Run a backtest with simulated historical data (no broker connection needed)."""
from __future__ import annotations

import asyncio
import sys

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.strategy.technical import TechnicalStrategy
from src.reporting.report import generate_backtest_report
from backtest.engine import BacktestEngine


def generate_realistic_data(symbol: str, bars: int = 1000,
                            base_price: float = 1.1000,
                            volatility: float = 0.0008) -> pd.DataFrame:
    """Generate realistic OHLCV data with trends, ranges, and patterns."""
    np.random.seed(42)

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="1h")
    prices = np.zeros(bars)
    prices[0] = base_price

    # Generate price movement with regime changes
    regime_length = 50
    for i in range(1, bars):
        # Change regime periodically (trending vs ranging)
        regime = (i // regime_length) % 4
        if regime == 0:    # uptrend
            drift = volatility * 0.3
        elif regime == 1:  # range
            drift = 0
        elif regime == 2:  # downtrend
            drift = -volatility * 0.3
        else:              # volatile range
            drift = 0
            volatility_mult = 1.5

        noise = np.random.normal(drift, volatility)
        prices[i] = prices[i - 1] * (1 + noise)

    # Generate OHLCV from close prices
    rows = []
    for i in range(bars):
        c = prices[i]
        spread = c * volatility * np.random.uniform(0.5, 2.0)
        h = c + abs(np.random.normal(0, spread))
        l = c - abs(np.random.normal(0, spread))
        o = prices[i - 1] if i > 0 else c
        # Ensure OHLC consistency
        h = max(h, o, c)
        l = min(l, o, c)
        vol = np.random.uniform(1000, 10000)

        rows.append({
            "timestamp": int(timestamps[i].timestamp() * 1000),
            "open": round(o, 5),
            "high": round(h, 5),
            "low": round(l, 5),
            "close": round(c, 5),
            "volume": round(vol, 0),
        })

    df = pd.DataFrame(rows)
    df["datetime"] = timestamps
    df = df.set_index("datetime")
    return df


# Presets for different assets
PRESETS = {
    "EURUSD": {"base_price": 1.0850, "volatility": 0.0006},
    "GBPUSD": {"base_price": 1.2700, "volatility": 0.0008},
    "USDJPY": {"base_price": 150.50, "volatility": 0.0007},
    "BITCOIN": {"base_price": 65000.0, "volatility": 0.0150},
    "US500":  {"base_price": 5200.0,  "volatility": 0.0050},
    "GOLD":   {"base_price": 2350.0,  "volatility": 0.0040},
}


async def run_backtest(symbol: str, bars: int, initial_balance: float,
                       min_confidence: float):
    # Load strategy config
    with open("config/strategies.yaml") as f:
        strat_config = yaml.safe_load(f)

    strategy = TechnicalStrategy(strat_config.get("analysis", {}))
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=initial_balance,
        risk_per_trade=0.02,
        min_confidence=min_confidence,
    )

    # Generate data
    preset = PRESETS.get(symbol, {"base_price": 100.0, "volatility": 0.005})
    logger.info(f"Generating {bars} bars of simulated data for {symbol}...")
    df = generate_realistic_data(symbol, bars=bars, **preset)

    logger.info(f"Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
    logger.info(f"Period: {df.index[0]} to {df.index[-1]}")

    # Run backtest
    result = await engine.run(df, symbol, lookback=200)

    # Print results
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS: {symbol}")
    print("=" * 60)
    print(f"  Period:           {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Bars:             {len(df)}")
    print(f"  Initial Balance:  ${initial_balance:,.2f}")
    print(f"  Final Balance:    ${initial_balance * (1 + result.total_return_pct / 100):,.2f}")
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
        print(f"  {'-'*50}")
        for t in result.trades[-10:]:
            print(f"  {t.side:<6} {t.entry_price:>10.5f} {t.exit_price:>10.5f} "
                  f"${t.pnl:>+8.2f}  {t.reason}")

    # Generate HTML report
    report_path = generate_backtest_report(
        result, output_path=f"reports/backtest_{symbol}.html",
    )
    print(f"\n  HTML report: {report_path}")

    return result


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    bars = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    balance = float(sys.argv[3]) if len(sys.argv) > 3 else 10000
    confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 60

    available = list(PRESETS.keys())
    if symbol not in PRESETS:
        print(f"Available symbols: {', '.join(available)}")
        print(f"Using {symbol} with default price preset")

    print(f"\nRunning backtest: {symbol} | {bars} bars | ${balance:,.0f} balance | "
          f"min confidence: {confidence}")

    asyncio.run(run_backtest(symbol, bars, balance, confidence))


if __name__ == "__main__":
    main()
