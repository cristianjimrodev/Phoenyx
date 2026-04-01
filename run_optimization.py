"""Run grid search optimization on strategy parameters."""
from __future__ import annotations

import asyncio
import sys

import yaml
from loguru import logger

from backtest.optimizer import GridSearchOptimizer, OptimizationParam
from run_backtest import generate_realistic_data, PRESETS
from src.reporting.report import generate_optimization_report


async def run_optimization(symbol: str, bars: int):
    # Load strategy config
    with open("config/strategies.yaml") as f:
        strat_config = yaml.safe_load(f)

    base_config = strat_config.get("analysis", {})

    # Load optimization config
    with open("backtest/optimization_config.yaml") as f:
        opt_config = yaml.safe_load(f)

    opt = opt_config["optimization"]

    # Build optimization parameters
    params = [
        OptimizationParam(path=p["path"], values=p["values"])
        for p in opt["parameters"]
    ]

    # Generate data
    preset = PRESETS.get(symbol, {"base_price": 100.0, "volatility": 0.005})
    logger.info(f"Generating {bars} bars of simulated data for {symbol}...")
    df = generate_realistic_data(symbol, bars=bars, **preset)

    logger.info(f"Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
    logger.info(f"Period: {df.index[0]} to {df.index[-1]}")

    # Create optimizer and run
    optimizer = GridSearchOptimizer(
        base_config=base_config,
        params=params,
        objective=opt.get("objective", "sharpe_ratio"),
        initial_balance=opt.get("initial_balance", 10000),
        min_confidence=opt.get("min_confidence", 60),
    )

    result = await optimizer.run(df, symbol, lookback=200)

    # Print summary
    print("\n" + "=" * 80)
    print(f"  OPTIMIZATION RESULTS: {symbol}")
    print("=" * 80)
    print(f"  Objective:          {result.objective}")
    print(f"  Total Combinations: {result.total_combinations}")
    print(f"  Successful Runs:    {len(result.all_runs)}")
    print(f"  Runtime:            {result.runtime_seconds:.1f}s")
    print(f"  Best Score:         {result.best_score:.4f}")
    print("-" * 80)
    print(f"  Best Parameters:")
    for param, value in result.best_params.items():
        print(f"    {param}: {value}")
    print("=" * 80)

    # Print top 10 results
    top_n = min(10, len(result.all_runs))
    if top_n > 0:
        print(f"\n  Top {top_n} Results:")
        print(f"  {'Rank':<6} {'Score':>10} {'Return%':>10} {'WinRate%':>10} "
              f"{'Trades':>8} {'Sharpe':>10} {'MaxDD%':>10}")
        print(f"  {'-' * 64}")
        for i, run in enumerate(result.all_runs[:top_n], 1):
            r = run.result
            print(f"  {i:<6} {run.score:>10.4f} {r.total_return_pct:>10.2f} "
                  f"{r.win_rate:>10.1f} {r.total_trades:>8} "
                  f"{r.sharpe_ratio:>10.2f} {r.max_drawdown_pct:>10.2f}")
            # Print params on next line
            params_str = ", ".join(f"{k}={v}" for k, v in run.params.items())
            print(f"         Params: {params_str}")
        print()

    # Generate HTML report
    report_path = generate_optimization_report(
        result, output_path=f"reports/optimization_{symbol}.html",
    )
    print(f"  HTML report: {report_path}")

    return result


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    bars = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    available = list(PRESETS.keys())
    if symbol not in PRESETS:
        print(f"Available symbols: {', '.join(available)}")
        print(f"Using {symbol} with default price preset")

    total_combos = 1
    with open("backtest/optimization_config.yaml") as f:
        opt_config = yaml.safe_load(f)
    for p in opt_config["optimization"]["parameters"]:
        total_combos *= len(p["values"])

    print(f"\nRunning optimization: {symbol} | {bars} bars | "
          f"{total_combos} parameter combinations")

    asyncio.run(run_optimization(symbol, bars))


if __name__ == "__main__":
    main()
