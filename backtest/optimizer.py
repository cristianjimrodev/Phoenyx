"""Grid search optimizer for strategy parameters."""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.strategy.technical import TechnicalStrategy
from backtest.engine import BacktestEngine, BacktestResult


@dataclass
class OptimizationParam:
    path: str          # dot-notation, e.g. "weights.support_resistance"
    values: list       # grid values to try


@dataclass
class OptimizationRun:
    params: dict       # {path: value} for this run
    result: BacktestResult
    score: float


@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    objective: str
    all_runs: list[OptimizationRun]
    total_combinations: int
    runtime_seconds: float


class GridSearchOptimizer:
    def __init__(self, base_config: dict, params: list[OptimizationParam],
                 objective: str = "sharpe_ratio",
                 initial_balance: float = 10000,
                 min_confidence: float = 60):
        self._base_config = base_config
        self._params = params
        self._objective = objective
        self._initial_balance = initial_balance
        self._min_confidence = min_confidence

    def _set_nested(self, d: dict, path: str, value) -> None:
        """Set a value in a nested dict using dot notation."""
        keys = path.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _get_score(self, result: BacktestResult) -> float:
        return getattr(result, self._objective, 0)

    async def run(self, df: pd.DataFrame, symbol: str,
                  lookback: int = 200) -> OptimizationResult:
        # Generate all combinations
        param_names = [p.path for p in self._params]
        param_values = [p.values for p in self._params]
        combinations = list(itertools.product(*param_values))

        logger.info(f"Optimizer: {len(combinations)} combinations, objective={self._objective}")

        start_time = time.time()
        all_runs: list[OptimizationRun] = []
        best_score = float("-inf")
        best_params = {}

        for i, combo in enumerate(combinations):
            config = copy.deepcopy(self._base_config)
            param_dict = {}
            for name, val in zip(param_names, combo):
                self._set_nested(config, name, val)
                param_dict[name] = val

            strategy = TechnicalStrategy(config)
            engine = BacktestEngine(
                strategy, self._initial_balance,
                risk_per_trade=0.02, min_confidence=self._min_confidence,
            )

            try:
                result = await engine.run(df, symbol, lookback=lookback)
                score = self._get_score(result)

                run = OptimizationRun(params=param_dict, result=result, score=score)
                all_runs.append(run)

                if score > best_score:
                    best_score = score
                    best_params = param_dict.copy()

                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i+1}/{len(combinations)}, best {self._objective}={best_score:.4f}")
            except Exception as e:
                logger.warning(f"  Combination {i+1} failed: {e}")

        runtime = time.time() - start_time
        all_runs.sort(key=lambda r: r.score, reverse=True)

        logger.info(f"Optimization complete: {len(all_runs)} successful runs in {runtime:.1f}s")
        logger.info(f"Best {self._objective}={best_score:.4f} with params={best_params}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            objective=self._objective,
            all_runs=all_runs,
            total_combinations=len(combinations),
            runtime_seconds=runtime,
        )
