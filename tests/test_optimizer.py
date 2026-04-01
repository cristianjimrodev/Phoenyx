"""Tests for backtest/optimizer.py"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from backtest.engine import BacktestResult, BacktestTrade
from backtest.optimizer import (
    GridSearchOptimizer,
    OptimizationParam,
    OptimizationResult,
    OptimizationRun,
)
from tests.conftest import make_ohlcv_df


def _make_backtest_result(sharpe: float = 1.0, total_return: float = 5.0,
                          win_rate: float = 60.0, total_trades: int = 10) -> BacktestResult:
    """Create a BacktestResult with controllable metrics."""
    return BacktestResult(
        trades=[],
        total_return_pct=total_return,
        win_rate=win_rate,
        total_trades=total_trades,
        winning_trades=int(total_trades * win_rate / 100),
        losing_trades=total_trades - int(total_trades * win_rate / 100),
        max_drawdown_pct=3.0,
        sharpe_ratio=sharpe,
        avg_win=10.0,
        avg_loss=-5.0,
        profit_factor=2.0,
    )


class TestSetNested:
    """Tests for _set_nested helper."""

    def test_single_level(self):
        optimizer = GridSearchOptimizer(
            base_config={}, params=[], objective="sharpe_ratio",
        )
        d: dict = {}
        optimizer._set_nested(d, "key", 42)
        assert d == {"key": 42}

    def test_two_levels(self):
        optimizer = GridSearchOptimizer(
            base_config={}, params=[], objective="sharpe_ratio",
        )
        d: dict = {}
        optimizer._set_nested(d, "a.b", 99)
        assert d == {"a": {"b": 99}}

    def test_three_levels(self):
        optimizer = GridSearchOptimizer(
            base_config={}, params=[], objective="sharpe_ratio",
        )
        d: dict = {}
        optimizer._set_nested(d, "x.y.z", "hello")
        assert d == {"x": {"y": {"z": "hello"}}}

    def test_preserves_existing_keys(self):
        optimizer = GridSearchOptimizer(
            base_config={}, params=[], objective="sharpe_ratio",
        )
        d: dict = {"a": {"existing": 1}}
        optimizer._set_nested(d, "a.new_key", 2)
        assert d == {"a": {"existing": 1, "new_key": 2}}

    def test_overwrites_value(self):
        optimizer = GridSearchOptimizer(
            base_config={}, params=[], objective="sharpe_ratio",
        )
        d: dict = {"a": {"b": "old"}}
        optimizer._set_nested(d, "a.b", "new")
        assert d == {"a": {"b": "new"}}


class TestGridSearchCombinations:
    """Tests that the optimizer runs the correct number of combinations."""

    @pytest.mark.asyncio
    async def test_2x2_runs_4_combinations(self):
        """Two params with 2 values each should produce 4 runs."""
        params = [
            OptimizationParam(path="weights.support_resistance", values=[0.30, 0.35]),
            OptimizationParam(path="weights.patterns", values=[0.25, 0.30]),
        ]

        base_config = {
            "weights": {"support_resistance": 0.35, "patterns": 0.30,
                        "indicators": 0.25, "news": 0.10},
        }

        optimizer = GridSearchOptimizer(
            base_config=base_config,
            params=params,
            objective="sharpe_ratio",
            initial_balance=10000,
            min_confidence=60,
        )

        df = make_ohlcv_df(length=250, trend="up")
        mock_result = _make_backtest_result(sharpe=1.5)

        with patch("backtest.optimizer.BacktestEngine") as MockEngine, \
             patch("backtest.optimizer.TechnicalStrategy"):
            mock_engine_inst = AsyncMock()
            mock_engine_inst.run = AsyncMock(return_value=mock_result)
            MockEngine.return_value = mock_engine_inst

            result = await optimizer.run(df, "EURUSD", lookback=200)

        assert result.total_combinations == 4
        assert len(result.all_runs) == 4

    @pytest.mark.asyncio
    async def test_3x2_runs_6_combinations(self):
        """Three values x two values = 6 runs."""
        params = [
            OptimizationParam(path="indicators.macd.fast", values=[8, 10, 12]),
            OptimizationParam(path="indicators.macd.slow", values=[21, 26]),
        ]

        optimizer = GridSearchOptimizer(
            base_config={}, params=params, objective="sharpe_ratio",
        )

        df = make_ohlcv_df(length=250, trend="flat")
        mock_result = _make_backtest_result(sharpe=0.5)

        with patch("backtest.optimizer.BacktestEngine") as MockEngine, \
             patch("backtest.optimizer.TechnicalStrategy"):
            mock_engine_inst = AsyncMock()
            mock_engine_inst.run = AsyncMock(return_value=mock_result)
            MockEngine.return_value = mock_engine_inst

            result = await optimizer.run(df, "EURUSD", lookback=200)

        assert result.total_combinations == 6
        assert len(result.all_runs) == 6


class TestBestResultIdentification:
    """Tests that the optimizer correctly identifies the best run."""

    @pytest.mark.asyncio
    async def test_best_score_selected(self):
        """The combination with the highest score should be identified as best."""
        params = [
            OptimizationParam(path="weights.support_resistance", values=[0.25, 0.35, 0.45]),
        ]

        optimizer = GridSearchOptimizer(
            base_config={}, params=params, objective="sharpe_ratio",
        )

        df = make_ohlcv_df(length=250, trend="up")

        # Return different sharpe ratios for different configs
        call_count = 0
        sharpe_values = [0.5, 2.5, 1.0]  # Best is index 1 (value=0.35)

        async def mock_run(df_arg, symbol, lookback=200):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return _make_backtest_result(sharpe=sharpe_values[idx])

        with patch("backtest.optimizer.BacktestEngine") as MockEngine, \
             patch("backtest.optimizer.TechnicalStrategy"):
            mock_engine_inst = AsyncMock()
            mock_engine_inst.run = AsyncMock(side_effect=mock_run)
            MockEngine.return_value = mock_engine_inst

            result = await optimizer.run(df, "EURUSD", lookback=200)

        assert result.best_score == 2.5
        assert result.best_params == {"weights.support_resistance": 0.35}
        # all_runs should be sorted descending
        assert result.all_runs[0].score == 2.5
        assert result.all_runs[-1].score == 0.5

    @pytest.mark.asyncio
    async def test_objective_attribute_used(self):
        """The optimizer should use the specified objective attribute."""
        params = [
            OptimizationParam(path="weights.indicators", values=[0.20, 0.30]),
        ]

        optimizer = GridSearchOptimizer(
            base_config={}, params=params, objective="total_return_pct",
        )

        df = make_ohlcv_df(length=250, trend="up")

        call_count = 0

        async def mock_run(df_arg, symbol, lookback=200):
            nonlocal call_count
            idx = call_count
            call_count += 1
            # First combo: low return, high sharpe; second: high return, low sharpe
            if idx == 0:
                return _make_backtest_result(sharpe=3.0, total_return=2.0)
            return _make_backtest_result(sharpe=0.5, total_return=15.0)

        with patch("backtest.optimizer.BacktestEngine") as MockEngine, \
             patch("backtest.optimizer.TechnicalStrategy"):
            mock_engine_inst = AsyncMock()
            mock_engine_inst.run = AsyncMock(side_effect=mock_run)
            MockEngine.return_value = mock_engine_inst

            result = await optimizer.run(df, "EURUSD", lookback=200)

        assert result.objective == "total_return_pct"
        assert result.best_score == 15.0
        assert result.best_params == {"weights.indicators": 0.30}


class TestErrorHandling:
    """Tests that failing combinations are handled gracefully."""

    @pytest.mark.asyncio
    async def test_failing_combination_skipped(self):
        """If one combination raises an exception, it should be skipped."""
        params = [
            OptimizationParam(path="weights.support_resistance", values=[0.25, 0.35, 0.45]),
        ]

        optimizer = GridSearchOptimizer(
            base_config={}, params=params, objective="sharpe_ratio",
        )

        df = make_ohlcv_df(length=250, trend="up")

        call_count = 0

        async def mock_run(df_arg, symbol, lookback=200):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 1:
                raise RuntimeError("Simulated failure")
            return _make_backtest_result(sharpe=1.0 + idx)

        with patch("backtest.optimizer.BacktestEngine") as MockEngine, \
             patch("backtest.optimizer.TechnicalStrategy"):
            mock_engine_inst = AsyncMock()
            mock_engine_inst.run = AsyncMock(side_effect=mock_run)
            MockEngine.return_value = mock_engine_inst

            result = await optimizer.run(df, "EURUSD", lookback=200)

        # 3 total combinations, 1 failed, so 2 successful
        assert result.total_combinations == 3
        assert len(result.all_runs) == 2

    @pytest.mark.asyncio
    async def test_all_combinations_fail(self):
        """If every combination fails, the result should have no runs."""
        params = [
            OptimizationParam(path="weights.support_resistance", values=[0.25, 0.35]),
        ]

        optimizer = GridSearchOptimizer(
            base_config={}, params=params, objective="sharpe_ratio",
        )

        df = make_ohlcv_df(length=250, trend="up")

        with patch("backtest.optimizer.BacktestEngine") as MockEngine, \
             patch("backtest.optimizer.TechnicalStrategy"):
            mock_engine_inst = AsyncMock()
            mock_engine_inst.run = AsyncMock(side_effect=RuntimeError("Always fails"))
            MockEngine.return_value = mock_engine_inst

            result = await optimizer.run(df, "EURUSD", lookback=200)

        assert result.total_combinations == 2
        assert len(result.all_runs) == 0
        assert result.best_score == float("-inf")
        assert result.best_params == {}

    @pytest.mark.asyncio
    async def test_empty_params_runs_one_combination(self):
        """With no optimization params, there should be exactly 1 combination (the base config)."""
        optimizer = GridSearchOptimizer(
            base_config={"weights": {"support_resistance": 0.35}},
            params=[],
            objective="sharpe_ratio",
        )

        df = make_ohlcv_df(length=250, trend="up")
        mock_result = _make_backtest_result(sharpe=1.0)

        with patch("backtest.optimizer.BacktestEngine") as MockEngine, \
             patch("backtest.optimizer.TechnicalStrategy"):
            mock_engine_inst = AsyncMock()
            mock_engine_inst.run = AsyncMock(return_value=mock_result)
            MockEngine.return_value = mock_engine_inst

            result = await optimizer.run(df, "EURUSD", lookback=200)

        assert result.total_combinations == 1
        assert len(result.all_runs) == 1
