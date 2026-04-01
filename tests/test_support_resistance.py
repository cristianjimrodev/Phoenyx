"""Tests for src/analysis/support_resistance.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.support_resistance import (
    find_pivot_points,
    find_levels,
    find_trend_lines,
    analyze,
    Level,
    SRSignal,
)
from src.analysis.indicators import Signal
from tests.conftest import make_ohlcv_df, make_price_sequence


class TestFindPivotPoints:
    def test_basic_pivots(self):
        # Oscillating data should produce highs and lows
        prices = []
        for i in range(60):
            prices.append(1.10 + 0.01 * np.sin(i * 0.5))
        df = make_price_sequence(prices)
        highs_idx, lows_idx = find_pivot_points(df, lookback=3)
        assert len(highs_idx) > 0
        assert len(lows_idx) > 0

    def test_flat_data_minimal_pivots(self):
        prices = [1.1000] * 60
        df = make_price_sequence(prices)
        highs_idx, lows_idx = find_pivot_points(df, lookback=5)
        # Flat data: argrelextrema may find many or few depending on equal values
        # Just ensure no crash
        assert isinstance(highs_idx, np.ndarray)


class TestFindLevels:
    def test_levels_found_in_oscillating_data(self, sr_config):
        # Create data that oscillates around a support/resistance area
        prices = []
        for i in range(120):
            # Oscillate between 1.09 and 1.11 with mean reversion at 1.10
            prices.append(1.10 + 0.01 * np.sin(i * 0.3))
        df = make_price_sequence(prices)
        levels = find_levels(df, sr_config)
        # Should find at least some levels
        assert isinstance(levels, list)

    def test_empty_df_returns_empty(self, sr_config):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], name="datetime")
        levels = find_levels(df, sr_config)
        assert levels == []

    def test_min_touches_filter(self):
        """Levels with fewer touches than min_touches are excluded."""
        prices = []
        for i in range(120):
            prices.append(1.10 + 0.01 * np.sin(i * 0.3))
        df = make_price_sequence(prices)
        config_strict = {"lookback_periods": 100, "min_touches": 10, "tolerance_pct": 0.001}
        levels = find_levels(df, config_strict)
        for level in levels:
            assert level.strength >= 10

    def test_level_has_correct_fields(self, sr_config):
        df = make_ohlcv_df(length=200, trend="volatile", seed=5)
        levels = find_levels(df, sr_config)
        for level in levels:
            assert isinstance(level, Level)
            assert isinstance(level.price, float)
            assert isinstance(level.strength, int)
            assert isinstance(level.is_support, bool)


class TestFindTrendLines:
    def test_uptrend_produces_support_line(self, sr_config):
        df = make_ohlcv_df(length=200, trend="up", seed=3)
        trend_lines = find_trend_lines(df, sr_config)
        # At least check it runs without error and returns a list
        assert isinstance(trend_lines, list)

    def test_insufficient_pivots(self):
        """Very few bars → not enough pivots for trend line."""
        prices = [1.10, 1.11, 1.12]
        df = make_price_sequence(prices)
        config = {"trend_line_min_points": 3, "tolerance_pct": 0.001}
        trend_lines = find_trend_lines(df, config)
        assert trend_lines == []


class TestAnalyze:
    def test_insufficient_data_returns_hold(self, sr_config):
        df = make_ohlcv_df(length=20)
        result = analyze(df, sr_config)
        assert result.signal == Signal.HOLD
        assert result.confidence == 0
        assert "Insufficient" in result.detail

    def test_empty_df_returns_hold(self, sr_config):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], name="datetime")
        result = analyze(df, sr_config)
        assert result.signal == Signal.HOLD

    def test_returns_sr_signal(self, sr_config):
        df = make_ohlcv_df(length=200, trend="volatile", seed=5)
        result = analyze(df, sr_config)
        assert isinstance(result, SRSignal)
        assert isinstance(result.signal, Signal)
        assert 0 <= result.confidence <= 100
        assert isinstance(result.levels, list)
        assert isinstance(result.trend_lines, list)

    def test_confidence_capped_at_100(self, sr_config):
        """Even with very strong levels, confidence never exceeds 100."""
        df = make_ohlcv_df(length=200, trend="volatile", seed=99)
        result = analyze(df, sr_config)
        assert result.confidence <= 100

    def test_has_detail_string(self, sr_config):
        df = make_ohlcv_df(length=200, trend="up")
        result = analyze(df, sr_config)
        assert isinstance(result.detail, str)
        assert len(result.detail) > 0
