"""Tests for src/analysis/patterns.py"""
from __future__ import annotations

import pytest

from src.analysis.patterns import (
    _zigzag,
    _detect_double_top,
    _detect_double_bottom,
    _detect_head_and_shoulders,
    _detect_inverse_head_and_shoulders,
    _detect_triangle,
    analyze,
    PatternMatch,
)
from src.analysis.indicators import Signal
from tests.conftest import make_ohlcv_df, make_price_sequence


# ───────────── Zigzag tests ─────────────

class TestZigzag:
    def test_basic_up_down_up(self):
        # Clear up-down-up sequence with >3% moves
        prices = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05,
                  1.04, 1.03, 1.02, 1.01,
                  1.02, 1.03, 1.04, 1.05, 1.06]
        df = make_price_sequence(prices)
        pivots = _zigzag(df, threshold=0.03)
        assert len(pivots) >= 2

    def test_flat_returns_minimal_pivots(self):
        prices = [1.10] * 20
        df = make_price_sequence(prices)
        pivots = _zigzag(df, threshold=0.03)
        # Flat → no moves above threshold, so just the last appended pivot
        assert len(pivots) >= 1

    def test_short_input_returns_empty(self):
        prices = [1.10, 1.11]
        df = make_price_sequence(prices)
        pivots = _zigzag(df, threshold=0.03)
        assert pivots == []

    def test_pivots_are_tuples(self):
        df = make_ohlcv_df(length=200, trend="volatile", seed=5)
        pivots = _zigzag(df, threshold=0.03)
        for p in pivots:
            assert len(p) == 3
            idx, price, is_high = p
            assert isinstance(idx, int)
            assert isinstance(price, float)
            assert isinstance(is_high, bool)

    def test_large_threshold_fewer_pivots(self):
        df = make_ohlcv_df(length=200, trend="volatile", seed=5)
        pivots_small = _zigzag(df, threshold=0.01)
        pivots_large = _zigzag(df, threshold=0.10)
        assert len(pivots_large) <= len(pivots_small)


# ───────────── Pattern detector unit tests (hand-crafted pivots) ─────────────

class TestDetectDoubleTop:
    def test_valid_double_top(self):
        # Two highs at similar price, valley between, far enough apart
        pivots = [
            (0, 1.1500, True),
            (5, 1.1400, False),
            (12, 1.1505, True),   # within 1.5% tolerance of 1.1500
            (18, 1.1350, False),
        ]
        result = _detect_double_top(pivots, tolerance=0.015)
        assert result is not None
        assert result.name == "double_top"
        assert result.signal == Signal.SELL
        assert result.confidence == 70
        # Target = neckline - (peak - neckline) = 1.14 - (1.15 - 1.14) = 1.13
        assert result.target_price == pytest.approx(1.1300, abs=0.002)

    def test_outside_tolerance_returns_none(self):
        pivots = [
            (0, 1.1500, True),
            (5, 1.1400, False),
            (12, 1.2000, True),   # 4.3% difference > 1.5%
            (18, 1.1350, False),
        ]
        result = _detect_double_top(pivots, tolerance=0.015)
        assert result is None

    def test_too_close_returns_none(self):
        # Two highs only 3 bars apart (< 5)
        pivots = [
            (10, 1.1500, True),
            (11, 1.1400, False),
            (13, 1.1505, True),
            (18, 1.1350, False),
        ]
        result = _detect_double_top(pivots, tolerance=0.015)
        assert result is None

    def test_insufficient_highs_returns_none(self):
        pivots = [
            (0, 1.1500, True),
            (5, 1.1400, False),
        ]
        result = _detect_double_top(pivots, tolerance=0.015)
        assert result is None

    def test_no_valley_between_returns_none(self):
        # Two highs with no valley between
        pivots = [
            (0, 1.1500, True),
            (10, 1.1505, True),
        ]
        result = _detect_double_top(pivots, tolerance=0.015)
        assert result is None


class TestDetectDoubleBottom:
    def test_valid_double_bottom(self):
        pivots = [
            (0, 1.1000, False),
            (6, 1.1100, True),
            (13, 1.1005, False),   # within tolerance
            (20, 1.1150, True),
        ]
        result = _detect_double_bottom(pivots, tolerance=0.015)
        assert result is not None
        assert result.name == "double_bottom"
        assert result.signal == Signal.BUY
        assert result.confidence == 70
        # Target = neckline + (neckline - low) = 1.11 + (1.11 - 1.10) = 1.12
        assert result.target_price == pytest.approx(1.1200, abs=0.002)

    def test_insufficient_lows_returns_none(self):
        pivots = [
            (0, 1.1000, False),
            (5, 1.1100, True),
        ]
        result = _detect_double_bottom(pivots, tolerance=0.015)
        assert result is None


class TestDetectHeadAndShoulders:
    def test_valid_hs(self):
        pivots = [
            (0, 1.1400, True),    # left shoulder
            (5, 1.1300, False),   # valley 1
            (10, 1.1600, True),   # head (highest)
            (15, 1.1300, False),  # valley 2
            (20, 1.1410, True),   # right shoulder (~= left)
        ]
        result = _detect_head_and_shoulders(pivots, tolerance=0.015)
        assert result is not None
        assert result.name == "head_and_shoulders"
        assert result.signal == Signal.SELL
        assert result.confidence == 80

    def test_head_not_highest_returns_none(self):
        pivots = [
            (0, 1.1500, True),    # left shoulder higher than head
            (5, 1.1300, False),
            (10, 1.1400, True),   # "head" is lower
            (15, 1.1300, False),
            (20, 1.1500, True),
        ]
        result = _detect_head_and_shoulders(pivots, tolerance=0.015)
        assert result is None

    def test_insufficient_highs(self):
        pivots = [
            (0, 1.1400, True),
            (5, 1.1300, False),
            (10, 1.1600, True),
        ]
        result = _detect_head_and_shoulders(pivots, tolerance=0.015)
        assert result is None

    def test_shoulders_not_equal_returns_none(self):
        pivots = [
            (0, 1.1200, True),    # left shoulder
            (5, 1.1100, False),
            (10, 1.1500, True),   # head
            (15, 1.1100, False),
            (20, 1.1450, True),   # right shoulder far from left (1.14 vs 1.12 = 1.8%)
        ]
        result = _detect_head_and_shoulders(pivots, tolerance=0.01)
        assert result is None


class TestDetectInverseHeadAndShoulders:
    def test_valid_inverse_hs(self):
        pivots = [
            (0, 1.1100, False),   # left shoulder
            (5, 1.1200, True),    # peak 1
            (10, 1.0900, False),  # head (lowest)
            (15, 1.1200, True),   # peak 2
            (20, 1.1105, False),  # right shoulder (~= left)
        ]
        result = _detect_inverse_head_and_shoulders(pivots, tolerance=0.015)
        assert result is not None
        assert result.name == "inverse_head_and_shoulders"
        assert result.signal == Signal.BUY
        assert result.confidence == 80

    def test_head_not_lowest_returns_none(self):
        pivots = [
            (0, 1.0900, False),   # left shoulder is lowest
            (5, 1.1200, True),
            (10, 1.1000, False),  # "head" not the lowest
            (15, 1.1200, True),
            (20, 1.0900, False),
        ]
        result = _detect_inverse_head_and_shoulders(pivots, tolerance=0.015)
        assert result is None


class TestDetectTriangle:
    def test_ascending_triangle(self):
        # Flat highs, rising lows
        pivots = [
            (0, 1.1500, True),
            (3, 1.1300, False),
            (6, 1.1505, True),    # flat high
            (9, 1.1350, False),   # rising low
            (12, 1.1502, True),   # flat high
            (15, 1.1400, False),  # rising low
        ]
        result = _detect_triangle(pivots)
        assert result is not None
        assert result.name == "ascending_triangle"
        assert result.signal == Signal.BUY
        assert result.confidence == 65

    def test_descending_triangle(self):
        # Falling highs, flat lows
        pivots = [
            (0, 1.1500, True),
            (3, 1.1300, False),
            (6, 1.1400, True),    # falling high
            (9, 1.1305, False),   # flat low
            (12, 1.1350, True),   # falling high
            (15, 1.1302, False),  # flat low
        ]
        result = _detect_triangle(pivots)
        assert result is not None
        assert result.name == "descending_triangle"
        assert result.signal == Signal.SELL

    def test_symmetric_triangle(self):
        # Falling highs, rising lows (converging)
        pivots = [
            (0, 1.1600, True),
            (3, 1.1200, False),
            (6, 1.1500, True),    # falling
            (9, 1.1300, False),   # rising
            (12, 1.1450, True),   # falling
            (15, 1.1350, False),  # rising
        ]
        result = _detect_triangle(pivots)
        assert result is not None
        assert result.name == "symmetric_triangle"
        assert result.signal == Signal.HOLD

    def test_insufficient_pivots(self):
        pivots = [
            (0, 1.15, True),
            (5, 1.13, False),
        ]
        result = _detect_triangle(pivots)
        assert result is None


# ───────────── analyze() integration tests ─────────────

class TestAnalyze:
    def test_few_pivots_returns_empty(self, pattern_config):
        prices = [1.10] * 10  # flat → minimal pivots
        df = make_price_sequence(prices)
        result = analyze(df, pattern_config)
        assert result == []

    def test_enabled_filter(self, pattern_config):
        """Only enabled patterns should be returned."""
        df = make_ohlcv_df(length=300, trend="volatile", seed=5)
        config = {**pattern_config, "enabled": ["double_top"]}
        result = analyze(df, config)
        for pat in result:
            assert pat.name == "double_top"

    def test_empty_enabled_runs_all_detectors(self, pattern_config):
        """enabled=[] means run all detectors."""
        df = make_ohlcv_df(length=300, trend="volatile", seed=5)
        config = {**pattern_config, "enabled": []}
        result = analyze(df, config)
        assert isinstance(result, list)

    def test_return_types(self, pattern_config):
        df = make_ohlcv_df(length=300, trend="volatile", seed=5)
        result = analyze(df, pattern_config)
        for pat in result:
            assert isinstance(pat, PatternMatch)
            assert isinstance(pat.signal, Signal)
            assert 0 <= pat.confidence <= 100
