"""Chart pattern recognition (double top/bottom, H&S, triangles, flags, wedges)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from loguru import logger

from src.analysis.indicators import Signal


@dataclass
class PatternMatch:
    name: str
    signal: Signal
    confidence: float    # 0-100
    start_idx: int
    end_idx: int
    target_price: float  # projected price target
    detail: str


def _zigzag(df: pd.DataFrame, threshold: float = 0.03) -> list[tuple[int, float, bool]]:
    """Compute zigzag pivots. Returns list of (index, price, is_high)."""
    prices = df["close"].values
    if len(prices) < 5:
        return []

    pivots = []
    last_pivot_price = prices[0]
    last_pivot_idx = 0
    last_is_high = True

    for i in range(1, len(prices)):
        change = (prices[i] - last_pivot_price) / last_pivot_price

        if last_is_high and change < -threshold:
            pivots.append((last_pivot_idx, last_pivot_price, True))
            last_pivot_price = prices[i]
            last_pivot_idx = i
            last_is_high = False
        elif not last_is_high and change > threshold:
            pivots.append((last_pivot_idx, last_pivot_price, False))
            last_pivot_price = prices[i]
            last_pivot_idx = i
            last_is_high = True
        elif last_is_high and prices[i] > last_pivot_price:
            last_pivot_price = prices[i]
            last_pivot_idx = i
        elif not last_is_high and prices[i] < last_pivot_price:
            last_pivot_price = prices[i]
            last_pivot_idx = i

    pivots.append((last_pivot_idx, last_pivot_price, last_is_high))
    return pivots


def _detect_double_top(pivots: list, tolerance: float = 0.015) -> PatternMatch | None:
    """Detect double top pattern."""
    highs = [(idx, price) for idx, price, is_high in pivots if is_high]
    if len(highs) < 2:
        return None

    for i in range(len(highs) - 1):
        idx1, p1 = highs[i]
        idx2, p2 = highs[i + 1]
        if abs(p1 - p2) / p1 <= tolerance and idx2 - idx1 >= 5:
            # Find the valley between them
            valleys = [(idx, price) for idx, price, is_high in pivots
                       if not is_high and idx1 < idx < idx2]
            if valleys:
                neckline = valleys[0][1]
                target = neckline - (p1 - neckline)
                return PatternMatch(
                    name="double_top",
                    signal=Signal.SELL,
                    confidence=70,
                    start_idx=idx1,
                    end_idx=idx2,
                    target_price=target,
                    detail=f"Double top at {p1:.5f}/{p2:.5f}, neckline={neckline:.5f}, target={target:.5f}",
                )
    return None


def _detect_double_bottom(pivots: list, tolerance: float = 0.015) -> PatternMatch | None:
    """Detect double bottom pattern."""
    lows = [(idx, price) for idx, price, is_high in pivots if not is_high]
    if len(lows) < 2:
        return None

    for i in range(len(lows) - 1):
        idx1, p1 = lows[i]
        idx2, p2 = lows[i + 1]
        if abs(p1 - p2) / p1 <= tolerance and idx2 - idx1 >= 5:
            peaks = [(idx, price) for idx, price, is_high in pivots
                     if is_high and idx1 < idx < idx2]
            if peaks:
                neckline = peaks[0][1]
                target = neckline + (neckline - p1)
                return PatternMatch(
                    name="double_bottom",
                    signal=Signal.BUY,
                    confidence=70,
                    start_idx=idx1,
                    end_idx=idx2,
                    target_price=target,
                    detail=f"Double bottom at {p1:.5f}/{p2:.5f}, neckline={neckline:.5f}, target={target:.5f}",
                )
    return None


def _detect_head_and_shoulders(pivots: list, tolerance: float = 0.015) -> PatternMatch | None:
    """Detect head and shoulders (bearish)."""
    highs = [(idx, price) for idx, price, is_high in pivots if is_high]
    if len(highs) < 3:
        return None

    for i in range(len(highs) - 2):
        ls_idx, ls = highs[i]       # left shoulder
        h_idx, h = highs[i + 1]     # head
        rs_idx, rs = highs[i + 2]   # right shoulder

        # Head must be highest, shoulders roughly equal
        if h > ls and h > rs and abs(ls - rs) / ls <= tolerance:
            # Find neckline from valleys
            valleys = [(idx, price) for idx, price, is_high in pivots
                       if not is_high and ls_idx < idx < rs_idx]
            if len(valleys) >= 2:
                neckline = (valleys[0][1] + valleys[-1][1]) / 2
                target = neckline - (h - neckline)
                return PatternMatch(
                    name="head_and_shoulders",
                    signal=Signal.SELL,
                    confidence=80,
                    start_idx=ls_idx,
                    end_idx=rs_idx,
                    target_price=target,
                    detail=f"H&S: LS={ls:.5f}, Head={h:.5f}, RS={rs:.5f}, target={target:.5f}",
                )
    return None


def _detect_inverse_head_and_shoulders(pivots: list, tolerance: float = 0.015) -> PatternMatch | None:
    """Detect inverse head and shoulders (bullish)."""
    lows = [(idx, price) for idx, price, is_high in pivots if not is_high]
    if len(lows) < 3:
        return None

    for i in range(len(lows) - 2):
        ls_idx, ls = lows[i]
        h_idx, h = lows[i + 1]
        rs_idx, rs = lows[i + 2]

        if h < ls and h < rs and abs(ls - rs) / ls <= tolerance:
            peaks = [(idx, price) for idx, price, is_high in pivots
                     if is_high and ls_idx < idx < rs_idx]
            if len(peaks) >= 2:
                neckline = (peaks[0][1] + peaks[-1][1]) / 2
                target = neckline + (neckline - h)
                return PatternMatch(
                    name="inverse_head_and_shoulders",
                    signal=Signal.BUY,
                    confidence=80,
                    start_idx=ls_idx,
                    end_idx=rs_idx,
                    target_price=target,
                    detail=f"Inv H&S: LS={ls:.5f}, Head={h:.5f}, RS={rs:.5f}, target={target:.5f}",
                )
    return None


def _detect_triangle(pivots: list) -> PatternMatch | None:
    """Detect ascending, descending, and symmetric triangles."""
    if len(pivots) < 4:
        return None

    highs = [(idx, price) for idx, price, is_high in pivots if is_high]
    lows = [(idx, price) for idx, price, is_high in pivots if not is_high]

    if len(highs) < 2 or len(lows) < 2:
        return None

    # Trend of highs and lows
    high_prices = [p for _, p in highs[-4:]]
    low_prices = [p for _, p in lows[-4:]]

    highs_flat = all(abs(high_prices[i] - high_prices[0]) / high_prices[0] < 0.01
                     for i in range(len(high_prices)))
    lows_flat = all(abs(low_prices[i] - low_prices[0]) / low_prices[0] < 0.01
                    for i in range(len(low_prices)))
    highs_falling = all(high_prices[i] >= high_prices[i + 1] for i in range(len(high_prices) - 1))
    lows_rising = all(low_prices[i] <= low_prices[i + 1] for i in range(len(low_prices) - 1))

    start_idx = min(highs[-4][0] if len(highs) >= 4 else highs[0][0],
                    lows[-4][0] if len(lows) >= 4 else lows[0][0])
    end_idx = max(highs[-1][0], lows[-1][0])

    if highs_flat and lows_rising:
        height = high_prices[-1] - low_prices[0]
        return PatternMatch("ascending_triangle", Signal.BUY, 65, start_idx, end_idx,
                            high_prices[-1] + height, "Ascending triangle - bullish breakout expected")

    if lows_flat and highs_falling:
        height = high_prices[0] - low_prices[-1]
        return PatternMatch("descending_triangle", Signal.SELL, 65, start_idx, end_idx,
                            low_prices[-1] - height, "Descending triangle - bearish breakdown expected")

    if highs_falling and lows_rising:
        return PatternMatch("symmetric_triangle", Signal.HOLD, 50, start_idx, end_idx,
                            0, "Symmetric triangle - breakout direction uncertain")

    return None


def analyze(df: pd.DataFrame, config: dict) -> list[PatternMatch]:
    """Run all pattern detectors and return matches."""
    threshold = config.get("zigzag_threshold", 0.03)
    enabled = config.get("enabled", [])

    pivots = _zigzag(df, threshold)
    if len(pivots) < 4:
        return []

    patterns: list[PatternMatch] = []
    detectors = {
        "double_top": _detect_double_top,
        "double_bottom": _detect_double_bottom,
        "head_and_shoulders": _detect_head_and_shoulders,
        "inverse_head_and_shoulders": _detect_inverse_head_and_shoulders,
    }

    for name, detector in detectors.items():
        if enabled and name not in enabled:
            continue
        result = detector(pivots)
        if result:
            patterns.append(result)

    # Triangle detection
    triangle_types = {"ascending_triangle", "descending_triangle", "symmetric_triangle"}
    if not enabled or triangle_types & set(enabled):
        result = _detect_triangle(pivots)
        if result:
            patterns.append(result)

    if patterns:
        logger.info(f"Detected patterns: {[p.name for p in patterns]}")

    return patterns
