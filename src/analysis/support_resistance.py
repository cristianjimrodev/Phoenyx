"""Detection of support/resistance levels and trend lines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from loguru import logger

from src.analysis.indicators import Signal


@dataclass
class Level:
    price: float
    strength: int      # number of touches
    is_support: bool   # True=support, False=resistance


@dataclass
class TrendLine:
    slope: float
    intercept: float
    start_idx: int
    end_idx: int
    touches: int
    is_support: bool


@dataclass
class SRSignal:
    signal: Signal
    confidence: float   # 0-100
    detail: str
    levels: list[Level]
    trend_lines: list[TrendLine]


def find_pivot_points(df: pd.DataFrame, lookback: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Find local highs and lows using argrelextrema."""
    highs_idx = argrelextrema(df["high"].values, np.greater_equal, order=lookback)[0]
    lows_idx = argrelextrema(df["low"].values, np.less_equal, order=lookback)[0]
    return highs_idx, lows_idx


def find_levels(df: pd.DataFrame, config: dict) -> list[Level]:
    """Detect horizontal support and resistance levels."""
    lookback = config.get("lookback_periods", 100)
    min_touches = config.get("min_touches", 2)
    tolerance = config.get("tolerance_pct", 0.001)

    data = df.tail(lookback) if len(df) > lookback else df
    highs_idx, lows_idx = find_pivot_points(data)

    # Collect all pivot prices
    pivot_prices = []
    for i in highs_idx:
        pivot_prices.append((data["high"].iloc[i], False))  # resistance
    for i in lows_idx:
        pivot_prices.append((data["low"].iloc[i], True))    # support

    if not pivot_prices:
        return []

    # Cluster nearby levels
    levels: list[Level] = []
    pivot_prices.sort(key=lambda x: x[0])

    used = [False] * len(pivot_prices)
    for i, (price, is_sup) in enumerate(pivot_prices):
        if used[i]:
            continue

        cluster = [price]
        cluster_support = is_sup
        used[i] = True

        for j in range(i + 1, len(pivot_prices)):
            if used[j]:
                continue
            if abs(pivot_prices[j][0] - price) / price <= tolerance:
                cluster.append(pivot_prices[j][0])
                used[j] = True

        if len(cluster) >= min_touches:
            avg_price = float(np.mean(cluster))
            levels.append(Level(
                price=avg_price,
                strength=len(cluster),
                is_support=cluster_support,
            ))

    logger.debug(f"Found {len(levels)} S/R levels")
    return levels


def find_trend_lines(df: pd.DataFrame, config: dict) -> list[TrendLine]:
    """Detect trend lines by connecting pivot points with linear regression."""
    min_points = config.get("trend_line_min_points", 3)
    tolerance = config.get("tolerance_pct", 0.001)

    highs_idx, lows_idx = find_pivot_points(df)
    trend_lines: list[TrendLine] = []

    # Support trend lines (connecting lows)
    if len(lows_idx) >= min_points:
        tl = _fit_trend_line(df, lows_idx, is_support=True, tolerance=tolerance, min_points=min_points)
        if tl:
            trend_lines.append(tl)

    # Resistance trend lines (connecting highs)
    if len(highs_idx) >= min_points:
        tl = _fit_trend_line(df, highs_idx, is_support=False, tolerance=tolerance, min_points=min_points)
        if tl:
            trend_lines.append(tl)

    return trend_lines


def _fit_trend_line(df: pd.DataFrame, indices: np.ndarray, is_support: bool,
                    tolerance: float, min_points: int) -> TrendLine | None:
    """Fit a trend line through pivot points using least-squares."""
    prices = df["low"].values[indices] if is_support else df["high"].values[indices]

    if len(indices) < 2:
        return None

    # Linear regression: price = slope * index + intercept
    x = indices.astype(float)
    coeffs = np.polyfit(x, prices, 1)
    slope, intercept = coeffs

    # Count points near the line
    fitted = slope * x + intercept
    errors = np.abs(prices - fitted) / prices
    touches = int(np.sum(errors <= tolerance))

    if touches >= min_points:
        return TrendLine(
            slope=float(slope),
            intercept=float(intercept),
            start_idx=int(indices[0]),
            end_idx=int(indices[-1]),
            touches=touches,
            is_support=is_support,
        )
    return None


def analyze(df: pd.DataFrame, config: dict) -> SRSignal:
    """Run full support/resistance analysis and generate a signal."""
    if df.empty or len(df) < 30:
        return SRSignal(Signal.HOLD, 0, "Insufficient data for S/R analysis", [], [])

    levels = find_levels(df, config)
    trend_lines = find_trend_lines(df, config)

    current_price = df["close"].iloc[-1]
    tolerance = config.get("tolerance_pct", 0.001)

    signal = Signal.HOLD
    confidence = 0.0
    details = []

    # Check proximity to S/R levels
    for level in levels:
        distance_pct = (current_price - level.price) / level.price

        if abs(distance_pct) <= tolerance * 3:  # Near a level
            if level.is_support:
                signal = Signal.BUY
                confidence = min(100, 30 + level.strength * 15)
                details.append(f"Near support {level.price:.5f} (touches={level.strength})")
            else:
                signal = Signal.SELL
                confidence = min(100, 30 + level.strength * 15)
                details.append(f"Near resistance {level.price:.5f} (touches={level.strength})")

    # Check trend line proximity
    for tl in trend_lines:
        last_idx = len(df) - 1
        tl_price = tl.slope * last_idx + tl.intercept
        distance_pct = (current_price - tl_price) / tl_price

        if abs(distance_pct) <= tolerance * 3:
            if tl.is_support and tl.slope > 0:
                signal = Signal.BUY
                confidence = max(confidence, 40 + tl.touches * 10)
                details.append(f"Near ascending support trendline (touches={tl.touches})")
            elif not tl.is_support and tl.slope < 0:
                signal = Signal.SELL
                confidence = max(confidence, 40 + tl.touches * 10)
                details.append(f"Near descending resistance trendline (touches={tl.touches})")

    # Breakout detection
    for level in levels:
        prev_price = df["close"].iloc[-2]
        if level.is_support and prev_price > level.price and current_price < level.price:
            signal = Signal.SELL
            confidence = max(confidence, 50 + level.strength * 10)
            details.append(f"BREAKDOWN below support {level.price:.5f}")
        elif not level.is_support and prev_price < level.price and current_price > level.price:
            signal = Signal.BUY
            confidence = max(confidence, 50 + level.strength * 10)
            details.append(f"BREAKOUT above resistance {level.price:.5f}")

    detail_str = "; ".join(details) if details else "No significant S/R interaction"
    return SRSignal(signal, min(confidence, 100), detail_str, levels, trend_lines)
