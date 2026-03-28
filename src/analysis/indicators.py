"""Technical indicators using the `ta` library."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import ta
from loguru import logger


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class IndicatorSignal:
    name: str
    signal: Signal
    value: float
    detail: str


def compute_all(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[IndicatorSignal]]:
    """Add all indicators to DataFrame and generate signals.

    Returns the enriched DataFrame and a list of signals.
    """
    if df.empty or len(df) < 50:
        return df, []

    df = df.copy()
    signals: list[IndicatorSignal] = []

    # --- RSI ---
    rsi_cfg = config.get("rsi", {})
    rsi_period = rsi_cfg.get("period", 14)
    overbought = rsi_cfg.get("overbought", 70)
    oversold = rsi_cfg.get("oversold", 30)

    df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_period)
    rsi_val = df["rsi"].iloc[-1]

    if rsi_val <= oversold:
        signals.append(IndicatorSignal("RSI", Signal.BUY, rsi_val, f"RSI={rsi_val:.1f} (oversold)"))
    elif rsi_val >= overbought:
        signals.append(IndicatorSignal("RSI", Signal.SELL, rsi_val, f"RSI={rsi_val:.1f} (overbought)"))
    else:
        signals.append(IndicatorSignal("RSI", Signal.HOLD, rsi_val, f"RSI={rsi_val:.1f} (neutral)"))

    # --- MACD ---
    macd_cfg = config.get("macd", {})
    fast = macd_cfg.get("fast", 12)
    slow = macd_cfg.get("slow", 26)
    signal_period = macd_cfg.get("signal", 9)

    macd_ind = ta.trend.MACD(df["close"], window_slow=slow, window_fast=fast, window_sign=signal_period)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    macd_val = df["macd"].iloc[-1]
    macd_sig = df["macd_signal"].iloc[-1]
    macd_prev = df["macd"].iloc[-2]
    macd_sig_prev = df["macd_signal"].iloc[-2]

    if macd_prev < macd_sig_prev and macd_val > macd_sig:
        signals.append(IndicatorSignal("MACD", Signal.BUY, macd_val, "MACD bullish crossover"))
    elif macd_prev > macd_sig_prev and macd_val < macd_sig:
        signals.append(IndicatorSignal("MACD", Signal.SELL, macd_val, "MACD bearish crossover"))
    else:
        signals.append(IndicatorSignal("MACD", Signal.HOLD, macd_val, "MACD no crossover"))

    # --- Moving Averages ---
    ma_cfg = config.get("moving_averages", {})
    ma_fast = ma_cfg.get("fast", 20)
    ma_slow = ma_cfg.get("slow", 50)
    ma_long = ma_cfg.get("long", 200)

    df[f"sma_{ma_fast}"] = ta.trend.sma_indicator(df["close"], window=ma_fast)
    df[f"sma_{ma_slow}"] = ta.trend.sma_indicator(df["close"], window=ma_slow)
    df[f"sma_{ma_long}"] = ta.trend.sma_indicator(df["close"], window=ma_long)

    price = df["close"].iloc[-1]
    sma_fast_val = df[f"sma_{ma_fast}"].iloc[-1]
    sma_slow_val = df[f"sma_{ma_slow}"].iloc[-1]

    if sma_fast_val > sma_slow_val and price > sma_fast_val:
        signals.append(IndicatorSignal("MA", Signal.BUY, price, f"Price above fast SMA, fast > slow"))
    elif sma_fast_val < sma_slow_val and price < sma_fast_val:
        signals.append(IndicatorSignal("MA", Signal.SELL, price, f"Price below fast SMA, fast < slow"))
    else:
        signals.append(IndicatorSignal("MA", Signal.HOLD, price, "MAs mixed"))

    # --- Bollinger Bands ---
    bb_cfg = config.get("bollinger", {})
    bb_period = bb_cfg.get("period", 20)
    bb_std = bb_cfg.get("std_dev", 2)

    bb = ta.volatility.BollingerBands(df["close"], window=bb_period, window_dev=bb_std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    bb_upper = df["bb_upper"].iloc[-1]
    bb_lower = df["bb_lower"].iloc[-1]

    if price <= bb_lower:
        signals.append(IndicatorSignal("BB", Signal.BUY, price, "Price at lower Bollinger Band"))
    elif price >= bb_upper:
        signals.append(IndicatorSignal("BB", Signal.SELL, price, "Price at upper Bollinger Band"))
    else:
        signals.append(IndicatorSignal("BB", Signal.HOLD, price, "Price within Bollinger Bands"))

    # --- ATR (for risk management, not signal) ---
    atr_cfg = config.get("atr", {})
    atr_period = atr_cfg.get("period", 14)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)

    # --- Stochastic ---
    df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)

    stoch_k = df["stoch_k"].iloc[-1]
    if stoch_k < 20:
        signals.append(IndicatorSignal("Stochastic", Signal.BUY, stoch_k, f"Stoch={stoch_k:.1f} oversold"))
    elif stoch_k > 80:
        signals.append(IndicatorSignal("Stochastic", Signal.SELL, stoch_k, f"Stoch={stoch_k:.1f} overbought"))
    else:
        signals.append(IndicatorSignal("Stochastic", Signal.HOLD, stoch_k, f"Stoch={stoch_k:.1f} neutral"))

    return df, signals
