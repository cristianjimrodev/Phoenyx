"""Top-down multi-timeframe strategy with per-asset parameter sets.

1. W1 (Weekly):  Global trend direction + strong S/R levels
2. D1 (Daily):   Intermediate trend confirmation
3. H4 (4-Hour):  Entry/exit signals using RSI(14) + Stochastic(5,3,3)

Parameters (RSI thresholds, Stochastic zones, ATR multipliers, timeframes)
are loaded per-symbol from ``config/assets.yaml``.  Anything not specified
falls back to the ``default`` block.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import ta
import yaml
from loguru import logger

from src.analysis import support_resistance, patterns
from src.analysis.indicators import Signal
from src.strategy.base import Strategy, TradeSignal


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_asset_params(symbol: str, assets_path: str = "config/assets.yaml") -> dict:
    """Load parameters for *symbol*, merging with the default block."""
    path = Path(assets_path)
    if not path.exists():
        return {}

    with open(path) as f:
        all_cfg = yaml.safe_load(f) or {}

    defaults = all_cfg.get("default", {})
    overrides = all_cfg.get(symbol.upper(), all_cfg.get(symbol, {}))
    return _deep_merge(defaults, overrides)


class TopDownStrategy(Strategy):
    """Top-down strategy with per-asset parameter sets."""

    def __init__(self, config: dict, asset_params: dict | None = None):
        self._base_config = config
        self._sr_config = config.get("support_resistance", {})
        # Asset params can be injected or left empty (will be loaded per symbol)
        self._asset_cache: dict[str, dict] = {}
        if asset_params:
            self._asset_cache["_injected"] = asset_params

    def _get_params(self, symbol: str) -> dict:
        """Get merged params for symbol (cached)."""
        if "_injected" in self._asset_cache:
            return self._asset_cache["_injected"]
        if symbol not in self._asset_cache:
            self._asset_cache[symbol] = load_asset_params(symbol)
        return self._asset_cache[symbol]

    async def evaluate(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        """Single-timeframe fallback."""
        if df.empty or len(df) < 50:
            return TradeSignal(symbol, Signal.HOLD, 0, 0, 0, ["Insufficient data"])
        p = self._get_params(symbol)
        entry_signal, confidence, details = self._analyze_entry(df, p)
        sl, tp = self._compute_sl_tp(df, entry_signal, p)
        return TradeSignal(symbol, entry_signal, confidence, sl, tp, details)

    async def evaluate_mtf(self, symbol: str,
                           dataframes: dict[str, pd.DataFrame]) -> TradeSignal:
        """Top-down evaluation using per-asset parameters."""
        p = self._get_params(symbol)
        if p.get("excluded", False):
            return TradeSignal(symbol, Signal.HOLD, 0, 0, 0, [f"{symbol} excluded"])
        details: list[str] = []

        tf_cfg = p.get("timeframes", {})
        trend_tf = tf_cfg.get("trend", "W1")
        confirm_tf = tf_cfg.get("confirmation", "D1")
        entry_tf = tf_cfg.get("entry", "H4")

        entry_df = dataframes.get(entry_tf)
        trend_df = dataframes.get(trend_tf)
        confirm_df = dataframes.get(confirm_tf)

        if entry_df is None or entry_df.empty or len(entry_df) < 50:
            return TradeSignal(symbol, Signal.HOLD, 0, 0, 0,
                               [f"{entry_tf} data insufficient"])

        details.append(f"[Config] {symbol}: {trend_tf} -> {confirm_tf} -> {entry_tf}")

        # ── Step 1: Trend (W1) ─────────────────────────────────
        w1_trend = Signal.HOLD
        w1_sr_levels = []

        if trend_df is not None and len(trend_df) >= 50:
            w1_trend, _, w1_details, w1_sr_levels = self._analyze_trend(trend_df, trend_tf)
            details.extend(w1_details)
        else:
            details.append(f"[{trend_tf}] Insufficient data -- skipping trend filter")

        # ── Step 2: Confirmation (D1) ──────────────────────────
        d1_trend = Signal.HOLD
        d1_sr_levels = []

        if confirm_df is not None and len(confirm_df) >= 50:
            d1_trend, _, d1_details, d1_sr_levels = self._analyze_trend(confirm_df, confirm_tf)
            details.extend(d1_details)
        else:
            details.append(f"[{confirm_tf}] Insufficient data -- skipping confirmation")

        # ── Step 2b: S/R bounce zone detection ─────────────────
        sr_bounce_enabled = p.get("sr_bounce_entry", True)
        all_sr = w1_sr_levels + d1_sr_levels
        current_price = entry_df["close"].iloc[-1]
        sr_bounce = None
        if sr_bounce_enabled and all_sr:
            sr_bounce = self._detect_sr_bounce(current_price, all_sr, entry_df, p)
            if sr_bounce:
                details.extend(sr_bounce["details"])

        # ── Step 2c: RSI divergence detection ──────────────────
        divergence_enabled = p.get("divergence_entry", True)
        divergence = None
        if divergence_enabled:
            divergence = self._detect_rsi_divergence(entry_df, p)
            if divergence:
                details.extend(divergence["details"])

        # ── Step 3: Entry signal ───────────────────────────────
        entry_signal, entry_confidence, entry_details = self._analyze_entry(entry_df, p)
        details.extend(entry_details)

        # Count confluence factors
        confluence = 0
        confluence_details = []

        # S/R zone boost
        if sr_bounce and sr_bounce["signal"] != Signal.HOLD:
            if entry_signal != Signal.HOLD and sr_bounce["signal"] == entry_signal:
                entry_confidence = min(100, entry_confidence + sr_bounce["bonus"])
                confluence += 1
                confluence_details.append("S/R zone")
            elif entry_signal == Signal.HOLD and sr_bounce["strong"]:
                entry_signal = sr_bounce["signal"]
                entry_confidence = sr_bounce["confidence"]
                confluence += 1
                confluence_details.append("strong S/R bounce")

        # Divergence boost
        if divergence and divergence["signal"] != Signal.HOLD:
            if entry_signal != Signal.HOLD and divergence["signal"] == entry_signal:
                entry_confidence = min(100, entry_confidence + 15)
                confluence += 1
                confluence_details.append("RSI divergence")
            elif entry_signal == Signal.HOLD:
                entry_signal = divergence["signal"]
                entry_confidence = divergence["confidence"]
                confluence += 1
                confluence_details.append("RSI divergence entry")

        # H4 pattern boost (already computed in _analyze_entry)
        # Check if patterns in entry_details match signal
        pattern_confirms = any("[Pattern]" in d and entry_signal.value in d.lower()
                               for d in entry_details if entry_signal != Signal.HOLD)
        if pattern_confirms:
            confluence += 1
            confluence_details.append("chart pattern")

        if confluence_details:
            details.append(f"[Confluence] {confluence} factors: {', '.join(confluence_details)}")

        # ── Combine ────────────────────────────────────────────
        final_signal = Signal.HOLD
        final_confidence = 0.0

        # Long-only filter
        long_only = p.get("long_only", False)
        if long_only and entry_signal == Signal.SELL:
            details.append(f"[Decision] SELL blocked (long_only=true for {symbol})")
            entry_signal = Signal.HOLD

        if entry_signal == Signal.HOLD:
            details.append(f"[Decision] {entry_tf} no entry signal -> HOLD")

        elif w1_trend != Signal.HOLD and w1_trend != entry_signal:
            # Normally block against weekly trend, BUT allow if high confluence
            # (S/R bounce + divergence + pattern = strong counter-trend setup)
            if confluence >= 2 and sr_bounce and divergence:
                final_signal = entry_signal
                final_confidence = min(100, max(0, entry_confidence - 10))
                details.append(
                    f"[Decision] COUNTER-TREND {entry_signal.value.upper()} allowed: "
                    f"{confluence} confluence factors at HTF S/R zone "
                    f"(conf={final_confidence:.0f}%)")
            else:
                details.append(
                    f"[Decision] {entry_tf}={entry_signal.value} vs "
                    f"{trend_tf}={w1_trend.value} -> HOLD (against weekly trend)")

        else:
            final_signal = entry_signal
            bonus = 0
            penalty = 0
            agreeing = []

            if w1_trend == entry_signal:
                bonus += 15
                agreeing.append(trend_tf)

            if d1_trend == entry_signal:
                bonus += 10
                agreeing.append(confirm_tf)
            elif d1_trend != Signal.HOLD and d1_trend != entry_signal:
                penalty = 10
                details.append(
                    f"[Decision] {confirm_tf}={d1_trend.value} disagrees "
                    f"-> confidence penalty (-{penalty})")

            final_confidence = min(100, max(0, entry_confidence + bonus - penalty))
            details.append(
                f"[Decision] {entry_signal.value.upper()} "
                f"{'confirmed by ' + ', '.join(agreeing) if agreeing else 'from ' + entry_tf + ' only'} "
                f"(conf={final_confidence:.0f}%)")

        sl, tp = self._compute_sl_tp(entry_df, final_signal, p, w1_sr_levels)
        return TradeSignal(symbol, final_signal, final_confidence, sl, tp, details)

    # ─────────────────── Internal ──────────────────────────────

    def _detect_rsi_divergence(self, df: pd.DataFrame, p: dict,
                               lookback: int = 20) -> dict | None:
        """Detect RSI divergence (bullish or bearish).

        Bullish divergence: price makes lower low, RSI makes higher low (in oversold)
        Bearish divergence: price makes higher high, RSI makes lower high (in overbought)

        Returns dict with signal, confidence, details or None.
        """
        rsi_period = p.get("rsi", {}).get("period", 14)
        rsi_os = p.get("rsi", {}).get("oversold", 35)
        rsi_ob = p.get("rsi", {}).get("overbought", 65)

        df = df.copy()
        df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_period)

        if len(df) < lookback + rsi_period or df["rsi"].isna().all():
            return None

        recent = df.iloc[-lookback:]
        rsi_vals = recent["rsi"].values
        close_vals = recent["close"].values

        if any(pd.isna(rsi_vals)):
            return None

        current_rsi = rsi_vals[-1]
        current_price = close_vals[-1]

        # Find the lowest low and its RSI in the lookback window
        min_price_idx = close_vals.argmin()
        max_price_idx = close_vals.argmax()

        # ── Bullish divergence (price lower low, RSI higher low) ──
        if min_price_idx < len(close_vals) - 3:  # not at the very end
            price_at_low = close_vals[min_price_idx]
            rsi_at_low = rsi_vals[min_price_idx]

            # Current price near or below that low, but RSI is higher
            if (current_price <= price_at_low * 1.005 and
                    current_rsi > rsi_at_low + 3 and
                    current_rsi < rsi_os + 10):  # RSI still in oversold zone
                confidence = min(75, 45 + (current_rsi - rsi_at_low) * 2)
                return {
                    "signal": Signal.BUY,
                    "confidence": confidence,
                    "details": [
                        f"[Divergence] BULLISH: price near low {price_at_low:.5f}, "
                        f"RSI {rsi_at_low:.1f}->{current_rsi:.1f} (higher low)"
                    ],
                }

        # ── Bearish divergence (price higher high, RSI lower high) ──
        if max_price_idx < len(close_vals) - 3:
            price_at_high = close_vals[max_price_idx]
            rsi_at_high = rsi_vals[max_price_idx]

            if (current_price >= price_at_high * 0.995 and
                    current_rsi < rsi_at_high - 3 and
                    current_rsi > rsi_ob - 10):
                confidence = min(75, 45 + (rsi_at_high - current_rsi) * 2)
                return {
                    "signal": Signal.SELL,
                    "confidence": confidence,
                    "details": [
                        f"[Divergence] BEARISH: price near high {price_at_high:.5f}, "
                        f"RSI {rsi_at_high:.1f}->{current_rsi:.1f} (lower high)"
                    ],
                }

        return None

    def _detect_sr_bounce(self, price: float, sr_levels: list,
                          entry_df: pd.DataFrame, p: dict) -> dict | None:
        """Detect if price is near a strong HTF S/R zone with bounce potential.

        Looks for:
        - Price within 0.5% of a strong W1/D1 level
        - Confirming candle pattern (wick rejection, engulfing)
        - RSI/Stoch in favorable zone for bounce direction

        Returns dict with signal, bonus, confidence, details, strong flag.
        """
        if not sr_levels or len(entry_df) < 3:
            return None

        tolerance = 0.005  # 0.5% proximity to level
        details = []
        best_bounce = None

        for level in sr_levels:
            dist_pct = abs(price - level.price) / level.price

            if dist_pct > tolerance:
                continue

            # Price is near this S/R level
            is_support = level.is_support

            # Check for wick rejection (candle body away from level, wick touches it)
            last_candle = entry_df.iloc[-1]
            prev_candle = entry_df.iloc[-2]
            body_size = abs(last_candle["close"] - last_candle["open"])
            candle_range = last_candle["high"] - last_candle["low"]

            has_rejection = False
            if is_support:
                # Lower wick should be long (tested support and bounced)
                lower_wick = min(last_candle["open"], last_candle["close"]) - last_candle["low"]
                has_rejection = candle_range > 0 and lower_wick / candle_range > 0.5
            else:
                # Upper wick should be long (tested resistance and rejected)
                upper_wick = last_candle["high"] - max(last_candle["open"], last_candle["close"])
                has_rejection = candle_range > 0 and upper_wick / candle_range > 0.5

            # Check if price bounced (current close vs previous close)
            bouncing_up = last_candle["close"] > prev_candle["close"]
            bouncing_down = last_candle["close"] < prev_candle["close"]

            bounce_signal = Signal.HOLD
            bonus = 0
            confidence = 0
            strong = False

            if is_support and (bouncing_up or has_rejection):
                bounce_signal = Signal.BUY
                bonus = 15 + level.strength * 5
                confidence = min(70, 40 + level.strength * 10)
                strong = has_rejection and level.strength >= 3
                details.append(
                    f"[S/R Zone] Near HTF support {level.price:.5f} "
                    f"(str={level.strength}, dist={dist_pct:.3%})"
                    f"{' + wick rejection' if has_rejection else ''}"
                    f"{' + price bouncing' if bouncing_up else ''}")

            elif not is_support and (bouncing_down or has_rejection):
                bounce_signal = Signal.SELL
                bonus = 15 + level.strength * 5
                confidence = min(70, 40 + level.strength * 10)
                strong = has_rejection and level.strength >= 3
                details.append(
                    f"[S/R Zone] Near HTF resistance {level.price:.5f} "
                    f"(str={level.strength}, dist={dist_pct:.3%})"
                    f"{' + wick rejection' if has_rejection else ''}"
                    f"{' + price rejecting' if bouncing_down else ''}")

            if bounce_signal != Signal.HOLD:
                if best_bounce is None or bonus > best_bounce["bonus"]:
                    best_bounce = {
                        "signal": bounce_signal,
                        "bonus": bonus,
                        "confidence": confidence,
                        "strong": strong,
                        "details": details,
                    }

        return best_bounce

    def _analyze_trend(self, df: pd.DataFrame, label: str
                       ) -> tuple[Signal, float, list[str], list]:
        details: list[str] = []
        df = df.copy()
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)

        price = df["close"].iloc[-1]
        sma20 = df["sma_20"].iloc[-1]
        sma50 = df["sma_50"].iloc[-1]

        if pd.isna(sma20) or pd.isna(sma50):
            return Signal.HOLD, 0, [f"[{label}] MAs not ready"], []

        # ── MA trend ──
        if sma20 > sma50 and price > sma20:
            ma_trend = Signal.BUY
            strength = "strong uptrend" if price > sma20 * 1.005 else "uptrend"
        elif sma20 < sma50 and price < sma20:
            ma_trend = Signal.SELL
            strength = "strong downtrend" if price < sma20 * 0.995 else "downtrend"
        else:
            ma_trend = Signal.HOLD
            strength = "ranging/mixed"

        sma_sep = abs(sma20 - sma50) / sma50 * 100
        ma_confidence = min(90, 40 + sma_sep * 10)
        details.append(
            f"[{label}] MA Trend={ma_trend.value} ({strength}), "
            f"Price={price:.5f}, SMA20={sma20:.5f}, SMA50={sma50:.5f}")

        # ── Chart patterns ──
        pattern_config = {"zigzag_threshold": 0.04, "enabled": []}  # wider zigzag for HTF
        if label == "W1":
            pattern_config["zigzag_threshold"] = 0.05  # even wider for weekly
        detected = patterns.analyze(df, pattern_config)

        pattern_signal = Signal.HOLD
        pattern_confidence = 0
        if detected:
            # Count bullish vs bearish patterns
            bull_pats = [p for p in detected if p.signal == Signal.BUY]
            bear_pats = [p for p in detected if p.signal == Signal.SELL]

            for p in detected:
                details.append(f"[{label}] Pattern: {p.name} -> {p.signal.value} "
                               f"(conf={p.confidence:.0f}, target={p.target_price:.5f})")

            # Only assign pattern signal if there's a clear majority
            if bull_pats and not bear_pats:
                best = max(bull_pats, key=lambda p: p.confidence)
                pattern_signal = Signal.BUY
                pattern_confidence = best.confidence
            elif bear_pats and not bull_pats:
                best = max(bear_pats, key=lambda p: p.confidence)
                pattern_signal = Signal.SELL
                pattern_confidence = best.confidence
            else:
                # Contradictory patterns -> stay neutral
                details.append(f"[{label}] Contradictory patterns ({len(bull_pats)} bull vs {len(bear_pats)} bear) -> neutral")
        else:
            details.append(f"[{label}] No chart patterns detected")

        # ── S/R levels ──
        sr_levels = support_resistance.find_levels(df, self._sr_config)
        sr_signal = Signal.HOLD
        sr_confidence = 0

        if sr_levels:
            sups = [l for l in sr_levels if l.is_support]
            ress = [l for l in sr_levels if not l.is_support]
            details.append(f"[{label}] S/R: {len(sups)} supports, {len(ress)} resistances")

            # Check proximity to key levels
            tolerance = self._sr_config.get("tolerance_pct", 0.001) * 5  # wider for HTF
            for level in sr_levels:
                dist = (price - level.price) / level.price
                if abs(dist) < tolerance:
                    if level.is_support:
                        sr_signal = Signal.BUY
                        sr_confidence = min(80, 30 + level.strength * 15)
                        details.append(f"[{label}] Near strong support {level.price:.5f} (touches={level.strength})")
                    else:
                        sr_signal = Signal.SELL
                        sr_confidence = min(80, 30 + level.strength * 15)
                        details.append(f"[{label}] Near strong resistance {level.price:.5f} (touches={level.strength})")

        # ── Combine: MA trend is primary, patterns/S/R can reinforce or override ──
        final_trend = ma_trend
        final_confidence = ma_confidence

        # Pattern confirmation: if pattern agrees with MA, boost confidence
        if pattern_signal != Signal.HOLD:
            if pattern_signal == ma_trend:
                final_confidence = min(95, final_confidence + pattern_confidence * 0.3)
                details.append(f"[{label}] Pattern confirms MA trend -> boosted")
            elif ma_trend == Signal.HOLD:
                # MA is neutral but pattern gives direction
                final_trend = pattern_signal
                final_confidence = pattern_confidence * 0.7
                details.append(f"[{label}] Pattern overrides ranging MA -> {pattern_signal.value}")

        # S/R reinforcement
        if sr_signal != Signal.HOLD and sr_signal == final_trend:
            final_confidence = min(95, final_confidence + sr_confidence * 0.2)

        return final_trend, final_confidence, details, sr_levels

    def _analyze_entry(self, df: pd.DataFrame, p: dict
                       ) -> tuple[Signal, float, list[str]]:
        details: list[str] = []
        df = df.copy()

        entry_tf = p.get("timeframes", {}).get("entry", "H4")

        # ── Chart patterns on entry TF ──
        pattern_config = {"zigzag_threshold": 0.03, "enabled": []}
        detected = patterns.analyze(df, pattern_config)
        pattern_signal = Signal.HOLD
        pattern_confidence = 0
        self._last_pattern_target = 0.0  # store for TP calculation
        if detected:
            best = max(detected, key=lambda p_: p_.confidence)
            pattern_signal = best.signal
            pattern_confidence = best.confidence
            self._last_pattern_target = best.target_price
            for pat in detected:
                details.append(f"[{entry_tf}] Pattern: {pat.name} -> {pat.signal.value} "
                               f"(conf={pat.confidence:.0f}, target={pat.target_price:.5f})")

        rsi_cfg = p.get("rsi", {})
        stoch_cfg = p.get("stochastic", {})

        rsi_period = rsi_cfg.get("period", 14)
        rsi_os = rsi_cfg.get("oversold", 35)
        rsi_ob = rsi_cfg.get("overbought", 65)
        rsi_deep_os = rsi_cfg.get("deep_oversold", 25)
        rsi_deep_ob = rsi_cfg.get("deep_overbought", 75)

        k_period = stoch_cfg.get("k_period", 5)
        d_period = stoch_cfg.get("d_period", 3)
        smooth = stoch_cfg.get("smooth", 3)
        stoch_os = stoch_cfg.get("oversold", 25)
        stoch_ob = stoch_cfg.get("overbought", 75)

        # Compute indicators
        df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_period)
        df["stoch_k"] = ta.momentum.stoch(
            df["high"], df["low"], df["close"],
            window=k_period, smooth_window=smooth)
        df["stoch_d"] = ta.momentum.stoch_signal(
            df["high"], df["low"], df["close"],
            window=k_period, smooth_window=d_period)

        rsi = df["rsi"].iloc[-1]
        rsi_prev = df["rsi"].iloc[-2] if len(df) > 1 else rsi
        stoch_k = df["stoch_k"].iloc[-1]
        stoch_d = df["stoch_d"].iloc[-1]
        stoch_k_prev = df["stoch_k"].iloc[-2] if len(df) > 1 else stoch_k
        stoch_d_prev = df["stoch_d"].iloc[-2] if len(df) > 1 else stoch_d

        if pd.isna(rsi) or pd.isna(stoch_k):
            return Signal.HOLD, 0, ["Indicators not ready"]

        entry_tf = p.get("timeframes", {}).get("entry", "H4")
        details.append(
            f"[{entry_tf}] RSI({rsi_period})={rsi:.1f} "
            f"[OS<{rsi_os}, OB>{rsi_ob}], "
            f"Stoch(%K={stoch_k:.1f}, %D={stoch_d:.1f}) "
            f"[OS<{stoch_os}, OB>{stoch_ob}]")

        # ── BUY ──
        rsi_oversold = rsi < rsi_os
        rsi_recovering = rsi_prev < rsi_deep_os and rsi > rsi_prev
        stoch_buy_cross = stoch_k_prev < stoch_d_prev and stoch_k > stoch_d
        stoch_deep_os_cond = stoch_k < stoch_os and stoch_d < stoch_os

        # ── SELL ──
        rsi_overbought = rsi > rsi_ob
        rsi_declining = rsi_prev > rsi_deep_ob and rsi < rsi_prev
        stoch_sell_cross = stoch_k_prev > stoch_d_prev and stoch_k < stoch_d
        stoch_deep_ob_cond = stoch_k > stoch_ob and stoch_d > stoch_ob

        signal = Signal.HOLD
        confidence = 0.0

        if stoch_buy_cross and (rsi_oversold or rsi_recovering) and stoch_deep_os_cond:
            signal = Signal.BUY
            confidence = 60
            if rsi < rsi_deep_os:
                confidence += 15
            if stoch_k < stoch_os * 0.6:
                confidence += 10
            if rsi_recovering:
                confidence += 5
            details.append(
                f"[{entry_tf}] BUY: RSI={rsi:.1f} + Stoch crossover "
                f"K={stoch_k:.1f} in deep oversold")

        elif stoch_sell_cross and (rsi_overbought or rsi_declining) and stoch_deep_ob_cond:
            signal = Signal.SELL
            confidence = 60
            if rsi > rsi_deep_ob:
                confidence += 15
            if stoch_k > stoch_ob + (100 - stoch_ob) * 0.4:
                confidence += 10
            if rsi_declining:
                confidence += 5
            details.append(
                f"[{entry_tf}] SELL: RSI={rsi:.1f} + Stoch crossover "
                f"K={stoch_k:.1f} in deep overbought")

        else:
            details.append(f"[{entry_tf}] No entry signal")

        # Boost confidence if H4 pattern agrees with indicator signal
        if signal != Signal.HOLD and pattern_signal == signal:
            confidence = min(100, confidence + pattern_confidence * 0.2)
            details.append(f"[{entry_tf}] Pattern confirms indicator signal -> boosted")

        return signal, min(confidence, 100), details

    def _compute_sl_tp(self, df: pd.DataFrame, signal: Signal,
                       p: dict, htf_sr_levels: list | None = None
                       ) -> tuple[float, float]:
        atr_cfg = p.get("atr", {})
        atr_period = atr_cfg.get("period", 14)
        sl_mult = atr_cfg.get("sl_multiplier", 1.5)
        tp_mult = atr_cfg.get("tp_multiplier", 3.0)

        df = df.copy()
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=atr_period)

        if df["atr"].isna().all() or signal == Signal.HOLD:
            return 0.0, 0.0

        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        pattern_target = getattr(self, "_last_pattern_target", 0.0)

        if signal == Signal.BUY:
            sl = price - atr * sl_mult

            # TP: use pattern target if available and better than ATR-based
            tp_atr = price + atr * tp_mult
            if pattern_target > price and pattern_target > tp_atr:
                tp = pattern_target
            else:
                tp = tp_atr

            # Adjust SL to HTF support
            if htf_sr_levels:
                supports = [l.price for l in htf_sr_levels
                            if l.is_support and l.price < price and l.price > sl]
                if supports:
                    sl = max(supports) - atr * 0.2

        elif signal == Signal.SELL:
            sl = price + atr * sl_mult

            tp_atr = price - atr * tp_mult
            if pattern_target > 0 and pattern_target < price and pattern_target < tp_atr:
                tp = pattern_target
            else:
                tp = tp_atr

            if htf_sr_levels:
                resistances = [l.price for l in htf_sr_levels
                               if not l.is_support and l.price > price and l.price < sl]
                if resistances:
                    sl = min(resistances) + atr * 0.2
        else:
            return 0.0, 0.0

        # Ensure minimum 2:1 R:R ratio
        sl_distance = abs(price - sl)
        tp_distance = abs(tp - price)
        if sl_distance > 0 and tp_distance / sl_distance < 2.0:
            # Adjust TP to maintain at least 2:1
            if signal == Signal.BUY:
                tp = price + sl_distance * 2.0
            else:
                tp = price - sl_distance * 2.0

        return round(sl, 5), round(tp, 5)
