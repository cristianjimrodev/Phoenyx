"""Technical analysis strategy combining S/R, patterns, indicators, and news."""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.analysis import indicators, support_resistance, patterns, news
from src.analysis.indicators import Signal
from src.strategy.base import Strategy, TradeSignal


class TechnicalStrategy(Strategy):
    """Multi-factor technical analysis strategy.

    Combines signals with weighted priorities:
    1. Support/Resistance & Trend Lines (highest weight)
    2. Chart Patterns
    3. Technical Indicators
    4. News Sentiment (lowest weight)
    """

    def __init__(self, config: dict):
        self._weights = config.get("weights", {
            "support_resistance": 0.35,
            "patterns": 0.30,
            "indicators": 0.25,
            "news": 0.10,
        })
        self._sr_config = config.get("support_resistance", {})
        self._pattern_config = config.get("patterns", {})
        self._indicator_config = config.get("indicators", {})
        self._news_config = config.get("news", {})

    async def evaluate(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        if df.empty or len(df) < 50:
            return TradeSignal(symbol, Signal.HOLD, 0, 0, 0, ["Insufficient data"])

        details = []
        weighted_score = 0.0  # positive = buy, negative = sell

        # 1. Support / Resistance Analysis
        sr_result = support_resistance.analyze(df, self._sr_config)
        sr_score = self._signal_to_score(sr_result.signal) * sr_result.confidence
        weighted_score += sr_score * self._weights["support_resistance"]
        details.append(f"[S/R] {sr_result.detail} (conf={sr_result.confidence:.0f})")

        # 2. Pattern Recognition
        pattern_matches = patterns.analyze(df, self._pattern_config)
        if pattern_matches:
            best_pattern = max(pattern_matches, key=lambda p: p.confidence)
            pat_score = self._signal_to_score(best_pattern.signal) * best_pattern.confidence
            weighted_score += pat_score * self._weights["patterns"]
            details.append(f"[Pattern] {best_pattern.detail} (conf={best_pattern.confidence:.0f})")
        else:
            details.append("[Pattern] No patterns detected")

        # 3. Technical Indicators
        df, ind_signals = indicators.compute_all(df, self._indicator_config)
        if ind_signals:
            buy_count = sum(1 for s in ind_signals if s.signal == Signal.BUY)
            sell_count = sum(1 for s in ind_signals if s.signal == Signal.SELL)
            total = len(ind_signals)

            if buy_count > sell_count:
                ind_confidence = (buy_count / total) * 100
                ind_score = ind_confidence
            elif sell_count > buy_count:
                ind_confidence = (sell_count / total) * 100
                ind_score = -ind_confidence
            else:
                ind_score = 0

            weighted_score += ind_score * self._weights["indicators"]
            ind_details = ", ".join(f"{s.name}={s.signal.value}" for s in ind_signals)
            details.append(f"[Indicators] {ind_details}")

        # 4. News Sentiment
        if self._news_config.get("enabled", False):
            news_result = await news.analyze(symbol, self._news_config)
            news_score = self._signal_to_score(news_result.signal) * news_result.confidence
            weighted_score += news_score * self._weights["news"]
            details.append(f"[News] {news_result.detail}")
        else:
            details.append("[News] Disabled")

        # Final signal
        if weighted_score > 0:
            final_signal = Signal.BUY
        elif weighted_score < 0:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD

        # Normalize confidence: the active weights sum determines the max possible score.
        # Scale so that if all active signals agree at 100%, confidence = 100.
        active_weight_sum = (
            self._weights["support_resistance"]
            + self._weights["patterns"]
            + self._weights["indicators"]
            + (self._weights["news"] if self._news_config.get("enabled", False) else 0)
        )
        max_possible = active_weight_sum * 100
        confidence = min(abs(weighted_score) / max_possible * 100, 100) if max_possible > 0 else 0

        # Calculate SL/TP from ATR
        sl, tp = self._compute_sl_tp(df, final_signal, sr_result.levels)

        logger.info(f"[{symbol}] Signal={final_signal.value} Confidence={confidence:.1f} "
                     f"Score={weighted_score:.1f}")

        return TradeSignal(symbol, final_signal, confidence, sl, tp, details)

    def _signal_to_score(self, signal: Signal) -> float:
        if signal == Signal.BUY:
            return 1.0
        elif signal == Signal.SELL:
            return -1.0
        return 0.0

    def _compute_sl_tp(self, df: pd.DataFrame, signal: Signal,
                       levels: list) -> tuple[float, float]:
        """Compute stop loss and take profit based on ATR and nearby S/R levels."""
        if "atr" not in df.columns or df["atr"].isna().all():
            return 0.0, 0.0

        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]

        if signal == Signal.BUY:
            sl = price - atr * 1.5
            tp = price + atr * 3.0

            # Adjust SL to nearest support below
            supports = [l.price for l in levels if l.is_support and l.price < price]
            if supports:
                nearest_support = max(supports)
                sl = min(sl, nearest_support - atr * 0.2)

        elif signal == Signal.SELL:
            sl = price + atr * 1.5
            tp = price - atr * 3.0

            # Adjust SL to nearest resistance above
            resistances = [l.price for l in levels if not l.is_support and l.price > price]
            if resistances:
                nearest_resistance = min(resistances)
                sl = max(sl, nearest_resistance + atr * 0.2)
        else:
            sl, tp = 0.0, 0.0

        return round(sl, 5), round(tp, 5)
