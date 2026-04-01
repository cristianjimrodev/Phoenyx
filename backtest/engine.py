"""Backtesting engine to test strategies on historical data."""

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.analysis.indicators import Signal
from src.strategy.base import Strategy


@dataclass
class BacktestTrade:
    entry_idx: int
    exit_idx: int
    side: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    pnl_pct: float
    confidence: float
    reason: str


@dataclass
class BacktestResult:
    trades: list[BacktestTrade]
    total_return_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class BacktestEngine:
    """Simulates strategy execution on historical data."""

    def __init__(self, strategy: Strategy, initial_balance: float = 10000,
                 risk_per_trade: float = 0.02, min_confidence: float = 60):
        self._strategy = strategy
        self._initial_balance = initial_balance
        self._risk_per_trade = risk_per_trade
        self._min_confidence = min_confidence

    async def run(self, df: pd.DataFrame, symbol: str,
                  lookback: int = 100) -> BacktestResult:
        """Run backtest over the DataFrame.

        Args:
            df: Full historical OHLCV data.
            symbol: Symbol name.
            lookback: Minimum bars needed before evaluating.
        """
        trades: list[BacktestTrade] = []
        balance = self._initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        in_trade = False
        trade_entry = None

        logger.info(f"Backtesting {symbol} on {len(df)} bars, lookback={lookback}")

        for i in range(lookback, len(df)):
            window = df.iloc[:i + 1]

            if in_trade and trade_entry:
                # Check if SL or TP hit
                current_high = df["high"].iloc[i]
                current_low = df["low"].iloc[i]

                hit_sl = False
                hit_tp = False

                if trade_entry["side"] == "buy":
                    hit_sl = current_low <= trade_entry["sl"] and trade_entry["sl"] > 0
                    hit_tp = current_high >= trade_entry["tp"] and trade_entry["tp"] > 0
                else:
                    hit_sl = current_high >= trade_entry["sl"] and trade_entry["sl"] > 0
                    hit_tp = current_low <= trade_entry["tp"] and trade_entry["tp"] > 0

                if hit_sl or hit_tp:
                    exit_price = trade_entry["sl"] if hit_sl else trade_entry["tp"]
                    if trade_entry["side"] == "buy":
                        pnl_pct = (exit_price - trade_entry["price"]) / trade_entry["price"]
                    else:
                        pnl_pct = (trade_entry["price"] - exit_price) / trade_entry["price"]

                    pnl = balance * self._risk_per_trade * (pnl_pct / abs(
                        (trade_entry["price"] - trade_entry["sl"]) / trade_entry["price"]
                    )) if trade_entry["sl"] != trade_entry["price"] else 0

                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, dd)

                    trades.append(BacktestTrade(
                        entry_idx=trade_entry["idx"],
                        exit_idx=i,
                        side=trade_entry["side"],
                        entry_price=trade_entry["price"],
                        exit_price=exit_price,
                        sl=trade_entry["sl"],
                        tp=trade_entry["tp"],
                        pnl=pnl,
                        pnl_pct=pnl_pct * 100,
                        confidence=trade_entry["confidence"],
                        reason="SL hit" if hit_sl else "TP hit",
                    ))
                    in_trade = False
                    trade_entry = None
                continue

            # Evaluate strategy
            signal = await self._strategy.evaluate(symbol, window)

            if signal.signal == Signal.HOLD or signal.confidence < self._min_confidence:
                continue

            if signal.suggested_sl == 0:
                continue

            # Enter trade
            entry_price = df["close"].iloc[i]
            trade_entry = {
                "idx": i,
                "price": entry_price,
                "side": signal.signal.value,
                "sl": signal.suggested_sl,
                "tp": signal.suggested_tp,
                "confidence": signal.confidence,
            }
            in_trade = True

        # Compute stats
        total_return = ((balance - self._initial_balance) / self._initial_balance) * 100
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = (len(winning) / len(trades) * 100) if trades else 0

        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Simplified Sharpe (annualized from trade returns)
        if trades:
            returns = [t.pnl_pct for t in trades]
            import numpy as np
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0
        else:
            sharpe = 0

        result = BacktestResult(
            trades=trades,
            total_return_pct=total_return,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
        )

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                     f"return={result.total_return_pct:.2f}%, "
                     f"win_rate={result.win_rate:.1f}%, "
                     f"max_dd={result.max_drawdown_pct:.2f}%, "
                     f"sharpe={result.sharpe_ratio:.2f}")
        return result

    async def run_mtf(self, dataframes: dict[str, pd.DataFrame], symbol: str,
                      lookback: int = 200) -> BacktestResult:
        """Run backtest using multi-timeframe evaluation.

        The primary timeframe (first key in *dataframes*) drives bar-by-bar
        iteration and SL/TP checks.  At each bar the engine builds per-TF
        windows and calls ``strategy.evaluate_mtf()``.

        Higher-timeframe DataFrames are sliced so that only bars whose
        timestamp <= the current primary bar are visible (no look-ahead).

        Args:
            dataframes: ``{"H1": df_h1, "H4": df_h4, "D1": df_d1, ...}``
            symbol: Symbol name.
            lookback: Min primary bars before evaluating.
        """
        tf_names = list(dataframes.keys())
        primary_tf = tf_names[0]
        primary_df = dataframes[primary_tf]

        trades: list[BacktestTrade] = []
        balance = self._initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        in_trade = False
        trade_entry = None
        cooldown_until = 0       # bar index until which new entries are blocked
        partial_closed = False   # whether 50% was already taken at 1:1
        consecutive_losses = 0   # count of consecutive losing trades

        # Compute ATR for volatility filter
        import ta as _ta
        _atr_df = primary_df.copy()
        _atr_df["_atr"] = _ta.volatility.average_true_range(
            _atr_df["high"], _atr_df["low"], _atr_df["close"], window=14,
        )
        _atr_df["_atr_sma"] = _atr_df["_atr"].rolling(50).mean()

        # Config
        cooldown_bars = 6        # bars to wait after a loss
        atr_filter = True        # skip trades when ATR is abnormal
        breakeven_at_1r = True   # move SL to entry after 1:1 R:R
        partial_close = True     # close 50% at 1:1 R:R

        logger.info(
            f"MTF Backtesting {symbol} on {len(primary_df)} primary bars "
            f"({primary_tf}), timeframes={tf_names}, lookback={lookback}"
        )

        for i in range(lookback, len(primary_df)):
            # --- Check open trade SL/TP on the primary TF ---
            if in_trade and trade_entry:
                current_high = primary_df["high"].iloc[i]
                current_low = primary_df["low"].iloc[i]
                current_close = primary_df["close"].iloc[i]

                entry_price = trade_entry["price"]
                sl = trade_entry["sl"]
                tp = trade_entry["tp"]
                sl_distance = abs(entry_price - trade_entry["original_sl"])

                # Check if price reached 1:1 R:R (for breakeven + partial close)
                if not partial_closed and sl_distance > 0:
                    if trade_entry["side"] == "buy":
                        reached_1r = current_high >= entry_price + sl_distance
                    else:
                        reached_1r = current_low <= entry_price - sl_distance

                    if reached_1r:
                        # Move SL to breakeven
                        if breakeven_at_1r:
                            trade_entry["sl"] = entry_price
                            sl = entry_price

                        # Partial close: book 50% of the 1R profit
                        if partial_close:
                            partial_pnl = balance * self._risk_per_trade * 0.5
                            balance += partial_pnl
                            trade_entry["partial_pnl"] = partial_pnl
                            partial_closed = True

                # Trailing stop: after breakeven, trail SL behind price by 1.5x ATR
                if partial_closed and i < len(_atr_df):
                    atr_val = _atr_df["_atr"].iloc[i]
                    if not pd.isna(atr_val) and atr_val > 0:
                        trail_dist = atr_val * 1.5
                        if trade_entry["side"] == "buy":
                            trail_sl = current_close - trail_dist
                            if trail_sl > sl and trail_sl > entry_price:
                                trade_entry["sl"] = trail_sl
                                sl = trail_sl
                        else:
                            trail_sl = current_close + trail_dist
                            if trail_sl < sl and trail_sl < entry_price:
                                trade_entry["sl"] = trail_sl
                                sl = trail_sl

                hit_sl = False
                hit_tp = False

                if trade_entry["side"] == "buy":
                    hit_sl = current_low <= sl and sl > 0
                    hit_tp = current_high >= tp and tp > 0
                else:
                    hit_sl = current_high >= sl and sl > 0
                    hit_tp = current_low <= tp and tp > 0

                if hit_sl or hit_tp:
                    exit_price = sl if hit_sl else tp
                    if trade_entry["side"] == "buy":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    original_sl_dist = abs(
                        (entry_price - trade_entry["original_sl"]) / entry_price
                    )
                    # Remaining position (50% if partial was taken)
                    position_factor = 0.5 if partial_closed else 1.0
                    pnl = balance * self._risk_per_trade * position_factor * (
                        pnl_pct / original_sl_dist
                    ) if original_sl_dist > 0 else 0

                    # Add partial PnL that was already booked
                    total_pnl = pnl + trade_entry.get("partial_pnl", 0)

                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, dd)

                    reason = "SL hit" if hit_sl else "TP hit"
                    if partial_closed and hit_sl and sl == entry_price:
                        reason = "Breakeven (partial taken)"

                    trades.append(BacktestTrade(
                        entry_idx=trade_entry["idx"],
                        exit_idx=i,
                        side=trade_entry["side"],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        sl=trade_entry["original_sl"],
                        tp=tp,
                        pnl=total_pnl,
                        pnl_pct=pnl_pct * 100,
                        confidence=trade_entry["confidence"],
                        reason=reason,
                    ))

                    # Track consecutive losses + cooldown
                    if total_pnl < 0:
                        consecutive_losses += 1
                        cooldown_until = i + cooldown_bars
                    else:
                        consecutive_losses = 0

                    in_trade = False
                    trade_entry = None
                    partial_closed = False
                continue

            # --- Cooldown check ---
            if i < cooldown_until:
                continue

            # --- Consecutive loss filter: skip after 2 losses in a row ---
            if consecutive_losses >= 2:
                consecutive_losses = 0  # reset, skip this one signal
                continue

            # --- Volatility filter ---
            if atr_filter and i < len(_atr_df):
                atr_now = _atr_df["_atr"].iloc[i]
                atr_avg = _atr_df["_atr_sma"].iloc[i]
                if not pd.isna(atr_now) and not pd.isna(atr_avg) and atr_avg > 0:
                    atr_ratio = atr_now / atr_avg
                    if atr_ratio < 0.5 or atr_ratio > 2.5:
                        continue  # skip: too quiet (choppy) or too volatile (news)

            # --- Build per-TF windows (no look-ahead) ---
            current_ts = primary_df["timestamp"].iloc[i] if "timestamp" in primary_df.columns else i

            windows: dict[str, pd.DataFrame] = {}
            for tf_name, tf_df in dataframes.items():
                if tf_name == primary_tf:
                    windows[tf_name] = primary_df.iloc[:i + 1]
                else:
                    if "timestamp" in tf_df.columns:
                        mask = tf_df["timestamp"] <= current_ts
                        windows[tf_name] = tf_df.loc[mask]
                    else:
                        ratio = len(tf_df) / len(primary_df)
                        htf_idx = min(int(i * ratio) + 1, len(tf_df))
                        windows[tf_name] = tf_df.iloc[:htf_idx]

            # --- Evaluate MTF ---
            signal = await self._strategy.evaluate_mtf(symbol, windows)

            if signal.signal == Signal.HOLD or signal.confidence < self._min_confidence:
                continue

            if signal.suggested_sl == 0:
                continue

            # Enter trade
            entry_price = primary_df["close"].iloc[i]
            trade_entry = {
                "idx": i,
                "price": entry_price,
                "side": signal.signal.value,
                "sl": signal.suggested_sl,
                "tp": signal.suggested_tp,
                "original_sl": signal.suggested_sl,  # keep original for R:R calc
                "confidence": signal.confidence,
                "partial_pnl": 0,
            }
            in_trade = True
            partial_closed = False

        # --- Compute stats ---
        total_return = ((balance - self._initial_balance) / self._initial_balance) * 100
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = (len(winning) / len(trades) * 100) if trades else 0

        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        if trades:
            import numpy as np
            returns = [t.pnl_pct for t in trades]
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0
        else:
            sharpe = 0

        result = BacktestResult(
            trades=trades,
            total_return_pct=total_return,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
        )

        logger.info(f"MTF Backtest complete: {result.total_trades} trades, "
                     f"return={result.total_return_pct:.2f}%, "
                     f"win_rate={result.win_rate:.1f}%, "
                     f"max_dd={result.max_drawdown_pct:.2f}%, "
                     f"sharpe={result.sharpe_ratio:.2f}")
        return result
