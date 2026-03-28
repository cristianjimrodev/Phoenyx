"""Backtesting engine to test strategies on historical data."""

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.analysis.indicators import Signal
from src.strategy.technical import TechnicalStrategy


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

    def __init__(self, strategy: TechnicalStrategy, initial_balance: float = 10000,
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
