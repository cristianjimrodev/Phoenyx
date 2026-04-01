"""Tests for src/reporting/report.py"""
from __future__ import annotations

from pathlib import Path

import pytest

from backtest.engine import BacktestResult, BacktestTrade
from src.reporting.report import generate_backtest_report


def _make_result(n_trades: int = 5) -> BacktestResult:
    """Create a BacktestResult with synthetic trades."""
    trades = []
    for i in range(n_trades):
        pnl = 50.0 if i % 2 == 0 else -30.0
        trades.append(BacktestTrade(
            entry_idx=i * 10,
            exit_idx=i * 10 + 5,
            side="buy" if i % 2 == 0 else "sell",
            entry_price=1.1000,
            exit_price=1.1050 if pnl > 0 else 1.0970,
            sl=1.0950,
            tp=1.1100,
            pnl=pnl,
            pnl_pct=0.45 if pnl > 0 else -0.27,
            confidence=75.0,
            reason="TP hit" if pnl > 0 else "SL hit",
        ))

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    return BacktestResult(
        trades=trades,
        total_return_pct=0.9,
        win_rate=60.0,
        total_trades=n_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        max_drawdown_pct=1.5,
        sharpe_ratio=1.2,
        avg_win=50.0,
        avg_loss=-30.0,
        profit_factor=1.67,
    )


class TestGenerateBacktestReport:
    def test_creates_html_file(self, tmp_path):
        result = _make_result()
        output = str(tmp_path / "report.html")
        path = generate_backtest_report(result, output_path=output)
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_contains_metrics(self, tmp_path):
        result = _make_result()
        output = str(tmp_path / "report.html")
        path = generate_backtest_report(result, output_path=output)
        content = Path(path).read_text(encoding="utf-8")
        assert "Total Return" in content
        assert "Win Rate" in content
        assert "Sharpe Ratio" in content

    def test_contains_trade_table(self, tmp_path):
        result = _make_result(3)
        output = str(tmp_path / "report.html")
        path = generate_backtest_report(result, output_path=output)
        content = Path(path).read_text(encoding="utf-8")
        assert "Trade Log" in content
        assert "TP hit" in content or "SL hit" in content

    def test_no_trades_still_generates(self, tmp_path):
        result = BacktestResult(
            trades=[], total_return_pct=0, win_rate=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            max_drawdown_pct=0, sharpe_ratio=0,
            avg_win=0, avg_loss=0, profit_factor=0,
        )
        output = str(tmp_path / "report.html")
        path = generate_backtest_report(result, output_path=output)
        assert Path(path).exists()

    def test_contains_embedded_charts(self, tmp_path):
        result = _make_result()
        output = str(tmp_path / "report.html")
        path = generate_backtest_report(result, output_path=output)
        content = Path(path).read_text(encoding="utf-8")
        # Charts are embedded as base64 PNGs
        assert "data:image/png;base64," in content
