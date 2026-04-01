"""HTML report generator for backtest results and live trade history."""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger


def _fig_to_base64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_equity_curve(balances: list[float], labels: list[str] | None = None) -> str:
    """Generate an equity curve chart and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(balances, color="#2196F3", linewidth=1.5)
    ax.fill_between(range(len(balances)), balances, balances[0],
                    alpha=0.1, color="#2196F3")
    ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Balance")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=balances[0], color="gray", linestyle="--", alpha=0.5)
    return _fig_to_base64(fig)


def _plot_drawdown(balances: list[float]) -> str:
    """Generate a drawdown chart and return as base64 PNG."""
    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / peak * 100

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(drawdown)), drawdown, 0,
                    color="#F44336", alpha=0.4)
    ax.plot(drawdown, color="#F44336", linewidth=1)
    ax.set_title("Drawdown %", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown %")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _plot_pnl_distribution(pnls: list[float]) -> str:
    """Generate a PnL distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, width=0.8)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_title("PnL per Trade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def _plot_price_with_trades(df: pd.DataFrame, trades: list, title: str = "Price & Trades") -> str:
    """Plot candlestick chart with entry/exit markers for each trade.

    Args:
        df: OHLCV DataFrame (index=datetime or integer).
        trades: List of BacktestTrade objects with entry_idx, exit_idx, side, pnl, etc.
    """
    fig, ax = plt.subplots(figsize=(18, 7))

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(closes)

    # Draw candlesticks
    # Thin candles for large datasets, thicker for smaller
    body_width = max(0.3, min(0.8, 400 / n))
    wick_width = max(0.1, body_width * 0.15)

    for i in range(n):
        color = "#26A69A" if closes[i] >= opens[i] else "#EF5350"  # green / red
        # Wick (high-low line)
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=wick_width, solid_capstyle="round")
        # Body (open-close rectangle)
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        if body_height < (highs[i] - lows[i]) * 0.01:
            body_height = (highs[i] - lows[i]) * 0.01  # min visible body
        ax.bar(i, body_height, bottom=body_bottom, width=body_width,
               color=color, edgecolor=color, linewidth=0)

    # Trade markers
    for t in trades:
        entry_idx = t.entry_idx
        exit_idx = t.exit_idx

        # Clamp to valid range
        if entry_idx >= n or exit_idx >= n:
            continue

        is_buy = t.side == "buy"
        is_win = t.pnl > 0

        # Entry marker
        entry_color = "#2196F3" if is_buy else "#FF5722"
        entry_marker = "^" if is_buy else "v"
        ax.scatter(entry_idx, t.entry_price, color=entry_color,
                   marker=entry_marker, s=80, zorder=5, edgecolors="white", linewidths=0.5)

        # Exit marker
        exit_color = "#4CAF50" if is_win else "#F44336"
        exit_marker = "D"  # diamond
        ax.scatter(exit_idx, t.exit_price, color=exit_color,
                   marker=exit_marker, s=50, zorder=5, edgecolors="white", linewidths=0.5)

        # Connect entry to exit with a line
        line_color = "#4CAF50" if is_win else "#F44336"
        ax.plot([entry_idx, exit_idx], [t.entry_price, t.exit_price],
                color=line_color, linewidth=1.0, alpha=0.5, linestyle="--")

        # SL/TP levels (thin horizontal lines)
        if t.sl > 0:
            ax.plot([entry_idx, exit_idx], [t.sl, t.sl],
                    color="#F44336", linewidth=0.5, alpha=0.3)
        if t.tp > 0:
            ax.plot([entry_idx, exit_idx], [t.tp, t.tp],
                    color="#4CAF50", linewidth=0.5, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2196F3",
               markersize=10, label="BUY entry"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#FF5722",
               markersize=10, label="SELL entry"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="Win exit"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#F44336",
               markersize=8, label="Loss exit"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.set_xlabel("Bar index")
    ax.grid(True, alpha=0.2)

    # Add date labels on x-axis if we have datetime index
    if hasattr(df.index, 'strftime'):
        n_ticks = 8
        step = max(1, len(df) // n_ticks)
        tick_positions = list(range(0, len(df), step))
        tick_labels = [df.index[i].strftime("%Y-%m-%d") for i in tick_positions if i < len(df)]
        ax.set_xticks(tick_positions[:len(tick_labels)])
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_price_with_trades_zoomed(df: pd.DataFrame, trades: list,
                                    zoom_last_n: int = 500) -> str:
    """Zoomed-in view of the last N bars with trade markers."""
    if len(df) <= zoom_last_n:
        return ""

    # Reindex trades to zoomed window
    offset = len(df) - zoom_last_n
    zoomed_df = df.iloc[-zoom_last_n:].copy()
    zoomed_trades = []
    for t in trades:
        if t.entry_idx >= offset:
            # Create a copy-like object with adjusted indices
            class _ZoomedTrade:
                pass
            zt = _ZoomedTrade()
            zt.entry_idx = t.entry_idx - offset
            zt.exit_idx = t.exit_idx - offset
            zt.entry_price = t.entry_price
            zt.exit_price = t.exit_price
            zt.side = t.side
            zt.pnl = t.pnl
            zt.sl = t.sl
            zt.tp = t.tp
            zoomed_trades.append(zt)

    if not zoomed_trades:
        return ""

    return _plot_price_with_trades(
        zoomed_df, zoomed_trades,
        title=f"Price & Trades (last {zoom_last_n} bars)"
    )


def _plot_win_loss_pie(winning: int, losing: int) -> str:
    """Generate a win/loss pie chart."""
    fig, ax = plt.subplots(figsize=(4, 4))
    if winning + losing == 0:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=14)
    else:
        sizes = [winning, losing]
        labels = [f"Wins ({winning})", f"Losses ({losing})"]
        colors = ["#4CAF50", "#F44336"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 11})
    ax.set_title("Win/Loss Ratio", fontsize=14, fontweight="bold")
    return _fig_to_base64(fig)


def generate_backtest_report(result, output_path: str = "reports/backtest_report.html",
                             price_df: pd.DataFrame | None = None) -> str:
    """Generate an HTML report from a BacktestResult.

    Args:
        result: A BacktestResult dataclass instance.
        output_path: Where to save the HTML file.
        price_df: Optional primary-timeframe OHLCV DataFrame for candlestick chart.

    Returns:
        The absolute path of the generated report.
    """
    # Build equity curve from trades
    initial_balance = 10000  # inferred from result
    if result.trades:
        # Reconstruct balance progression
        total_pnl = sum(t.pnl for t in result.trades)
        if result.total_return_pct != 0:
            initial_balance = total_pnl / (result.total_return_pct / 100)

    balances = [initial_balance]
    pnls = []
    for trade in result.trades:
        balances.append(balances[-1] + trade.pnl)
        pnls.append(trade.pnl)

    # Generate charts
    equity_img = _plot_equity_curve(balances) if balances else ""
    drawdown_img = _plot_drawdown(balances) if len(balances) > 1 else ""
    pnl_img = _plot_pnl_distribution(pnls) if pnls else ""
    pie_img = _plot_win_loss_pie(result.winning_trades, result.losing_trades)

    # Candlestick chart with trade markers
    price_chart_img = ""
    price_chart_zoom_img = ""
    if price_df is not None and not price_df.empty and result.trades:
        price_chart_img = _plot_price_with_trades(
            price_df, result.trades, title="Candlestick Chart - All Trades"
        )
        price_chart_zoom_img = _plot_price_with_trades_zoomed(
            price_df, result.trades, zoom_last_n=400
        )

    # Build trades table
    trades_html = ""
    for i, t in enumerate(result.trades, 1):
        pnl_class = "win" if t.pnl > 0 else "loss"
        trades_html += f"""
        <tr class="{pnl_class}">
            <td>{i}</td>
            <td>{t.side.upper()}</td>
            <td>{t.entry_price:.5f}</td>
            <td>{t.exit_price:.5f}</td>
            <td>{t.sl:.5f}</td>
            <td>{t.tp:.5f}</td>
            <td class="pnl">{t.pnl:.2f}</td>
            <td>{t.pnl_pct:.2f}%</td>
            <td>{t.confidence:.0f}%</td>
            <td>{t.reason}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phoenyx Backtest Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px; margin: 20px 0; }}
    .metric-card {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
    .metric-card .value {{ font-size: 28px; font-weight: bold; margin: 5px 0; }}
    .metric-card .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
    .positive {{ color: #4CAF50; }}
    .negative {{ color: #F44336; }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin: 20px 0; }}
    .chart-row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
    .chart {{ background: white; padding: 15px; border-radius: 8px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .chart img {{ width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    th {{ background: #1a237e; color: white; padding: 12px 8px; font-size: 13px; }}
    td {{ padding: 10px 8px; border-bottom: 1px solid #eee; font-size: 13px;
          text-align: center; }}
    tr.win .pnl {{ color: #4CAF50; font-weight: bold; }}
    tr.loss .pnl {{ color: #F44336; font-weight: bold; }}
    .timestamp {{ color: #999; font-size: 12px; margin-top: 20px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <h1>Phoenyx Backtest Report</h1>

    <div class="metrics">
        <div class="metric-card">
            <div class="label">Total Return</div>
            <div class="value {'positive' if result.total_return_pct >= 0 else 'negative'}">
                {result.total_return_pct:+.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Trades</div>
            <div class="value">{result.total_trades}</div>
        </div>
        <div class="metric-card">
            <div class="label">Win Rate</div>
            <div class="value">{result.win_rate:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Max Drawdown</div>
            <div class="value negative">{result.max_drawdown_pct:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Sharpe Ratio</div>
            <div class="value">{result.sharpe_ratio:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Profit Factor</div>
            <div class="value">{"∞" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"}</div>
        </div>
        <div class="metric-card">
            <div class="label">Avg Win</div>
            <div class="value positive">{result.avg_win:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Avg Loss</div>
            <div class="value negative">{result.avg_loss:.2f}</div>
        </div>
    </div>

    <div class="charts">
        {"<div class='chart'><img src='data:image/png;base64," + price_chart_img + "' alt='Price Chart'></div>" if price_chart_img else ""}
        {"<div class='chart'><img src='data:image/png;base64," + price_chart_zoom_img + "' alt='Price Chart Zoom'></div>" if price_chart_zoom_img else ""}
        <div class="chart"><img src="data:image/png;base64,{equity_img}" alt="Equity Curve"></div>
        <div class="chart-row">
            <div class="chart"><img src="data:image/png;base64,{drawdown_img}" alt="Drawdown"></div>
            <div class="chart"><img src="data:image/png;base64,{pie_img}" alt="Win/Loss"></div>
        </div>
        <div class="chart"><img src="data:image/png;base64,{pnl_img}" alt="PnL Distribution"></div>
    </div>

    <h2>Trade Log</h2>
    <table>
        <thead>
            <tr>
                <th>#</th><th>Side</th><th>Entry</th><th>Exit</th>
                <th>SL</th><th>TP</th><th>PnL</th><th>PnL %</th>
                <th>Confidence</th><th>Reason</th>
            </tr>
        </thead>
        <tbody>
            {trades_html}
        </tbody>
    </table>

    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    logger.info(f"Report saved to {path.absolute()}")
    return str(path.absolute())


def generate_live_report(trade_df: pd.DataFrame,
                         output_path: str = "reports/live_report.html",
                         initial_balance: float = 1000.0) -> str:
    """Generate an HTML report from a live trading DataFrame (from TradeStore).

    Args:
        trade_df: DataFrame from TradeStore.load_trades().
        output_path: Where to save the HTML file.
        initial_balance: Account balance for margin % calculation.

    Returns:
        The absolute path of the generated report.
    """
    closed = trade_df[trade_df["status"] == "closed"]

    if closed.empty:
        total_return = 0
        win_rate = 0
        total_pnl = 0
        winning = 0
        losing = 0
    else:
        total_pnl = closed["pnl"].sum()
        winning = len(closed[closed["pnl"] > 0])
        losing = len(closed[closed["pnl"] <= 0])
        win_rate = winning / len(closed) * 100
        total_return = total_pnl

    # Build equity from closed trades
    balances = [0.0]
    pnls = []
    for _, row in closed.iterrows():
        balances.append(balances[-1] + row["pnl"])
        pnls.append(row["pnl"])

    equity_img = _plot_equity_curve(balances) if len(balances) > 1 else ""
    drawdown_img = _plot_drawdown(balances) if len(balances) > 2 else ""
    pnl_img = _plot_pnl_distribution(pnls) if pnls else ""
    pie_img = _plot_win_loss_pie(winning, losing)

    # Build table from all trades
    rows_html = ""
    for _, row in trade_df.iterrows():
        pnl_class = "win" if row.get("pnl", 0) > 0 else "loss"
        cs = row.get("contract_size", 100000)
        margin_eur = row.get("volume", 0) * row.get("entry_price", 0) * cs / 100
        margin_pct = (margin_eur / initial_balance * 100) if initial_balance > 0 else 0
        rows_html += f"""
        <tr class="{pnl_class}">
            <td>{row.get('timestamp', '')}</td>
            <td>{row.get('symbol', '')}</td>
            <td>{row.get('side', '').upper()}</td>
            <td>{row.get('volume', 0):.2f}</td>
            <td>{row.get('entry_price', 0):.5f}</td>
            <td>{row.get('exit_price', 0):.5f}</td>
            <td>{margin_eur:,.2f}</td>
            <td>{margin_pct:.1f}%</td>
            <td class="pnl">{row.get('pnl', 0):.2f}</td>
            <td>{row.get('status', '')}</td>
            <td>{row.get('confidence', 0):.0f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phoenyx Live Trading Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px; margin: 20px 0; }}
    .metric-card {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
    .metric-card .value {{ font-size: 28px; font-weight: bold; margin: 5px 0; }}
    .metric-card .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
    .positive {{ color: #4CAF50; }}
    .negative {{ color: #F44336; }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin: 20px 0; }}
    .chart-row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
    .chart {{ background: white; padding: 15px; border-radius: 8px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .chart img {{ width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    th {{ background: #1a237e; color: white; padding: 12px 8px; font-size: 13px; }}
    td {{ padding: 10px 8px; border-bottom: 1px solid #eee; font-size: 13px;
          text-align: center; }}
    tr.win .pnl {{ color: #4CAF50; font-weight: bold; }}
    tr.loss .pnl {{ color: #F44336; font-weight: bold; }}
    .timestamp {{ color: #999; font-size: 12px; margin-top: 20px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <h1>Phoenyx Live Trading Report</h1>

    <div class="metrics">
        <div class="metric-card">
            <div class="label">Total PnL</div>
            <div class="value {'positive' if total_pnl >= 0 else 'negative'}">{total_pnl:+.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Trades</div>
            <div class="value">{len(trade_df)}</div>
        </div>
        <div class="metric-card">
            <div class="label">Win Rate</div>
            <div class="value">{win_rate:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Wins / Losses</div>
            <div class="value">{winning} / {losing}</div>
        </div>
    </div>

    <div class="charts">
        {"<div class='chart'><img src='data:image/png;base64," + equity_img + "' alt='Equity'></div>" if equity_img else ""}
        <div class="chart-row">
            {"<div class='chart'><img src='data:image/png;base64," + drawdown_img + "' alt='Drawdown'></div>" if drawdown_img else "<div></div>"}
            <div class="chart"><img src="data:image/png;base64,{pie_img}" alt="Win/Loss"></div>
        </div>
        {"<div class='chart'><img src='data:image/png;base64," + pnl_img + "' alt='PnL'></div>" if pnl_img else ""}
    </div>

    <h2>Trade History</h2>
    <table>
        <thead>
            <tr>
                <th>Time</th><th>Symbol</th><th>Side</th><th>Volume</th>
                <th>Entry</th><th>Exit</th><th>Margin (EUR)</th><th>Margin %</th>
                <th>PnL</th><th>Status</th><th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    logger.info(f"Live report saved to {path.absolute()}")
    return str(path.absolute())


def _plot_param_sensitivity(param_name: str, values: list[float],
                            scores: list[float]) -> str:
    """Generate a bar chart showing how a parameter affects the objective score."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(values)), scores, color="#2196F3", width=0.6)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values], fontsize=9)
    ax.set_title(f"Sensitivity: {param_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Avg Score")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def generate_optimization_report(result, output_path: str = "reports/optimization_report.html") -> str:
    """Generate an HTML report from an OptimizationResult.

    Args:
        result: An OptimizationResult dataclass instance.
        output_path: Where to save the HTML file.

    Returns:
        The absolute path of the generated report.
    """
    # Build top-10 results table
    top_n = min(10, len(result.all_runs))
    top_runs_html = ""
    for i, run in enumerate(result.all_runs[:top_n], 1):
        r = run.result
        params_str = ", ".join(f"{k}={v}" for k, v in run.params.items())
        row_class = "best" if i == 1 else ""
        top_runs_html += f"""
        <tr class="{row_class}">
            <td>{i}</td>
            <td>{run.score:.4f}</td>
            <td>{r.total_return_pct:+.2f}%</td>
            <td>{r.win_rate:.1f}%</td>
            <td>{r.total_trades}</td>
            <td>{r.sharpe_ratio:.2f}</td>
            <td>{r.max_drawdown_pct:.2f}%</td>
            <td>{r.profit_factor:.2f}</td>
            <td class="params">{params_str}</td>
        </tr>"""

    # Best params table
    best_params_html = ""
    for param, value in result.best_params.items():
        best_params_html += f"""
        <tr>
            <td class="param-name">{param}</td>
            <td class="param-value">{value}</td>
        </tr>"""

    # Parameter sensitivity analysis
    # For each parameter, compute the average score for each of its values
    sensitivity_charts = []
    if result.all_runs:
        # Collect all unique param names
        all_param_names = list(result.all_runs[0].params.keys()) if result.all_runs else []
        for param_name in all_param_names:
            # Group scores by this param's value
            value_scores: dict[float, list[float]] = {}
            for run in result.all_runs:
                val = run.params.get(param_name)
                if val is not None:
                    value_scores.setdefault(val, []).append(run.score)

            # Sort by value and compute averages
            sorted_vals = sorted(value_scores.keys())
            avg_scores = [np.mean(value_scores[v]) for v in sorted_vals]

            chart_img = _plot_param_sensitivity(param_name, sorted_vals, avg_scores)
            sensitivity_charts.append(chart_img)

    sensitivity_html = ""
    for img in sensitivity_charts:
        sensitivity_html += f'<div class="chart"><img src="data:image/png;base64,{img}" alt="Sensitivity"></div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phoenyx Optimization Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #1a237e; margin-top: 30px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px; margin: 20px 0; }}
    .metric-card {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
    .metric-card .value {{ font-size: 28px; font-weight: bold; margin: 5px 0; }}
    .metric-card .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
    .positive {{ color: #4CAF50; }}
    .negative {{ color: #F44336; }}
    .best-params {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
    .best-params table {{ width: auto; border-collapse: collapse; }}
    .best-params td {{ padding: 8px 20px; border-bottom: 1px solid #eee; }}
    .best-params .param-name {{ font-weight: bold; color: #1a237e; }}
    .best-params .param-value {{ font-family: monospace; font-size: 15px; }}
    table.results {{ width: 100%; border-collapse: collapse; background: white;
                     border-radius: 8px; overflow: hidden;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    table.results th {{ background: #1a237e; color: white; padding: 12px 8px;
                        font-size: 13px; white-space: nowrap; }}
    table.results td {{ padding: 10px 8px; border-bottom: 1px solid #eee;
                        font-size: 13px; text-align: center; }}
    table.results td.params {{ text-align: left; font-family: monospace; font-size: 11px;
                               max-width: 400px; word-wrap: break-word; }}
    tr.best {{ background: #E8F5E9; font-weight: bold; }}
    .sensitivity {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 15px; margin: 20px 0; }}
    .chart {{ background: white; padding: 15px; border-radius: 8px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .chart img {{ width: 100%; height: auto; }}
    .timestamp {{ color: #999; font-size: 12px; margin-top: 20px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <h1>Phoenyx Optimization Report</h1>

    <div class="summary">
        <div class="metric-card">
            <div class="label">Objective</div>
            <div class="value" style="font-size:20px;">{result.objective}</div>
        </div>
        <div class="metric-card">
            <div class="label">Best Score</div>
            <div class="value positive">{result.best_score:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Combinations</div>
            <div class="value">{result.total_combinations}</div>
        </div>
        <div class="metric-card">
            <div class="label">Successful Runs</div>
            <div class="value">{len(result.all_runs)}</div>
        </div>
        <div class="metric-card">
            <div class="label">Runtime</div>
            <div class="value">{result.runtime_seconds:.1f}s</div>
        </div>
    </div>

    <h2>Best Parameters</h2>
    <div class="best-params">
        <table>
            {best_params_html}
        </table>
    </div>

    <h2>Top {top_n} Results</h2>
    <table class="results">
        <thead>
            <tr>
                <th>Rank</th><th>Score</th><th>Return</th><th>Win Rate</th>
                <th>Trades</th><th>Sharpe</th><th>Max DD</th><th>PF</th>
                <th>Parameters</th>
            </tr>
        </thead>
        <tbody>
            {top_runs_html}
        </tbody>
    </table>

    <h2>Parameter Sensitivity</h2>
    <div class="sensitivity">
        {sensitivity_html}
    </div>

    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    logger.info(f"Optimization report saved to {path.absolute()}")
    return str(path.absolute())
