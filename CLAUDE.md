# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phoenyx is a Python automated trading system supporting Interactive Brokers, XTB, and Paper (simulated) brokers. It combines multiple technical analysis methods (support/resistance, chart patterns, indicators, news sentiment) with configurable weights to generate trade signals, manage risk with trailing stops, and execute trades asynchronously.

## Commands

```bash
# Run tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_indicators.py -v

# Live trading (requires broker connection)
python -m src.main

# Paper trading (set broker.name: paper in config/settings.yaml)
python -m src.main

# Backtest with simulated data (args: symbol, bars, balance, min_confidence)
# Generates HTML report in reports/
python run_backtest.py EURUSD 1000 10000 60

# Backtest with real IB historical data
python run_backtest_ib.py EURUSD H1 10000 30

# Test IB connection
python test_ib_connection.py
```

## Architecture

**Entry point**: `src/main.py` — async main loop that assembles all components and runs a 60-second evaluation cycle: fetch candles → evaluate strategy → check risk → execute orders → update trailing stops. Includes auto-reconnection logic if broker disconnects.

**Broker layer** (`src/broker/`):
- `base.py` defines the abstract `BrokerClient` interface and shared data classes (`Candle`, `Trade`, `Symbol`, `AccountInfo`, etc.)
- `ib/client.py` — Interactive Brokers implementation via `ib_insync`
- `xtb/client.py` — XTB implementation via WebSocket (`xtb/websocket.py`)
- `paper.py` — `PaperBroker` wraps a real broker for market data but simulates trade execution in a virtual account. Set `broker.name: paper` in settings.yaml.

**Strategy layer** (`src/strategy/`):
- `base.py` defines abstract `Strategy` with `evaluate()` returning `TradeSignal` (signal + confidence 0–100)
- `technical.py` — `TechnicalStrategy` computes a weighted score from analysis modules:
  - S/R (35%), Patterns (30%), Indicators (25%), News (10%)
  - Confidence is normalized by active weight sum

**Analysis modules** (`src/analysis/`):
- `support_resistance.py` — pivot-based S/R detection using `scipy.signal.argrelextrema`
- `patterns.py` — 11 chart patterns via zigzag algorithm
- `indicators.py` — RSI, MACD, MAs, Bollinger Bands, ATR via `ta` library
- `news.py` — RSS-based news sentiment analysis with keyword scoring. Fetches from Google News + configurable feeds, filters by symbol relevance, scores with bullish/bearish keyword dictionaries.

**Risk management** (`src/risk/manager.py`):
- `can_trade()` checks position count, daily drawdown, and free margin
- `compute_position_size()` uses Kelly Criterion variant: `risk_amount / sl_distance`
- `compute_trailing_sl()` — ATR-based trailing stop that only moves SL in the favorable direction, never back. Enabled via `trailing_stop: true` in settings.yaml.

**Order execution** (`src/orders/manager.py`):
- `execute_signal()` opens trades through the broker, tracks history with `OrderRecord`
- Optionally persists every order to SQLite via `TradeStore`

**Data layer** (`src/data/`):
- `feed.py` — caches candles locally, fetches missing data from broker
- `store.py` — SQLite persistence for candle data
- `trade_store.py` — SQLite persistence for trade/order history. Supports save, update, load with filters, and summary stats.

**Reporting** (`src/reporting/report.py`):
- `generate_backtest_report()` — creates an HTML report with equity curve, drawdown chart, PnL distribution, win/loss pie chart, and full trade log. Embedded base64 PNG charts via matplotlib.
- `generate_live_report()` — same for live trading data from `TradeStore`

**Backtesting** (`backtest/engine.py`):
- `BacktestEngine.run()` simulates strategy bar-by-bar, returns metrics (return %, win rate, max drawdown, Sharpe ratio). Backtest reports are auto-generated as HTML.

## Configuration

- `config/settings.yaml` — broker connection (ib/xtb/paper), risk parameters (including trailing stop), symbol list, timeframe, logging
- `config/strategies.yaml` — analysis weights, indicator parameters, pattern settings, news sentiment config
- `.env` (from `.env.example`) — broker credentials (`BROKER`, `IB_HOST`, `XTB_USER_ID`, etc.)

## Key Patterns

- **Abstract base classes** for broker and strategy allow adding new implementations without changing core logic
- **Fully async** — all broker I/O uses asyncio
- **Dataclass-heavy** — typed data classes for signals, candles, trades, account info
- **Config-driven** — all trading parameters live in YAML files, not hardcoded
- **Auto-reconnection** — main loop detects broker disconnections and retries with exponential backoff
