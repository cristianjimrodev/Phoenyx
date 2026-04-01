"""Main entry point for the automated trading system."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.broker.base import BrokerClient
from src.data.feed import DataFeed
from src.data.store import DataStore
from src.data.trade_store import TradeStore
from src.strategy.technical import TechnicalStrategy
from src.risk.manager import RiskManager
from src.risk.correlation import CorrelationMatrix
from src.risk.portfolio import PortfolioRiskManager
from src.orders.manager import OrderManager
from src.notifications.manager import NotificationManager
from src.notifications.base import NotificationEvent
from src.dashboard.state import DashboardState
from src.dashboard.app import create_app as create_dashboard_app
from src.utils.logging import setup_logging


def create_broker(settings: dict) -> BrokerClient:
    """Create the appropriate broker client based on config."""
    broker_cfg = settings.get("broker", {})
    name = os.getenv("BROKER", broker_cfg.get("name", "ib"))

    if name == "ib":
        from src.broker.ib.client import IBClient
        ib_cfg = broker_cfg.get("ib", {})
        return IBClient(
            host=os.getenv("IB_HOST", ib_cfg.get("host", "127.0.0.1")),
            port=int(os.getenv("IB_PORT", ib_cfg.get("port", 7497))),
            client_id=int(os.getenv("IB_CLIENT_ID", ib_cfg.get("client_id", 1))),
        )

    elif name == "xtb":
        from src.broker.xtb.client import XTBClient
        xtb_cfg = broker_cfg.get("xtb", {})
        user_id = os.getenv("XTB_USER_ID")
        password = os.getenv("XTB_PASSWORD")
        if not user_id or not password:
            logger.error("Set XTB_USER_ID and XTB_PASSWORD in .env file")
            sys.exit(1)
        return XTBClient(
            user_id=user_id,
            password=password,
            mode=os.getenv("XTB_MODE", xtb_cfg.get("mode", "demo")),
            urls=xtb_cfg.get("urls"),
        )

    elif name == "paper":
        from src.broker.paper import PaperBroker
        paper_cfg = broker_cfg.get("paper", {})
        data_broker_name = paper_cfg.get("data_broker", "ib")
        # Recursively create the data broker
        data_settings = {**settings, "broker": {**broker_cfg, "name": data_broker_name}}
        data_broker = create_broker(data_settings)
        return PaperBroker(
            data_broker=data_broker,
            initial_balance=paper_cfg.get("initial_balance", 10000),
            currency=paper_cfg.get("currency", "USD"),
        )

    else:
        logger.error(f"Unknown broker: {name}. Use 'ib', 'xtb', or 'paper'.")
        sys.exit(1)


async def main():
    load_dotenv()

    # Load config
    config_path = Path("config/settings.yaml")
    with open(config_path) as f:
        settings = yaml.safe_load(f)

    strat_path = Path("config/strategies.yaml")
    with open(strat_path) as f:
        strat_config = yaml.safe_load(f)

    # Setup logging
    log_cfg = settings.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/trading.log"),
        rotation=log_cfg.get("rotation", "10 MB"),
        retention=log_cfg.get("retention", "30 days"),
    )

    # Create broker
    broker = create_broker(settings)
    broker_name = os.getenv("BROKER", settings.get("broker", {}).get("name", "ib"))

    logger.info("=" * 60)
    logger.info("AUTOMATED TRADING SYSTEM STARTING")
    logger.info(f"Broker: {broker_name.upper()}")
    logger.info("=" * 60)

    # Initialize components
    store = DataStore()
    trade_store = TradeStore()
    feed = DataFeed(broker, store)
    strategy = TechnicalStrategy(strat_config.get("analysis", {}))
    risk_mgr = RiskManager(settings.get("risk", {}))
    order_mgr = OrderManager(broker, trade_store=trade_store)
    notification_mgr = NotificationManager.from_config(settings)

    # Portfolio-level correlation risk
    portfolio_cfg = settings.get("risk", {}).get("portfolio", {})
    correlation_matrix: CorrelationMatrix | None = None
    if portfolio_cfg.get("enabled", False):
        correlation_matrix = CorrelationMatrix(
            lookback=portfolio_cfg.get("correlation_lookback", 100),
            update_interval=portfolio_cfg.get("update_interval", 3600),
        )
        portfolio_risk = PortfolioRiskManager(portfolio_cfg, correlation_matrix)
        risk_mgr.set_portfolio_risk(portfolio_risk)
        logger.info("Portfolio correlation risk management enabled")

    # Dashboard
    dash_cfg = settings.get("dashboard", {})
    dash_state: DashboardState | None = None
    if dash_cfg.get("enabled", False):
        import uvicorn

        dash_state = DashboardState()
        dash_app = create_dashboard_app(dash_state)
        dash_host = dash_cfg.get("host", "0.0.0.0")
        dash_port = int(dash_cfg.get("port", 8080))
        uvi_config = uvicorn.Config(dash_app, host=dash_host, port=dash_port, log_level="warning")
        uvi_server = uvicorn.Server(uvi_config)
        asyncio.create_task(uvi_server.serve())
        logger.info(f"Dashboard started at http://{dash_host}:{dash_port}")

    trading_cfg = settings.get("trading", {})
    symbols = trading_cfg.get("symbols", ["EURUSD"])
    timeframe = trading_cfg.get("timeframe", "H1")
    min_confidence = trading_cfg.get("min_confidence", 30)
    candles_count = trading_cfg.get("candles_history", 500)
    confirmation_timeframes = trading_cfg.get("confirmation_timeframes", [])

    try:
        connected = await broker.connect()
        if not connected:
            logger.error(f"Failed to connect to {broker_name}. Exiting.")
            return

        account = await broker.get_account_info()
        risk_mgr.set_daily_start_balance(account.balance)
        logger.info(f"Account: balance={account.balance:.2f} {account.currency}, "
                     f"equity={account.equity:.2f}, free_margin={account.free_margin:.2f}")

        for symbol in symbols:
            await feed.subscribe(symbol)

        # Compute initial correlation matrix from historical data
        if correlation_matrix is not None:
            candle_data = {}
            for symbol in symbols:
                df = await feed.get_candles(symbol, timeframe, candles_count)
                if not df.empty:
                    candle_data[symbol] = df
            correlation_matrix.update(candle_data)

        logger.info(f"Monitoring symbols: {symbols}")
        if confirmation_timeframes:
            logger.info(f"Timeframe: {timeframe} | Confirmation: {confirmation_timeframes} | Min confidence: {min_confidence}")
        else:
            logger.info(f"Timeframe: {timeframe} | Min confidence: {min_confidence}")
        logger.info("Trading loop started. Press Ctrl+C to stop.")

        while True:
            # Check connection health and reconnect if needed
            if not await _ensure_connected(broker):
                logger.warning("Broker disconnected, waiting 30s before retry...")
                await asyncio.sleep(30)
                continue

            # Update dashboard state
            if dash_state is not None:
                try:
                    account = await broker.get_account_info()
                    dash_state.update_account({
                        "balance": account.balance,
                        "equity": account.equity,
                        "margin": account.margin,
                        "free_margin": account.free_margin,
                        "currency": account.currency,
                    })
                    open_trades = await broker.get_open_trades()
                    dash_state.update_positions([
                        {
                            "symbol": t.symbol,
                            "side": t.side.value,
                            "volume": t.volume,
                            "open_price": t.open_price,
                            "sl": t.sl,
                            "tp": t.tp,
                            "profit": t.profit,
                        }
                        for t in open_trades
                    ])
                except Exception as e:
                    logger.debug(f"Dashboard state update error: {e}")

            for symbol in symbols:
                try:
                    await _evaluate_and_trade(
                        symbol, timeframe, candles_count, min_confidence,
                        feed, strategy, risk_mgr, order_mgr, broker,
                        confirmation_timeframes=confirmation_timeframes,
                        notification_mgr=notification_mgr,
                        dash_state=dash_state,
                    )
                except Exception as e:
                    logger.error(f"Error evaluating {symbol}: {e}")

            # Trailing stop updates
            if risk_mgr.trailing_stop_enabled:
                try:
                    await _update_trailing_stops(
                        symbols, timeframe, candles_count,
                        feed, risk_mgr, broker,
                    )
                except Exception as e:
                    logger.error(f"Error updating trailing stops: {e}")

            # Periodically refresh correlation matrix
            if correlation_matrix is not None and correlation_matrix.needs_update:
                try:
                    candle_data = {}
                    for symbol in symbols:
                        df = await feed.get_candles(symbol, timeframe, candles_count)
                        if not df.empty:
                            candle_data[symbol] = df
                    correlation_matrix.update(candle_data)
                except Exception as e:
                    logger.error(f"Error updating correlation matrix: {e}")

            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await broker.disconnect()
        store.close()
        trade_store.close()
        logger.info("Trading system stopped")


async def _evaluate_and_trade(
    symbol: str, timeframe: str, candles_count: int, min_confidence: float,
    feed: DataFeed, strategy: TechnicalStrategy, risk_mgr: RiskManager,
    order_mgr: OrderManager, broker: BrokerClient,
    confirmation_timeframes: list[str] | None = None,
    notification_mgr: NotificationManager | None = None,
    dash_state: DashboardState | None = None,
):
    """Evaluate a single symbol and execute if signal is strong enough."""
    if confirmation_timeframes:
        all_timeframes = [timeframe] + [tf for tf in confirmation_timeframes if tf != timeframe]
        dataframes = await feed.get_multi_timeframe_candles(symbol, all_timeframes, candles_count)
        if not dataframes or all(df.empty for df in dataframes.values()):
            return
        signal = await strategy.evaluate_mtf(symbol, dataframes)
    else:
        df = await feed.get_candles(symbol, timeframe, candles_count)
        if df.empty:
            return
        signal = await strategy.evaluate(symbol, df)

    # Push signal to dashboard
    if dash_state is not None:
        dash_state.update_signal(symbol, {
            "signal": signal.signal.value,
            "confidence": signal.confidence,
            "suggested_sl": signal.suggested_sl,
            "suggested_tp": signal.suggested_tp,
            "details": signal.details,
        })

    if signal.signal.value == "hold":
        return

    if signal.confidence < min_confidence:
        logger.debug(f"[{symbol}] Signal {signal.signal.value} rejected: "
                     f"confidence {signal.confidence:.0f} < {min_confidence}")
        return

    account = await broker.get_account_info()
    open_trades = await broker.get_open_trades()

    can_trade, reason = risk_mgr.can_trade(account, open_trades)
    if not can_trade:
        logger.warning(f"[{symbol}] Trade blocked by risk manager: {reason}")
        if notification_mgr:
            await notification_mgr.notify(NotificationEvent(
                event_type="risk_blocked",
                symbol=symbol,
                message=f"Trade blocked by risk manager: {reason}",
                details={"reason": reason, "side": signal.signal.value},
            ))
        return

    existing = [t for t in open_trades if t.symbol == symbol]
    if existing:
        logger.debug(f"[{symbol}] Already have {len(existing)} open position(s), skipping")
        return

    sym_info = await broker.get_symbol(symbol)
    sl, tp = risk_mgr.adjust_sl_tp(signal)
    entry_price = df["close"].iloc[-1] if not df.empty else 0.0
    volume = risk_mgr.compute_position_size(
        account, signal,
        pip_value=sym_info.pip_size,
        contract_size=sym_info.contract_size,
        entry_price=entry_price,
    )

    if volume <= 0:
        logger.warning(f"[{symbol}] Computed volume is 0, skipping")
        return

    # Portfolio-level correlation risk check
    from src.broker.base import Side as _Side
    side = _Side.BUY if signal.signal.value == "buy" else _Side.SELL
    allowed, pr_reason, volume = risk_mgr.check_portfolio_risk(
        symbol, side, volume, open_trades, account,
    )
    if not allowed:
        logger.warning(f"[{symbol}] Trade blocked by portfolio risk: {pr_reason}")
        if notification_mgr:
            await notification_mgr.notify(NotificationEvent(
                event_type="risk_blocked",
                symbol=symbol,
                message=f"Trade blocked by portfolio risk: {pr_reason}",
                details={"reason": pr_reason, "side": signal.signal.value},
            ))
        return
    if volume <= 0:
        logger.warning(f"[{symbol}] Portfolio risk reduced volume to 0, skipping")
        return

    logger.info(f"[{symbol}] EXECUTING: {signal.signal.value} {volume} lots "
                f"(confidence={signal.confidence:.0f}%)")
    for detail in signal.details:
        logger.info(f"  {detail}")

    trade = await order_mgr.execute_signal(signal, volume, sl, tp)

    if trade and notification_mgr:
        await notification_mgr.notify(NotificationEvent(
            event_type="trade_opened",
            symbol=symbol,
            message=f"{signal.signal.value.upper()} {volume} lots @ {trade.open_price}",
            details={
                "side": signal.signal.value,
                "volume": volume,
                "entry_price": trade.open_price,
                "sl": sl,
                "tp": tp,
                "confidence": signal.confidence,
            },
        ))


async def _ensure_connected(broker: BrokerClient, max_retries: int = 3) -> bool:
    """Check if broker is connected; attempt reconnection if not.

    Returns True if connected, False if all retries failed.
    """
    try:
        # A lightweight call to test the connection
        await broker.get_account_info()
        return True
    except Exception:
        pass

    logger.warning("Broker connection lost. Attempting to reconnect...")
    for attempt in range(1, max_retries + 1):
        try:
            await broker.disconnect()
        except Exception:
            pass

        try:
            connected = await broker.connect()
            if connected:
                logger.info(f"Reconnected to broker (attempt {attempt}/{max_retries})")
                return True
        except Exception as e:
            logger.error(f"Reconnection attempt {attempt}/{max_retries} failed: {e}")

        wait = min(attempt * 10, 60)
        logger.info(f"Waiting {wait}s before next reconnection attempt...")
        await asyncio.sleep(wait)

    logger.error(f"Failed to reconnect after {max_retries} attempts")
    return False


async def _update_trailing_stops(
    symbols: list[str], timeframe: str, candles_count: int,
    feed: DataFeed, risk_mgr: RiskManager, broker: BrokerClient,
):
    """Check open trades and update SL via trailing stop logic."""
    open_trades = await broker.get_open_trades()
    if not open_trades:
        return

    for trade in open_trades:
        if trade.symbol not in symbols:
            continue

        df = await feed.get_candles(trade.symbol, timeframe, candles_count)
        if df.empty or "atr" not in df.columns:
            # Compute ATR on the fly if not present
            import ta
            if not df.empty and len(df) >= 14:
                df["atr"] = ta.volatility.average_true_range(
                    df["high"], df["low"], df["close"], window=14,
                )
            else:
                continue

        atr = df["atr"].iloc[-1]
        current_price = df["close"].iloc[-1]

        new_sl = risk_mgr.compute_trailing_sl(trade, current_price, atr)
        if new_sl is not None:
            success = await broker.modify_trade(trade.trade_id, sl=new_sl, tp=trade.tp)
            if success:
                logger.info(
                    f"[Trailing SL] {trade.symbol}: SL moved "
                    f"{trade.sl:.5f} → {new_sl:.5f}"
                )


if __name__ == "__main__":
    asyncio.run(main())
