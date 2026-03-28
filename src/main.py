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
from src.strategy.technical import TechnicalStrategy
from src.risk.manager import RiskManager
from src.orders.manager import OrderManager
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

    else:
        logger.error(f"Unknown broker: {name}. Use 'ib' or 'xtb'.")
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
    feed = DataFeed(broker, store)
    strategy = TechnicalStrategy(strat_config.get("analysis", {}))
    risk_mgr = RiskManager(settings.get("risk", {}))
    order_mgr = OrderManager(broker)

    trading_cfg = settings.get("trading", {})
    symbols = trading_cfg.get("symbols", ["EURUSD"])
    timeframe = trading_cfg.get("timeframe", "H1")
    min_confidence = trading_cfg.get("min_confidence", 30)
    candles_count = trading_cfg.get("candles_history", 500)

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

        logger.info(f"Monitoring symbols: {symbols}")
        logger.info(f"Timeframe: {timeframe} | Min confidence: {min_confidence}")
        logger.info("Trading loop started. Press Ctrl+C to stop.")

        while True:
            for symbol in symbols:
                try:
                    await _evaluate_and_trade(
                        symbol, timeframe, candles_count, min_confidence,
                        feed, strategy, risk_mgr, order_mgr, broker,
                    )
                except Exception as e:
                    logger.error(f"Error evaluating {symbol}: {e}")

            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await broker.disconnect()
        store.close()
        logger.info("Trading system stopped")


async def _evaluate_and_trade(
    symbol: str, timeframe: str, candles_count: int, min_confidence: float,
    feed: DataFeed, strategy: TechnicalStrategy, risk_mgr: RiskManager,
    order_mgr: OrderManager, broker: BrokerClient,
):
    """Evaluate a single symbol and execute if signal is strong enough."""
    df = await feed.get_candles(symbol, timeframe, candles_count)
    if df.empty:
        return

    signal = await strategy.evaluate(symbol, df)

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
        return

    existing = [t for t in open_trades if t.symbol == symbol]
    if existing:
        logger.debug(f"[{symbol}] Already have {len(existing)} open position(s), skipping")
        return

    sym_info = await broker.get_symbol(symbol)
    sl, tp = risk_mgr.adjust_sl_tp(signal)
    volume = risk_mgr.compute_position_size(
        account, signal,
        pip_value=sym_info.pip_size,
        contract_size=sym_info.contract_size,
    )

    if volume <= 0:
        logger.warning(f"[{symbol}] Computed volume is 0, skipping")
        return

    logger.info(f"[{symbol}] EXECUTING: {signal.signal.value} {volume} lots "
                f"(confidence={signal.confidence:.0f}%)")
    for detail in signal.details:
        logger.info(f"  {detail}")

    await order_mgr.execute_signal(signal, volume, sl, tp)


if __name__ == "__main__":
    asyncio.run(main())
