"""Tests for portfolio-level risk management (correlation + exposure limits)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.broker.base import AccountInfo, Side, Trade
from src.risk.correlation import CorrelationMatrix
from src.risk.portfolio import PortfolioRiskManager


# --------------- helpers ---------------

def _make_correlated_candles(
    n: int = 150,
    correlation: float = 0.95,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two DataFrames whose close-price returns have *approximately*
    the requested Pearson correlation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, size=n)
    noise = rng.normal(0, 0.01, size=n)
    series_a = 100 + np.cumsum(base)
    series_b = 100 + np.cumsum(correlation * base + (1 - abs(correlation)) * noise)

    def _to_df(closes: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": np.arange(len(closes)) * 3600,
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.ones(len(closes)) * 1000,
        })

    return _to_df(series_a), _to_df(series_b)


def _make_uncorrelated_candles(
    n: int = 150,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two DataFrames with near-zero correlation."""
    rng = np.random.default_rng(seed)
    series_a = 100 + np.cumsum(rng.normal(0, 0.01, size=n))
    series_b = 100 + np.cumsum(rng.normal(0, 0.01, size=n))

    def _to_df(closes: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": np.arange(len(closes)) * 3600,
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.ones(len(closes)) * 1000,
        })

    return _to_df(series_a), _to_df(series_b)


def _make_trade(symbol: str, volume: float = 0.1, price: float = 100.0,
                trade_id: int = 1) -> Trade:
    return Trade(
        trade_id=trade_id,
        symbol=symbol,
        side=Side.BUY,
        volume=volume,
        open_price=price,
        open_time=0,
    )


def _make_account(balance: float = 10000, equity: float = 10000) -> AccountInfo:
    return AccountInfo(
        balance=balance,
        equity=equity,
        margin=0,
        free_margin=equity,
        margin_level=0,
        currency="USD",
    )


# --------------- CorrelationMatrix tests ---------------

class TestCorrelationMatrixUpdate:
    def test_computes_correct_high_correlation(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150, correlation=0.95)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        corr = cm.get_correlation("SYM_A", "SYM_B")
        assert corr > 0.5, f"Expected high positive correlation, got {corr}"

    def test_computes_low_correlation_for_independent(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_uncorrelated_candles(n=150)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        corr = cm.get_correlation("SYM_A", "SYM_B")
        assert abs(corr) < 0.5, f"Expected low correlation, got {corr}"

    def test_self_correlation_is_one(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        assert cm.get_correlation("SYM_A", "SYM_A") == pytest.approx(1.0)

    def test_returns_none_matrix_with_single_symbol(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, _ = _make_correlated_candles(n=150)
        cm.update({"SYM_A": df_a})
        assert cm._matrix is None
        assert cm.get_correlation("SYM_A", "SYM_B") == 0.0

    def test_skips_symbols_with_insufficient_data(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150)
        short_df = df_a.head(10)  # only 10 rows, below lookback
        cm.update({"SYM_A": short_df, "SYM_B": df_b})
        # Only one valid symbol -> matrix should be None
        assert cm._matrix is None

    def test_empty_data_handled(self):
        cm = CorrelationMatrix(lookback=100)
        cm.update({})
        assert cm._matrix is None
        assert cm.get_correlation("X", "Y") == 0.0

    def test_empty_dataframe_handled(self):
        cm = CorrelationMatrix(lookback=100)
        cm.update({"SYM_A": pd.DataFrame(), "SYM_B": pd.DataFrame()})
        assert cm._matrix is None

    def test_unknown_symbol_returns_zero(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        assert cm.get_correlation("SYM_A", "UNKNOWN") == 0.0
        assert cm.get_correlation("UNKNOWN", "SYM_A") == 0.0


class TestCorrelationMatrixGetCorrelatedSymbols:
    def test_returns_correlated_above_threshold(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150, correlation=0.95)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        result = cm.get_correlated_symbols("SYM_A", threshold=0.5)
        symbols = [sym for sym, _ in result]
        assert "SYM_B" in symbols

    def test_excludes_below_threshold(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_uncorrelated_candles(n=150)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        result = cm.get_correlated_symbols("SYM_A", threshold=0.7)
        symbols = [sym for sym, _ in result]
        assert "SYM_B" not in symbols

    def test_unknown_symbol_returns_empty(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        assert cm.get_correlated_symbols("UNKNOWN", threshold=0.5) == []

    def test_no_matrix_returns_empty(self):
        cm = CorrelationMatrix(lookback=100)
        assert cm.get_correlated_symbols("SYM_A") == []

    def test_sorted_by_absolute_correlation(self):
        cm = CorrelationMatrix(lookback=100)
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, size=150)
        noise1 = rng.normal(0, 0.01, size=150)
        noise2 = rng.normal(0, 0.01, size=150)
        series_a = 100 + np.cumsum(base)
        # High correlation
        series_b = 100 + np.cumsum(0.95 * base + 0.05 * noise1)
        # Moderate correlation
        series_c = 100 + np.cumsum(0.60 * base + 0.40 * noise2)

        def _to_df(closes):
            return pd.DataFrame({
                "timestamp": np.arange(len(closes)) * 3600,
                "open": closes, "high": closes + 0.1,
                "low": closes - 0.1, "close": closes,
                "volume": np.ones(len(closes)) * 1000,
            })

        cm.update({"A": _to_df(series_a), "B": _to_df(series_b), "C": _to_df(series_c)})
        result = cm.get_correlated_symbols("A", threshold=0.3)
        if len(result) >= 2:
            # First result should have higher absolute correlation
            assert abs(result[0][1]) >= abs(result[1][1])


class TestCorrelationMatrixNeedsUpdate:
    def test_needs_update_initially(self):
        cm = CorrelationMatrix(lookback=100, update_interval=3600)
        assert cm.needs_update is True

    def test_no_update_needed_after_refresh(self):
        cm = CorrelationMatrix(lookback=100, update_interval=3600)
        df_a, df_b = _make_correlated_candles(n=150)
        cm.update({"SYM_A": df_a, "SYM_B": df_b})
        assert cm.needs_update is False


# --------------- PortfolioRiskManager tests ---------------

class TestPortfolioRiskBlocksMaxPositions:
    def test_blocks_when_max_correlated_positions_reached(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150, correlation=0.95)
        cm.update({"EURUSD": df_a, "GBPUSD": df_b})

        prm = PortfolioRiskManager(
            {"max_sector_positions": 2, "correlation_threshold": 0.5,
             "max_correlated_exposure": 0.10},
            cm,
        )

        open_trades = [
            _make_trade("EURUSD", volume=0.1, trade_id=1),
            _make_trade("GBPUSD", volume=0.1, trade_id=2),
        ]
        account = _make_account()

        allowed, reason, vol = prm.check_portfolio_risk(
            "EURUSD", Side.BUY, 0.1, open_trades, account,
        )
        assert allowed is False
        assert "Max correlated positions" in reason
        assert vol == 0.0

    def test_allows_when_under_max_positions(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150, correlation=0.95)
        cm.update({"EURUSD": df_a, "GBPUSD": df_b})

        prm = PortfolioRiskManager(
            {"max_sector_positions": 3, "correlation_threshold": 0.5,
             "max_correlated_exposure": 1.0},  # high limit so exposure won't block
            cm,
        )

        open_trades = [_make_trade("EURUSD", volume=0.1, trade_id=1)]
        account = _make_account()

        allowed, reason, vol = prm.check_portfolio_risk(
            "GBPUSD", Side.BUY, 0.1, open_trades, account,
        )
        assert allowed is True
        assert vol == 0.1


class TestPortfolioRiskAdjustsVolume:
    def test_reduces_volume_when_exposure_limit_approached(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150, correlation=0.95)
        cm.update({"EURUSD": df_a, "GBPUSD": df_b})

        prm = PortfolioRiskManager(
            {"max_sector_positions": 5, "correlation_threshold": 0.5,
             "max_correlated_exposure": 0.04},
            cm,
        )

        # Current exposure from existing trade: 0.1 * 100 = 10
        open_trades = [_make_trade("EURUSD", volume=0.1, price=100.0, trade_id=1)]
        account = _make_account(balance=10000, equity=10000)

        # max_exposure = 10000 * 0.04 = 400
        # current_exposure = 10
        # proposed_exposure = large_vol * 10000/100 = large_vol * 100
        # Try a volume that would blow the limit
        allowed, reason, vol = prm.check_portfolio_risk(
            "GBPUSD", Side.BUY, 5.0, open_trades, account,
        )
        # proposed_exposure = 5.0 * 100 = 500; 10 + 500 > 400 -> reduced
        assert allowed is True
        assert "adjusted" in reason.lower()
        assert vol < 5.0
        assert vol > 0

    def test_blocks_when_exposure_fully_consumed(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150, correlation=0.95)
        cm.update({"EURUSD": df_a, "GBPUSD": df_b})

        prm = PortfolioRiskManager(
            {"max_sector_positions": 5, "correlation_threshold": 0.5,
             "max_correlated_exposure": 0.002},  # tiny limit: 10000*0.002=20
            cm,
        )

        # Current exposure already exceeds the limit
        open_trades = [_make_trade("EURUSD", volume=0.5, price=100.0, trade_id=1)]
        account = _make_account(balance=10000, equity=10000)
        # current_exposure = 0.5 * 100 = 50  > max_exposure = 20

        allowed, reason, vol = prm.check_portfolio_risk(
            "GBPUSD", Side.BUY, 0.1, open_trades, account,
        )
        assert allowed is False
        assert vol == 0.0


class TestPortfolioRiskEdgeCases:
    def test_no_correlation_data_allows_trade(self):
        """When correlation matrix has no data, trade passes through."""
        cm = CorrelationMatrix(lookback=100)
        # Don't call update -> matrix is None

        prm = PortfolioRiskManager(
            {"max_sector_positions": 3, "correlation_threshold": 0.7,
             "max_correlated_exposure": 0.04},
            cm,
        )

        account = _make_account()
        allowed, reason, vol = prm.check_portfolio_risk(
            "EURUSD", Side.BUY, 0.5, [], account,
        )
        assert allowed is True
        assert vol == 0.5

    def test_no_open_trades_allows_trade(self):
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_correlated_candles(n=150, correlation=0.95)
        cm.update({"EURUSD": df_a, "GBPUSD": df_b})

        prm = PortfolioRiskManager(
            {"max_sector_positions": 3, "correlation_threshold": 0.5,
             "max_correlated_exposure": 0.04},
            cm,
        )
        account = _make_account()

        allowed, reason, vol = prm.check_portfolio_risk(
            "EURUSD", Side.BUY, 0.1, [], account,
        )
        assert allowed is True
        assert vol == 0.1

    def test_uncorrelated_symbol_not_counted(self):
        """Trades in uncorrelated symbols should not count against the group."""
        cm = CorrelationMatrix(lookback=100)
        df_a, df_b = _make_uncorrelated_candles(n=150)
        cm.update({"EURUSD": df_a, "GOLD": df_b})

        prm = PortfolioRiskManager(
            {"max_sector_positions": 2, "correlation_threshold": 0.7,
             "max_correlated_exposure": 0.04},
            cm,
        )

        # GOLD trade should not block EURUSD since they are uncorrelated
        open_trades = [
            _make_trade("GOLD", volume=0.1, trade_id=1),
            _make_trade("GOLD", volume=0.1, trade_id=2),
        ]
        account = _make_account()

        allowed, _, vol = prm.check_portfolio_risk(
            "EURUSD", Side.BUY, 0.1, open_trades, account,
        )
        # GOLD is not correlated with EURUSD at threshold=0.7, so only
        # the EURUSD symbol itself counts (0 trades) -> allowed
        assert allowed is True
