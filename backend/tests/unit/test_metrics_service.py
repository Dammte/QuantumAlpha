import numpy as np
import pandas as pd
import pytest

from app.services import metrics_service as m


@pytest.fixture
def rising_prices() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    values = 100 * (1.0005 ** np.arange(252))
    return pd.Series(values, index=dates)


def test_daily_returns_length(rising_prices: pd.Series) -> None:
    returns = m.daily_returns(rising_prices)
    assert len(returns) == len(rising_prices) - 1


def test_daily_returns_drops_infinite_jump_from_zero() -> None:
    # A portfolio worth 0 before its first transaction settles, then positive, must not
    # produce an infinite one-day return that poisons every downstream metric.
    prices = pd.Series([0.0, 0.0, 100.0, 105.0])
    returns = m.daily_returns(prices)
    assert np.isfinite(returns).all()


def test_cumulative_return_positive_for_rising_prices(rising_prices: pd.Series) -> None:
    returns = m.daily_returns(rising_prices)
    assert m.cumulative_return(returns) > 0


def test_max_drawdown_is_never_positive(rising_prices: pd.Series) -> None:
    returns = m.daily_returns(rising_prices)
    assert m.max_drawdown(returns) <= 0


def test_max_drawdown_detects_a_drop() -> None:
    prices = pd.Series([100, 110, 90, 95])
    returns = m.daily_returns(prices)
    # Peak at 110, trough at 90 -> drawdown of (90-110)/110
    assert m.max_drawdown(returns) == pytest.approx((90 - 110) / 110)


def test_sharpe_ratio_zero_when_no_volatility() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])
    assert m.sharpe_ratio(returns) == 0.0


def test_beta_of_series_against_itself_is_one() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    assert m.beta(returns, returns) == pytest.approx(1.0)


def test_win_rate_counts_positive_days() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.03, 0.0])
    assert m.win_rate(returns) == pytest.approx(2 / 5)


def test_compute_portfolio_metrics_returns_dataclass(rising_prices: pd.Series) -> None:
    metrics = m.compute_portfolio_metrics(rising_prices, risk_free_rate=0.02)
    assert metrics.cumulative_return > 0
    assert metrics.beta is None
    assert metrics.alpha is None


def test_time_weighted_returns_ignores_new_contributions() -> None:
    # Prices never move; the only reason value jumps is a fresh deposit on day 2.
    # A naive pct_change on value would read that as a 150% gain - TWR must read it as 0.
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    value_series = pd.Series([100.0, 100.0, 250.0, 250.0], index=dates)
    cash_flows = pd.Series([100.0, 0.0, 150.0, 0.0], index=dates)

    returns = m.time_weighted_returns(value_series, cash_flows)

    assert returns.eq(0.0).all()


def test_compute_portfolio_metrics_with_benchmark(rising_prices: pd.Series) -> None:
    metrics = m.compute_portfolio_metrics(rising_prices, risk_free_rate=0.02, benchmark_prices=rising_prices)
    assert metrics.beta == pytest.approx(1.0)
    assert metrics.alpha == pytest.approx(0.0, abs=1e-9)


def test_best_and_worst_day() -> None:
    returns = pd.Series([0.02, -0.05, 0.01, 0.07, -0.01])
    assert m.best_day(returns) == pytest.approx(0.07)
    assert m.worst_day(returns) == pytest.approx(-0.05)


def test_current_drawdown_is_zero_at_a_new_high() -> None:
    prices = pd.Series([100, 90, 110])  # ends above the prior peak
    returns = m.daily_returns(prices)
    assert m.current_drawdown(returns) == pytest.approx(0.0)


def test_current_drawdown_reflects_latest_dip_not_the_worst_one() -> None:
    prices = pd.Series([100, 60, 120, 108])  # -40% mid-series, but ends only 10% off its peak
    returns = m.daily_returns(prices)
    assert m.current_drawdown(returns) == pytest.approx((108 - 120) / 120)
    assert m.max_drawdown(returns) == pytest.approx((60 - 100) / 100)
