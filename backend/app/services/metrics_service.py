"""Portfolio risk & performance metrics.

Pure functions over pandas Series so they stay easy to unit test and have
no dependency on FastAPI, the database, or a specific data provider.
Formulas follow the conventions used by empyrical/quantstats so results are
comparable with those libraries.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: pd.Series) -> pd.Series:
    returns = prices.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if returns.empty:
        return 0.0
    total_growth = (1 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0 or total_growth <= 0:
        return 0.0
    return float(total_growth ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    return float(excess_returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    if returns.empty:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    downside = excess_returns[excess_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if not downside_std:
        return 0.0
    return float(excess_returns.mean() / downside_std * np.sqrt(periods_per_year))


def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at every point, measured against the running peak.

    The peak has to include the pre-return baseline (1.0) as a candidate, not just the
    cumulative values that follow: if the very first return is negative, the wealth index
    never revisits that starting value, so a plain `cumprod().cummax()` would silently miss
    it as the true peak and understate the drawdown.
    """
    wealth_with_baseline = np.concatenate([[1.0], (1 + returns).cumprod().to_numpy()])
    running_max = np.maximum.accumulate(wealth_with_baseline)
    drawdown = wealth_with_baseline / running_max - 1
    return pd.Series(drawdown[1:], index=returns.index)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(_drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / abs(mdd))


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if not tail.empty else var


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    covariance = aligned.cov().iloc[0, 1]
    benchmark_variance = aligned.iloc[:, 1].var(ddof=1)
    return float(covariance / benchmark_variance) if benchmark_variance else 0.0


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Jensen's alpha, annualized."""
    b = beta(returns, benchmark_returns)
    portfolio_cagr = cagr(returns, periods_per_year)
    benchmark_cagr = cagr(benchmark_returns, periods_per_year)
    return float(portfolio_cagr - (risk_free_rate + b * (benchmark_cagr - risk_free_rate)))


def win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def best_day(returns: pd.Series) -> float:
    return float(returns.max()) if not returns.empty else 0.0


def worst_day(returns: pd.Series) -> float:
    return float(returns.min()) if not returns.empty else 0.0


def current_drawdown(returns: pd.Series) -> float:
    """How far the latest value sits below its running peak, i.e. the drawdown "right now"
    rather than the worst one over the whole window (`max_drawdown`)."""
    if returns.empty:
        return 0.0
    return float(_drawdown_series(returns).iloc[-1])


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    cumulative_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    win_rate: float
    current_drawdown: float
    best_day: float
    worst_day: float
    beta: float | None = None
    alpha: float | None = None


def time_weighted_returns(value_series: pd.Series, cash_flows: pd.Series | None = None) -> pd.Series:
    """Daily returns adjusted for external cash flows (new contributions/withdrawals).

    Without this adjustment, buying more of a position with fresh money shows up as a
    portfolio "gain" and inflates every return-based metric. `cash_flows` should hold the
    net amount added to the portfolio on each date (positive = money in, e.g. a buy;
    negative = money out, e.g. a sell), aligned to `value_series`'s index.
    """
    if cash_flows is None:
        return daily_returns(value_series)

    flows = cash_flows.reindex(value_series.index, fill_value=0.0)
    previous_value = value_series.shift(1)
    returns = (value_series - flows - previous_value) / previous_value
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def compute_portfolio_metrics_from_returns(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_returns: pd.Series | None = None,
) -> PortfolioMetrics:
    portfolio_beta = None
    portfolio_alpha = None
    if benchmark_returns is not None:
        portfolio_beta = beta(returns, benchmark_returns)
        portfolio_alpha = alpha(returns, benchmark_returns, risk_free_rate)

    return PortfolioMetrics(
        cumulative_return=cumulative_return(returns),
        cagr=cagr(returns),
        annualized_volatility=annualized_volatility(returns),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate),
        sortino_ratio=sortino_ratio(returns, risk_free_rate),
        max_drawdown=max_drawdown(returns),
        calmar_ratio=calmar_ratio(returns),
        var_95=historical_var(returns),
        cvar_95=conditional_var(returns),
        win_rate=win_rate(returns),
        current_drawdown=current_drawdown(returns),
        best_day=best_day(returns),
        worst_day=worst_day(returns),
        beta=portfolio_beta,
        alpha=portfolio_alpha,
    )


def compute_portfolio_metrics(
    prices: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_prices: pd.Series | None = None,
) -> PortfolioMetrics:
    returns = daily_returns(prices)
    benchmark_returns = daily_returns(benchmark_prices) if benchmark_prices is not None else None
    return compute_portfolio_metrics_from_returns(returns, risk_free_rate, benchmark_returns)
