from pydantic import BaseModel


class PortfolioMetricsResponse(BaseModel):
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
