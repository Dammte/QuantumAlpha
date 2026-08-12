from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PortfolioCreate(BaseModel):
    name: str
    base_currency: str = "USD"


class PortfolioRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    base_currency: str
    created_at: datetime


class PositionRead(BaseModel):
    ticker: str
    currency: str
    quantity: float
    average_cost: float
    current_price: float | None  # None when a live quote couldn't be fetched right now
    previous_close: float | None
    market_value: float | None  # in the position's own currency - what you'd see quoting it directly
    market_value_base: float | None  # in the portfolio's base currency - use this for weights/totals
    cost_basis: float
    unrealized_pnl: float | None
    unrealized_pnl_pct: float | None
    day_change: float | None
    day_change_pct: float | None
    realized_pnl: float  # banked from partial sells of this ticker while still holding some


class PortfolioSummary(BaseModel):
    id: int
    name: str
    base_currency: str
    positions: list[PositionRead]
    total_market_value: float
    total_cost_basis: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    total_day_change: float
    total_day_change_pct: float | None
    cash_balance: float  # deposits + sale proceeds - withdrawals - purchase cost, in base_currency
    total_portfolio_value: float  # total_market_value + cash_balance - the true "what is this worth" figure
