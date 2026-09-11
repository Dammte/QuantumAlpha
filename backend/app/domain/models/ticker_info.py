from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TickerInfo:
    ticker: str
    name: str | None
    sector: str | None
    industry: str | None
    market_cap: float | None
    currency: str | None
    trailing_pe: float | None
    forward_pe: float | None
    dividend_yield: float | None
    beta: float | None
    average_volume: float | None
    revenue_growth: float | None
    profit_margins: float | None
    debt_to_equity: float | None


@dataclass(frozen=True, slots=True)
class NewsArticle:
    title: str
    publisher: str | None
    link: str | None
    published_at: str | None
