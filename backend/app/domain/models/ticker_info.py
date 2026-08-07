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
    analyst_recommendation: str | None
    analyst_target_mean_price: float | None
    analyst_opinion_count: int | None


@dataclass(frozen=True, slots=True)
class NewsArticle:
    title: str
    publisher: str | None
    link: str | None
    published_at: str | None


@dataclass(frozen=True, slots=True)
class InstitutionalHolder:
    holder: str
    shares: float | None
    value: float | None
    pct_held: float | None
    date_reported: str | None


@dataclass(frozen=True, slots=True)
class HoldersSummary:
    pct_held_by_institutions: float | None
    pct_held_by_insiders: float | None
    top_institutional_holders: list[InstitutionalHolder]
