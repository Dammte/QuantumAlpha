from datetime import date

from pydantic import BaseModel


class HistoryPoint(BaseModel):
    date: date
    value: float


class PortfolioHistoryResponse(BaseModel):
    base_currency: str
    points: list[HistoryPoint]
    benchmark_points: list[HistoryPoint] | None = None
