from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class PriceBar:
    ticker: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
