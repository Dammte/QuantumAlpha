from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PriceQuote:
    """A single real-time-ish quote (yfinance's `fast_info`) - just enough to price
    a portfolio position and show today's move, without a full history download."""

    price: float
    previous_close: float | None
    currency: str
