from abc import ABC, abstractmethod
from datetime import date

from app.domain.models.price_bar import PriceBar
from app.domain.models.price_quote import PriceQuote
from app.domain.models.ticker_info import HoldersSummary, NewsArticle, TickerInfo


class MarketDataProvider(ABC):
    """Port for retrieving market data.

    Keeping this abstract means the rest of the app (services, API) never
    talks to yfinance directly, so swapping providers later (Alpha Vantage,
    Twelve Data, a broker feed) only means adding a new adapter.
    """

    @abstractmethod
    def get_price_history(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        ...

    @abstractmethod
    def get_latest_quote(self, ticker: str) -> PriceQuote | None:
        """Current price + previous close + currency for one ticker. Returns None
        (never raises) if the ticker can't be quoted right now - delisted, a typo,
        a momentary provider hiccup - so one bad ticker in a portfolio degrades
        that single position instead of taking down the whole summary."""
        ...

    @abstractmethod
    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        """Multiplier to convert an amount in `from_currency` into `to_currency`
        (1 unit of from = X units of to) - lets a portfolio holding both US and
        European tickers still add up to one meaningful total instead of
        summing dollars and euros as if they were the same unit. Returns None
        (never raises) if the rate can't be fetched right now."""
        ...

    @abstractmethod
    def get_bulk_price_history(self, tickers: list[str], start: date, end: date) -> dict[str, list[PriceBar]]:
        """One round trip for many tickers instead of one call each — the market
        screener needs this to stay fast over a universe of a hundred-plus tickers.
        Tickers that fail to fetch (delisted, rate-limited, no data) are simply
        omitted from the result rather than raising."""
        ...

    @abstractmethod
    def get_ticker_info(self, ticker: str) -> TickerInfo | None:
        """Fundamentals/company profile for a single ticker - only called for a
        one-off deep dive (never bulk-scanned), so a slower per-ticker call is fine.
        Returns None if the ticker doesn't exist or the provider has no data for it."""
        ...

    @abstractmethod
    def get_ticker_news(self, ticker: str, limit: int = 8) -> list[NewsArticle]:
        ...

    @abstractmethod
    def get_holders(self, ticker: str) -> HoldersSummary | None:
        """Institutional/insider ownership - a directional "who's holding this"
        signal, not authoritative (see YFinanceProvider for caveats). Returns
        None if the provider has no holders data for this ticker."""
        ...
