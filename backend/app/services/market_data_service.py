from datetime import date

import pandas as pd

from app.domain.interfaces.market_data_provider import MarketDataProvider
from app.domain.models.price_quote import PriceQuote
from app.domain.models.ticker_info import HoldersSummary, NewsArticle, TickerInfo


class MarketDataService:
    """Thin, provider-agnostic wrapper used by the rest of the app."""

    def __init__(self, provider: MarketDataProvider) -> None:
        self.provider = provider

    def get_close_price_series(self, ticker: str, start: date, end: date) -> pd.Series:
        bars = self.provider.get_price_history(ticker, start, end)
        if not bars:
            return pd.Series(dtype=float, name=ticker)
        index = [bar.trade_date for bar in bars]
        values = [bar.close for bar in bars]
        return pd.Series(values, index=pd.to_datetime(index), name=ticker)

    def get_latest_quotes(self, tickers: list[str]) -> dict[str, PriceQuote]:
        """ticker -> quote, for every ticker a live quote could be fetched for.
        Tickers that fail (delisted, mistyped, momentary provider hiccup) are
        simply omitted rather than raising - see `PriceQuote`."""
        quotes = {}
        for ticker in tickers:
            quote = self.provider.get_latest_quote(ticker)
            if quote is not None:
                quotes[ticker] = quote
        return quotes

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        return self.provider.get_fx_rate(from_currency, to_currency)

    def get_bulk_ohlcv(self, tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        """ticker -> OHLCV DataFrame (columns: open, high, low, close, volume; DatetimeIndex).
        Tickers with no data are omitted rather than raising."""
        bars_by_ticker = self.provider.get_bulk_price_history(tickers, start, end)
        result = {}
        for ticker, bars in bars_by_ticker.items():
            frame = pd.DataFrame(
                {
                    "open": [b.open for b in bars],
                    "high": [b.high for b in bars],
                    "low": [b.low for b in bars],
                    "close": [b.close for b in bars],
                    "volume": [b.volume for b in bars],
                },
                index=pd.to_datetime([b.trade_date for b in bars]),
            )
            if not frame.empty:
                result[ticker] = frame
        return result

    def get_ticker_info(self, ticker: str) -> TickerInfo | None:
        return self.provider.get_ticker_info(ticker)

    def get_ticker_news(self, ticker: str, limit: int = 8) -> list[NewsArticle]:
        return self.provider.get_ticker_news(ticker, limit)

    def get_holders(self, ticker: str) -> HoldersSummary | None:
        return self.provider.get_holders(ticker)
