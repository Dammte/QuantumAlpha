import logging
from datetime import date

import pandas as pd
import yfinance as yf

from app.domain.interfaces.market_data_provider import MarketDataProvider
from app.domain.models.price_bar import PriceBar
from app.domain.models.price_quote import PriceQuote
from app.domain.models.ticker_info import HoldersSummary, InstitutionalHolder, NewsArticle, TickerInfo

logger = logging.getLogger(__name__)

MAX_INSTITUTIONAL_HOLDERS = 10

# London Stock Exchange quotes most stocks in pence (GBp/GBX), not pounds - not
# a real ISO 4217 currency an FX pair exists for, so it's normalized to GBP/100
# before looking up a rate (mirrors the same fix already applied client-side
# for display, see frontend/src/format.js).
PENCE_CURRENCIES = frozenset({"GBp", "GBX"})


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def _bars_from_frame(ticker: str, frame: pd.DataFrame) -> list[PriceBar]:
    frame = frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return [
        PriceBar(
            ticker=ticker,
            trade_date=index.date(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        )
        for index, row in frame.iterrows()
    ]


class YFinanceProvider(MarketDataProvider):
    def get_price_history(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        return _bars_from_frame(ticker, history)

    def get_latest_quote(self, ticker: str) -> PriceQuote | None:
        try:
            fast_info = yf.Ticker(ticker).fast_info
            price = _safe_float(fast_info.get("lastPrice"))
            if price is None:
                return None
            return PriceQuote(
                price=price,
                previous_close=_safe_float(fast_info.get("previousClose")),
                currency=str(fast_info.get("currency") or "USD"),
            )
        except Exception:
            logger.exception("Failed to fetch latest quote for %s", ticker)
            return None

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        pence_factor = 0.01 if from_currency in PENCE_CURRENCIES else 1.0
        normalized_from = "GBP" if from_currency in PENCE_CURRENCIES else from_currency
        if normalized_from == to_currency:
            return pence_factor
        try:
            fast_info = yf.Ticker(f"{normalized_from}{to_currency}=X").fast_info
            rate = _safe_float(fast_info.get("lastPrice"))
            return rate * pence_factor if rate is not None else None
        except Exception:
            logger.exception("Failed to fetch FX rate %s->%s", normalized_from, to_currency)
            return None

    def get_bulk_price_history(self, tickers: list[str], start: date, end: date) -> dict[str, list[PriceBar]]:
        if not tickers:
            return {}

        try:
            frame = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by="ticker",
                threads=True,
                auto_adjust=True,
                progress=False,
            )
        except Exception:
            logger.exception("Bulk yfinance download failed for %d tickers", len(tickers))
            return {}

        if frame.empty:
            return {}

        result: dict[str, list[PriceBar]] = {}
        for ticker in tickers:
            if ticker not in frame.columns.get_level_values(0):
                continue
            bars = _bars_from_frame(ticker, frame[ticker])
            if bars:
                result[ticker] = bars
        return result

    def get_ticker_info(self, ticker: str) -> TickerInfo | None:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            logger.exception("Failed to fetch info for %s", ticker)
            return None
        # yfinance returns a near-empty dict for a ticker that doesn't exist,
        # rather than raising - a handful of keys is the practical "does this exist" test.
        if not info or len(info) < 5:
            return None

        return TickerInfo(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("marketCap"),
            currency=info.get("currency"),
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            dividend_yield=info.get("dividendYield"),
            beta=info.get("beta"),
            average_volume=info.get("averageVolume"),
            analyst_recommendation=info.get("recommendationKey"),
            analyst_target_mean_price=info.get("targetMeanPrice"),
            analyst_opinion_count=info.get("numberOfAnalystOpinions"),
        )

    def get_ticker_news(self, ticker: str, limit: int = 8) -> list[NewsArticle]:
        try:
            raw_news = yf.Ticker(ticker).news or []
        except Exception:
            logger.exception("Failed to fetch news for %s", ticker)
            return []

        articles = []
        for item in raw_news[:limit]:
            content = item.get("content", item)
            title = content.get("title")
            if not title:
                continue
            provider = content.get("provider")
            publisher = provider.get("displayName") if isinstance(provider, dict) else content.get("publisher")
            canonical = content.get("canonicalUrl")
            link = canonical.get("url") if isinstance(canonical, dict) else content.get("link")
            articles.append(
                NewsArticle(
                    title=title,
                    publisher=publisher,
                    link=link,
                    published_at=content.get("pubDate") or content.get("displayTime"),
                )
            )
        return articles

    def get_holders(self, ticker: str) -> HoldersSummary | None:
        try:
            t = yf.Ticker(ticker)
            major = t.major_holders
            institutional = t.institutional_holders
        except Exception:
            logger.exception("Failed to fetch holders for %s", ticker)
            return None

        pct_institutions = None
        pct_insiders = None
        if major is not None and not major.empty and "Value" in major.columns:
            values = major["Value"]
            pct_institutions = _safe_float(values.get("institutionsPercentHeld"))
            pct_insiders = _safe_float(values.get("insidersPercentHeld"))

        top_holders: list[InstitutionalHolder] = []
        if institutional is not None and not institutional.empty:
            for _, row in institutional.head(MAX_INSTITUTIONAL_HOLDERS).iterrows():
                holder_name = row.get("Holder")
                if not holder_name or pd.isna(holder_name):
                    continue
                date_reported = row.get("Date Reported")
                top_holders.append(
                    InstitutionalHolder(
                        holder=str(holder_name),
                        shares=_safe_float(row.get("Shares")),
                        value=_safe_float(row.get("Value")),
                        pct_held=_safe_float(row.get("pctHeld")),
                        date_reported=str(date_reported) if date_reported is not None else None,
                    )
                )

        if pct_institutions is None and pct_insiders is None and not top_holders:
            return None

        return HoldersSummary(
            pct_held_by_institutions=pct_institutions,
            pct_held_by_insiders=pct_insiders,
            top_institutional_holders=top_holders,
        )
