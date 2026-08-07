import random
from collections.abc import Generator
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_asset_repository, get_market_data_provider, get_portfolio_repository
from app.domain.interfaces.market_data_provider import MarketDataProvider
from app.domain.models.price_bar import PriceBar
from app.domain.models.price_quote import PriceQuote
from app.domain.models.ticker_info import HoldersSummary, InstitutionalHolder, NewsArticle, TickerInfo
from app.infrastructure.db import models  # noqa: F401 - registers ORM tables on Base.metadata
from app.infrastructure.db.repositories.asset_repository import AssetRepository
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.db.session import Base
from app.main import app


class FakeMarketDataProvider(MarketDataProvider):
    """Deterministic in-memory provider so tests never hit the network."""

    def get_price_history(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        bars = []
        current = start
        price = 100.0
        while current <= end:
            bars.append(PriceBar(ticker, current, price, price, price, price, 1000))
            price += 1
            current += timedelta(days=1)
        return bars

    def get_latest_quote(self, ticker: str) -> PriceQuote | None:
        if ticker == "UNKNOWN":
            return None
        if ticker == "EURO.DE":
            return PriceQuote(price=100.0, previous_close=98.0, currency="EUR")
        return PriceQuote(price=150.0, previous_close=148.0, currency="USD")

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        if from_currency == to_currency:
            return 1.0
        # Deterministic fake cross-rate, just distinct enough from 1.0 that
        # multi-currency tests can tell a real conversion happened.
        return 1.1

    def get_bulk_price_history(self, tickers: list[str], start: date, end: date) -> dict[str, list[PriceBar]]:
        return {ticker: self._random_walk(ticker, start, end) for ticker in tickers}

    @staticmethod
    def _random_walk(ticker: str, start: date, end: date) -> list[PriceBar]:
        """A deterministic (seeded by ticker) pseudo-random walk, varied enough that
        screener/movers/sector tests can tell tickers apart by trend and volume."""
        if ticker == "UNKNOWN":
            return []
        rng = random.Random(ticker)
        drift = rng.uniform(-0.4, 0.6)
        price = 100.0
        bars = []
        current = start
        while current <= end:
            price = max(1.0, price + drift + rng.uniform(-2, 2))
            wiggle = abs(rng.uniform(0, 1.5))
            volume = rng.uniform(500, 5000)
            bars.append(PriceBar(ticker, current, price, price + wiggle, price - wiggle, price, volume))
            current += timedelta(days=1)
        return bars

    def get_ticker_info(self, ticker: str) -> TickerInfo | None:
        if ticker == "UNKNOWN":
            return None
        return TickerInfo(
            ticker=ticker,
            name=f"{ticker} Inc.",
            sector="Tecnología",
            industry="Software",
            market_cap=1_000_000_000.0,
            currency="USD",
            trailing_pe=25.0,
            forward_pe=22.0,
            dividend_yield=0.01,
            beta=1.1,
            average_volume=1_000_000.0,
            analyst_recommendation="buy",
            analyst_target_mean_price=175.0,
            analyst_opinion_count=20,
            revenue_growth=0.12,
            profit_margins=0.18,
            debt_to_equity=45.0,
        )

    def get_ticker_news(self, ticker: str, limit: int = 8) -> list[NewsArticle]:
        if ticker == "UNKNOWN":
            return []
        return [
            NewsArticle(
                title=f"{ticker} news {i}",
                publisher="Fake Wire",
                link="https://example.com",
                published_at="2026-01-01T00:00:00Z",
            )
            for i in range(min(limit, 3))
        ]

    def get_holders(self, ticker: str) -> HoldersSummary | None:
        if ticker == "UNKNOWN":
            return None
        return HoldersSummary(
            pct_held_by_institutions=0.62,
            pct_held_by_insiders=0.03,
            top_institutional_holders=[
                InstitutionalHolder(
                    holder="Fake Capital Management",
                    shares=1_000_000.0,
                    value=150_000_000.0,
                    pct_held=0.05,
                    date_reported="2026-06-30",
                )
            ],
        )


@pytest.fixture()
def engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(engine) -> Generator[Session, None, None]:
    testing_session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = testing_session_local()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_portfolio_repository] = lambda: PortfolioRepository(db_session)
    app.dependency_overrides[get_asset_repository] = lambda: AssetRepository(db_session)
    app.dependency_overrides[get_market_data_provider] = lambda: FakeMarketDataProvider()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
