from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domain.interfaces.market_data_provider import MarketDataProvider
from app.infrastructure.db.repositories.asset_repository import AssetRepository
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.db.repositories.recommendation_snapshot_repository import RecommendationSnapshotRepository
from app.infrastructure.db.session import get_db
from app.infrastructure.macro_data.fred_client import FredClient
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.services.macro_data_service import MacroDataService
from app.services.market_context_service import MarketContextService
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.portfolio_risk_service import PortfolioRiskService
from app.services.portfolio_service import PortfolioService
from app.services.premium_watchlist_service import PremiumWatchlistService
from app.services.ticker_analysis_service import TickerAnalysisService

DbSession = Annotated[Session, Depends(get_db)]


@lru_cache
def get_market_data_provider() -> MarketDataProvider:
    settings: Settings = get_settings()
    if settings.market_data_provider == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unsupported market data provider: {settings.market_data_provider}")


@lru_cache
def get_market_data_service(
    provider: Annotated[MarketDataProvider, Depends(get_market_data_provider)],
) -> MarketDataService:
    return MarketDataService(provider)


@lru_cache
def get_market_screener_service(
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
) -> MarketScreenerService:
    """Cached as a singleton: MarketScreenerService keeps its own in-process
    result cache (see CACHE_TTL in market_screener_service.py), which only helps
    if the same instance is reused across requests."""
    return MarketScreenerService(market_data)


@lru_cache
def get_market_context_service(
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
) -> MarketContextService:
    return MarketContextService(market_data)


@lru_cache
def get_fred_client() -> FredClient:
    settings: Settings = get_settings()
    return FredClient(api_key=settings.fred_api_key)


@lru_cache
def get_macro_data_service(
    client: Annotated[FredClient, Depends(get_fred_client)],
) -> MacroDataService:
    return MacroDataService(client)


def get_ticker_analysis_service(
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
    screener: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
) -> TickerAnalysisService:
    return TickerAnalysisService(market_data, screener)


@lru_cache
def get_premium_watchlist_service(
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
    screener: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
) -> PremiumWatchlistService:
    """Cached as a singleton: PremiumWatchlistService keeps its own per-tier
    in-process cache (see CACHE_TTL in premium_watchlist_service.py), which only
    helps if the same instance is reused across requests."""
    return PremiumWatchlistService(market_data, screener)


@lru_cache
def get_portfolio_risk_service() -> PortfolioRiskService:
    """Cached as a singleton: PortfolioRiskService keeps its own per-ticker
    in-process cache (see CACHE_TTL in portfolio_risk_service.py), which only
    helps if the same instance is reused across requests."""
    return PortfolioRiskService()


def get_portfolio_repository(db: DbSession) -> PortfolioRepository:
    return PortfolioRepository(db)


def get_recommendation_snapshot_repository(db: DbSession) -> RecommendationSnapshotRepository:
    return RecommendationSnapshotRepository(db)


def get_asset_repository(db: DbSession) -> AssetRepository:
    return AssetRepository(db)


def get_portfolio_service(
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
) -> PortfolioService:
    return PortfolioService(repository, market_data)
