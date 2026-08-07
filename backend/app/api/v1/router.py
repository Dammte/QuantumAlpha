from fastapi import APIRouter

from app.api.v1.endpoints import history, market, metrics, portfolios, ticker_analysis

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(portfolios.router)
api_router.include_router(metrics.router)
api_router.include_router(history.router)
api_router.include_router(market.router)
api_router.include_router(ticker_analysis.router)
