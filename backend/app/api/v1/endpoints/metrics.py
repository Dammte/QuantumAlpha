from dataclasses import asdict
from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_portfolio_service
from app.core.config import Settings, get_settings
from app.schemas.metrics import PortfolioMetricsResponse
from app.services.portfolio_service import PortfolioNotFoundError, PortfolioService

router = APIRouter(prefix="/portfolios/{portfolio_id}/metrics", tags=["metrics"])


@router.get("", response_model=PortfolioMetricsResponse)
def get_portfolio_metrics(
    portfolio_id: int,
    service: Annotated[PortfolioService, Depends(get_portfolio_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    start: date | None = Query(default=None, description="Defaults to one year before `end`"),
    end: date | None = Query(default=None, description="Defaults to today"),
    benchmark_ticker: str | None = Query(default=None, description="e.g. ^GSPC for the S&P 500"),
) -> PortfolioMetricsResponse:
    end = end or date.today()
    start = start or (end - timedelta(days=365))

    try:
        metrics = service.get_portfolio_metrics(
            portfolio_id,
            start=start,
            end=end,
            risk_free_rate=settings.risk_free_rate,
            benchmark_ticker=benchmark_ticker,
        )
    except PortfolioNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return PortfolioMetricsResponse(**asdict(metrics))
