from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_portfolio_repository, get_portfolio_service
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.schemas.history import HistoryPoint, PortfolioHistoryResponse
from app.services.portfolio_service import PortfolioNotFoundError, PortfolioService

router = APIRouter(prefix="/portfolios/{portfolio_id}/history", tags=["history"])


@router.get("", response_model=PortfolioHistoryResponse)
def get_portfolio_history(
    portfolio_id: int,
    service: Annotated[PortfolioService, Depends(get_portfolio_service)],
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
    start: date | None = Query(default=None, description="Defaults to one year before `end`"),
    end: date | None = Query(default=None, description="Defaults to today"),
    benchmark_ticker: str | None = Query(default=None, description="e.g. ^GSPC for the S&P 500"),
) -> PortfolioHistoryResponse:
    orm = repository.get(portfolio_id)
    if orm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")

    end = end or date.today()
    start = start or (end - timedelta(days=365))

    try:
        value_series, benchmark_series = service.get_portfolio_history(
            portfolio_id, start=start, end=end, benchmark_ticker=benchmark_ticker
        )
    except PortfolioNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    points = [HistoryPoint(date=ts.date(), value=value) for ts, value in value_series.items()]
    benchmark_points = None
    if benchmark_series is not None:
        benchmark_points = [HistoryPoint(date=ts.date(), value=value) for ts, value in benchmark_series.items()]

    return PortfolioHistoryResponse(
        base_currency=orm.base_currency, points=points, benchmark_points=benchmark_points
    )
