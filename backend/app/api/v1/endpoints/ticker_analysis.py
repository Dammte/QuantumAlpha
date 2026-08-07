from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_ticker_analysis_service
from app.domain.models.ticker_analysis import TickerAnalysis
from app.schemas.ticker_analysis import TickerAnalysisResponse
from app.services.ticker_analysis_service import TickerAnalysisService

router = APIRouter(prefix="/market/tickers", tags=["ticker-analysis"])


def _to_response(analysis: TickerAnalysis) -> TickerAnalysisResponse:
    data = asdict(analysis)
    data["trend"] = analysis.trend.value
    data["stage"] = analysis.stage.value if analysis.stage else None
    return TickerAnalysisResponse(**data)


@router.get("/{ticker}/analysis", response_model=TickerAnalysisResponse)
def get_ticker_analysis(
    ticker: str,
    service: Annotated[TickerAnalysisService, Depends(get_ticker_analysis_service)],
    horizon: Annotated[
        Literal["1m", "3m", "6m"], Query(description="Horizonte de la simulación Monte Carlo")
    ] = "3m",
) -> TickerAnalysisResponse:
    try:
        analysis = service.analyze(ticker, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return _to_response(analysis)
