import logging
from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_recommendation_snapshot_repository, get_ticker_analysis_service
from app.domain.models.ticker_analysis import TickerAnalysis
from app.infrastructure.db.repositories.recommendation_snapshot_repository import RecommendationSnapshotRepository
from app.schemas.ticker_analysis import RecommendationSnapshotResponse, TickerAnalysisResponse
from app.services.levels_engine import GATE_VERSION
from app.services.ticker_analysis_service import TickerAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market/tickers", tags=["ticker-analysis"])


def _to_response(analysis: TickerAnalysis) -> TickerAnalysisResponse:
    data = asdict(analysis)
    data["trend"] = analysis.trend.value
    data["stage"] = analysis.stage.value if analysis.stage else None
    data["market_trend"] = analysis.market_trend.value if analysis.market_trend else None
    return TickerAnalysisResponse(**data)


def _save_snapshot_best_effort(
    repository: RecommendationSnapshotRepository, analysis: TickerAnalysis, horizon: str
) -> None:
    """The audit trail (see RecommendationSnapshotORM's docstring) must never
    be able to break the actual analysis response a user is waiting on - a
    logging failure isn't a reason to 500 an otherwise-successful read.

    2026-09 (reconstruction, Fase 4): the live read is now `GateResult`
    (pass/fail conditions), not the old weighted `Recommendation` - remapped
    onto this table's unchanged verdict/score/factors columns rather than
    migrating the schema: `verdict` is "comprar"/"esperar" (the gate has no
    third, "evitar"-strength state - see `portfolio_risk_service.py`'s own
    Fase 4 note for why that state doesn't carry over), `score` is how many
    of the gate's conditions passed, and each condition becomes a `factors`
    entry with `points` 1 (passed) or 0 (didn't) - the same {label, points,
    triggered} shape this column already had, so no migration needed for a
    purely internal representation change."""
    try:
        gate = analysis.gate
        factors = [
            {"label": c.label, "points": 1 if c.passed else 0, "triggered": c.passed} for c in gate.conditions
        ]
        repository.save(
            ticker=analysis.ticker,
            verdict="comprar" if gate.passes else "esperar",
            score=sum(1 for c in gate.conditions if c.passed),
            price=analysis.price,
            currency=analysis.currency or "USD",
            horizon=horizon,
            engine_version=GATE_VERSION,
            factors=factors,
        )
    except Exception:
        logger.exception("Failed to persist recommendation snapshot for %s", analysis.ticker)


@router.get("/{ticker}/analysis", response_model=TickerAnalysisResponse)
def get_ticker_analysis(
    ticker: str,
    service: Annotated[TickerAnalysisService, Depends(get_ticker_analysis_service)],
    snapshot_repository: Annotated[
        RecommendationSnapshotRepository, Depends(get_recommendation_snapshot_repository)
    ],
    horizon: Annotated[
        Literal["1m", "3m", "6m"], Query(description="Horizonte de la recomendación, persistido en el snapshot")
    ] = "3m",
) -> TickerAnalysisResponse:
    try:
        analysis = service.analyze(ticker, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    _save_snapshot_best_effort(snapshot_repository, analysis, horizon)
    return _to_response(analysis)


@router.get("/{ticker}/history", response_model=list[RecommendationSnapshotResponse])
def get_ticker_recommendation_history(
    ticker: str,
    snapshot_repository: Annotated[
        RecommendationSnapshotRepository, Depends(get_recommendation_snapshot_repository)
    ],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[RecommendationSnapshotResponse]:
    """Every past verdict this ticker actually got shown, oldest last - the
    literal, queryable answer to "¿qué decía el sistema de esto antes?"."""
    snapshots = snapshot_repository.list_for_ticker(ticker, limit=limit)
    return [
        RecommendationSnapshotResponse(
            id=s.id,
            ticker=s.ticker,
            created_at=s.created_at,
            verdict=s.verdict,
            score=s.score,
            price=float(s.price),
            currency=s.currency,
            horizon=s.horizon,
            engine_version=s.engine_version,
            factors=s.factors,
        )
        for s in snapshots
    ]
