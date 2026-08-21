from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import (
    get_market_data_service,
    get_position_signal_snapshot_repository,
    get_recommendation_snapshot_repository,
)
from app.infrastructure.db.repositories.position_signal_snapshot_repository import PositionSignalSnapshotRepository
from app.infrastructure.db.repositories.recommendation_snapshot_repository import RecommendationSnapshotRepository
from app.schemas.system import FactorAblationReportResponse, SignalPerformanceResponse
from app.services import ablation_report_service as ars
from app.services import signal_performance_service as sps
from app.services.market_data_service import MarketDataService

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/signal-performance", response_model=SignalPerformanceResponse)
def get_signal_performance(
    recommendation_repo: Annotated[
        RecommendationSnapshotRepository, Depends(get_recommendation_snapshot_repository)
    ],
    position_repo: Annotated[PositionSignalSnapshotRepository, Depends(get_position_signal_snapshot_repository)],
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
) -> SignalPerformanceResponse:
    """Fase 0 (docs/quant_methodology.md): did the system's past verdicts and
    position signals actually work out - hit rate, mean/median forward
    return per verdict/signal at 5/10/21/63 sessions, and the concrete
    by-name list of `hold` signals immediately followed by a real drop
    (>5% within 10 sessions). See `signal_performance_service.py`.

    Answers with whatever history has accumulated since each snapshot type
    started being recorded - `RecommendationSnapshotORM` (buy-side verdicts)
    has existed since the audit that added signal traceability;
    `PositionSignalSnapshotORM` (position-level signal/exit_urgency) only
    since this phase, so its own numbers only cover what's happened *since*
    this endpoint shipped, never retroactively - that data was never kept
    before now."""
    recommendation_snapshots = recommendation_repo.list_all()
    position_snapshots = position_repo.list_all()
    report = sps.build_signal_performance_report(recommendation_snapshots, position_snapshots, market_data)
    return SignalPerformanceResponse(
        verdict_outcomes=[asdict(o) for o in report.verdict_outcomes],
        signal_outcomes=[asdict(o) for o in report.signal_outcomes],
        false_negatives=[asdict(f) for f in report.false_negatives],
        as_of=report.as_of,
    )


@router.get("/factor-ablation", response_model=FactorAblationReportResponse)
def get_factor_ablation_report(
    horizon_days: Annotated[
        int, Query(description="Horizonte en sesiones - ver ablation_report_service.AVAILABLE_HORIZONS_DAYS")
    ] = 21,
) -> FactorAblationReportResponse:
    """Peso actual vs. efecto medido (signo, IC, significancia) por factor de
    `recommendation_engine.py`, a un horizonte dado - lee directamente el CSV
    que `scripts/factor_ablation_study.py` ya guardó (docs/quant_methodology.md
    §12), nunca recalcula nada. `directionally_consistent = False` en un
    factor significa que el signo medido contradice el peso que tiene hoy -
    se muestra la contradicción, nunca se corrige el peso en silencio (ver
    docs/quant_methodology.md §1 y §6.1)."""
    results = ars.load_ablation_report(horizon_days)
    return FactorAblationReportResponse(
        horizon_days=horizon_days,
        available_horizons_days=list(ars.AVAILABLE_HORIZONS_DAYS),
        results=[asdict(r) for r in results],
    )
