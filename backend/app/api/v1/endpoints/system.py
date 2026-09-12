from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_market_data_service,
    get_position_signal_snapshot_repository,
    get_recommendation_snapshot_repository,
    get_trigger_event_repository,
)
from app.infrastructure.db.repositories.position_signal_snapshot_repository import PositionSignalSnapshotRepository
from app.infrastructure.db.repositories.recommendation_snapshot_repository import RecommendationSnapshotRepository
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository
from app.schemas.system import SignalPerformanceResponse
from app.services import signal_performance_service as sps
from app.services import trigger_performance_service as tps
from app.services.market_data_service import MarketDataService

router = APIRouter(prefix="/system", tags=["system"])

# "Since forever" for TriggerEventRepository.list_since - the table itself
# only starts accumulating rows from Fase 2 onward (daily_close.py's own
# first run), so this is just a sentinel older than any real row can be,
# not a claim that data exists back to this date. Mirrors this endpoint's
# own pre-existing "answers with whatever history has accumulated" framing
# for the verdict/signal snapshots below.
_TRIGGER_EVENTS_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


@router.get("/signal-performance", response_model=SignalPerformanceResponse)
def get_signal_performance(
    recommendation_repo: Annotated[
        RecommendationSnapshotRepository, Depends(get_recommendation_snapshot_repository)
    ],
    position_repo: Annotated[PositionSignalSnapshotRepository, Depends(get_position_signal_snapshot_repository)],
    trigger_event_repo: Annotated[TriggerEventRepository, Depends(get_trigger_event_repository)],
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
    before now.

    2026-09 (reconstruction, Fase 8): `trigger_outcomes` is the new primary
    read for this same question, measured against `TriggerEvent`
    (gate_passed/entry_triggered) instead of the retired checklist's
    verdict/signal labels - see `trigger_performance_service.py`. Kept
    alongside, not replacing, the verdict/signal outcomes above: those still
    correctly describe every snapshot recorded before the Fase 4 cutover."""
    recommendation_snapshots = recommendation_repo.list_all()
    position_snapshots = position_repo.list_all()
    report = sps.build_signal_performance_report(recommendation_snapshots, position_snapshots, market_data)
    trigger_events = trigger_event_repo.list_since(_TRIGGER_EVENTS_EPOCH, entity_type="ticker")
    trigger_report = tps.build_trigger_performance_report(trigger_events, market_data)
    return SignalPerformanceResponse(
        verdict_outcomes=[asdict(o) for o in report.verdict_outcomes],
        signal_outcomes=[asdict(o) for o in report.signal_outcomes],
        false_negatives=[asdict(f) for f in report.false_negatives],
        trigger_outcomes=[asdict(o) for o in trigger_report.outcomes],
        as_of=report.as_of,
    )
