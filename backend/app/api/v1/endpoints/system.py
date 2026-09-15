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
from app.core import trading_params as tp
from app.infrastructure.db.repositories.position_signal_snapshot_repository import PositionSignalSnapshotRepository
from app.infrastructure.db.repositories.recommendation_snapshot_repository import RecommendationSnapshotRepository
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository
from app.schemas.system import SignalPerformanceResponse, TradingParamsResponse
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


@router.get("/params", response_model=TradingParamsResponse)
def get_trading_params() -> TradingParamsResponse:
    """Reconstruction (2026-09), Parte 19: read-only mirror of
    `app.core.trading_params` - no computation, no DB, so it can never be
    slow or stale. Lets the UI show which config produced a given gate/
    geometry/exit decision without hardcoding a second copy of these
    numbers anywhere in the frontend."""
    return TradingParamsResponse(
        risk_per_trade_pct=tp.RISK_PER_TRADE_PCT,
        max_position_pct=tp.MAX_POSITION_PCT,
        max_aggregate_risk_pct=tp.MAX_AGGREGATE_RISK_PCT,
        max_open_positions=tp.MAX_OPEN_POSITIONS,
        min_position_usd=tp.MIN_POSITION_USD,
        min_position_for_scaling=tp.MIN_POSITION_FOR_SCALING,
        transaction_cost_pct=tp.TRANSACTION_COST_PCT,
        stop_atr_ceiling=tp.STOP_ATR_CEILING,
        risk_ceiling_atr_multiple=tp.RISK_CEILING_ATR_MULTIPLE,
        risk_ceiling_min_pct=tp.RISK_CEILING_MIN_PCT,
        risk_ceiling_max_pct=tp.RISK_CEILING_MAX_PCT,
        min_risk_reward_net=tp.MIN_RISK_REWARD_NET,
        scale_out_1r_fraction=tp.SCALE_OUT_1R_FRACTION,
        scale_out_2r_fraction=tp.SCALE_OUT_2R_FRACTION,
        last_tranche_time_stop_bars=tp.LAST_TRANCHE_TIME_STOP_BARS,
        chandelier_window=tp.CHANDELIER_WINDOW,
        chandelier_mult_by_vol=tp.CHANDELIER_MULT_BY_VOL,
        chandelier_profit_lock_r=tp.CHANDELIER_PROFIT_LOCK_R,
        chandelier_profit_lock_mult=tp.CHANDELIER_PROFIT_LOCK_MULT,
        trigger_max_distance_atr=tp.TRIGGER_MAX_DISTANCE_ATR,
        breakout_min_rel_volume=tp.BREAKOUT_MIN_REL_VOLUME,
        stall_min_bars=tp.STALL_MIN_BARS,
        stall_max_bars=tp.STALL_MAX_BARS,
        high_correlation_threshold=tp.HIGH_CORRELATION_THRESHOLD,
    )
