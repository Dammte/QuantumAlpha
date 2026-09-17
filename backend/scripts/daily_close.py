"""Reconstruction (2026-09), Fase 2, Job A: nightly precompute for every
ticker in the curated universe (the levels/triggers gate) and every open
position across every portfolio (the exit-engine read) - the whole reason
the tables Fase 2 introduced exist (see their own docstrings in
`app/domain/models/`). Run once per trading day after market close (a Render
Cron Job - see `render.yaml`), never from a live request path: the same "no
llamadas de red por ticker en los caminos calientes" rule CLAUDE.md already
applies elsewhere applies here too, just paid once a day instead of never
allowed at all.

Absorbs `MarketScreenerService.get_universe_snapshot()`'s own per-region live
computation for this purpose - the owner's explicit decision made during the
Fase 2 design (see this session's own record): this job is now the one place
per day that walks the full universe and evaluates
`levels_engine.evaluate_gate` for every ticker; every endpoint that used to
call the screener for its own live scoring instead reads `TickerDailyState` -
a plain DB read, no per-ticker computation, at any time of day (Fase 5).

Reuses the *exact same* position-side pipeline the live "/{portfolio_id}/risk"
endpoint already runs (`portfolio_risk_service.get_portfolio_positions_risk`)
rather than re-deriving the exit-engine wiring here - the same "never a
cheaper approximation of what searching it directly would show" principle
that pipeline's own docstring already states.

Usage (from `backend/`, same convention as the other testable scripts -
`factor_ablation_study.py`/`chandelier_calibration_study.py` - relying on
`pythonpath = ["."]` instead of a manual `sys.path` hack, so this module
imports cleanly from tests too):
    python -m scripts.daily_close                    # both regions, all portfolios
    python -m scripts.daily_close --region us
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.domain.models.daily_brief import DailyBrief
from app.domain.models.job_run import JobRun
from app.domain.models.position_daily_state import PositionDailyState
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.domain.models.trigger_event import TriggerEvent
from app.infrastructure.db.repositories.daily_brief_repository import DailyBriefRepository
from app.infrastructure.db.repositories.job_run_repository import JobRunRepository
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.db.repositories.position_daily_state_repository import PositionDailyStateRepository
from app.infrastructure.db.repositories.position_signal_snapshot_repository import (
    PositionSignalSnapshotRepository,
)
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.infrastructure.db.repositories.trade_plan_repository import TradePlanRepository
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.services import dynamic_universe_service as dus
from app.services import exit_engine as ee
from app.services import levels_engine as le
from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import benchmark_for_region
from app.services.portfolio_risk_service import PositionRisk, get_portfolio_positions_risk
from app.services.setups import arbitration as setups_arbitration
from app.services.setups import context_modifiers as setups_context_modifiers
from app.services.setups import registry as setups_registry
from app.services.setups.context import SetupContext
from app.services.setups.types import setup_match_to_dict
from app.services.ticker_analysis_service import MIN_BARS_REQUIRED
from app.services.trade_geometry import geometry_to_dict

logger = logging.getLogger(__name__)

REGIONS = ("us", "europe")


def _nearest_level(levels: list[ta.PriceLevel], kind: str) -> ta.PriceLevel | None:
    candidates = [lv for lv in levels if lv.kind == kind]
    return min(candidates, key=lambda lv: abs(lv.distance_pct)) if candidates else None


def build_ticker_daily_state(
    snapshot: TickerSnapshot,
    region: str,
    df: pd.DataFrame,
    trade_date: date,
    computed_at: datetime,
    next_earnings_date: date | None = None,
    benchmark_close: pd.Series | None = None,
) -> TickerDailyState | None:
    """Pure function (`next_earnings_date` is the one exception - a value
    the caller already paid the network cost for, never fetched here):
    everything `levels_engine.evaluate_gate` needs beyond what `TickerSnapshot`
    already carries (raw ATR, support/resistance, the fast-pair veto, the
    weekly Weinstein Stage, the liquidity floor) is derived here from the
    same OHLCV frame the screener already downloaded - no new network call,
    no re-fetch. `None` when there isn't enough history to say anything (same
    bar `ticker_analysis_service.compute_core_signals` uses).

    Sexta auditoría (Parte 6.2, texto literal completo): the gate's 5
    eligibility criteria replace the previous 6-condition approximation -
    see `levels_engine.py`'s own module docstring for the full reasoning."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    if len(close) < MIN_BARS_REQUIRED:
        return None

    atr_series = ta.atr(high, low, close)
    atr14 = float(atr_series.iloc[-1]) if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else None
    levels = ta.support_resistance_levels(high, low, close)
    nearest_support = _nearest_level(levels, "support")
    nearest_resistance = _nearest_level(levels, "resistance")
    fast_pair_veto = ta.detect_fast_pair_bearish_veto(close)

    # Parte 5.5: la MA30 semanal real (no la proxy diaria de `snapshot.stage`,
    # que sigue siendo el Stage *diario* mostrado en el Radar/screener como
    # contexto, sin cambios) - `weekly_not_stage4` del gate exige el
    # semanal genuino.
    multi_timeframe = mtf.analyze_multi_timeframe(df)
    weekly_stage = multi_timeframe.weekly.stage if multi_timeframe.weekly is not None else None

    # Divisa nativa, sin conversión a USD - misma simplificación documentada
    # en `ticker_analysis_service.compute_core_signals` (el universo dinámico
    # mensual, Job C, ya filtra por liquidez en USD real al entrar; esta es
    # la comprobación diaria adicional del propio gate, Parte 6.2).
    dollar_volume_20d = float((close.iloc[-20:] * volume.iloc[-20:]).mean()) if len(close) >= 20 else None
    liquidity_ok = dus.passes_liquidity_floor(snapshot.price, dollar_volume_20d)

    # Parte 7 (later pass): the same real EMA21/55 read
    # `ticker_analysis_service.compute_core_signals` already passes to
    # `evaluate_gate` for "Analizar activo"/`/risk` - cheap off the same
    # `close` series already in memory here, no new network call. Without
    # these two, `gate.entry_geometry` comes back `None` (see that
    # function's own docstring) and this job would keep persisting nothing
    # for Parte 7's real design, same gap CLAUDE.md used to flag.
    ema21_raw = ta.ema(close, mtf.FAST_MA_PERIOD).iloc[-1] if len(close) else None
    ema55_raw = ta.ema(close, mtf.SLOW_MA_PERIOD).iloc[-1] if len(close) else None
    ema21 = None if ema21_raw is None or pd.isna(ema21_raw) else float(ema21_raw)
    ema55 = None if ema55_raw is None or pd.isna(ema55_raw) else float(ema55_raw)

    # Biblioteca de setups del Radar (en curso, ver quant_methodology.md §28):
    # el mismo remuestreo semanal en memoria que `ticker_analysis_service.compute_core_signals`
    # ya paga por separado (§27.11) para darle a `detect_levels` un
    # `weekly_close` real - un segundo `resample_ohlcv` aquí, sobre el mismo
    # `df` ya en memoria, nunca una llamada de red nueva.
    weekly_df = ta.resample_ohlcv(df, mtf.WEEKLY_RULE)
    weekly_close = weekly_df["close"] if len(weekly_df) >= 2 else None
    weekly_high = weekly_df["high"] if len(weekly_df) >= 2 else None
    weekly_low = weekly_df["low"] if len(weekly_df) >= 2 else None
    weekly_volume = weekly_df["volume"] if len(weekly_df) >= 2 else None
    # `detect_levels` (Parte 5.1/§27.5) es aditivo junto a `support_resistance_levels`
    # de arriba, no un reemplazo - el gate sigue leyendo `nearest_support`/
    # `nearest_resistance` (`PriceLevel`, el tipo simple); esta lista rica
    # (`Level`, con estado FAR/APPROACHING/TESTING/BREAKING/...) es solo para
    # que los detectores de setups la lean sin recalcular nada.
    setup_levels = ta.detect_levels(high, low, close, volume, weekly_close=weekly_close)
    # Parte 2.2 (`stage1_rs_turning`): RS de Mansfield a 20 sesiones, sobre
    # el benchmark real de la región - `None` sin uno disponible (el
    # subestado simplemente no se evalúa, nunca se fabrica).
    mansfield_rs_series = (
        ta.mansfield_rs(close, benchmark_close, window=20) if benchmark_close is not None else None
    )

    setup_ctx = SetupContext(
        ticker=snapshot.ticker,
        region=region,
        trade_date=trade_date,
        close=close,
        high=high,
        low=low,
        volume=volume,
        open_=df["open"],
        weekly_close=weekly_close,
        weekly_high=weekly_high,
        weekly_low=weekly_low,
        weekly_volume=weekly_volume,
        atr_series=atr_series,
        atr14=atr14,
        ema21=ema21,
        ema55=ema55,
        sma20=snapshot.sma20,
        sma50=snapshot.sma50,
        sma150=snapshot.sma150,
        sma200=snapshot.sma200,
        rsi14=snapshot.rsi14,
        levels=setup_levels,
        multi_timeframe=multi_timeframe,
        trend=snapshot.trend,
        weekly_stage=weekly_stage,
        relative_volume=snapshot.relative_volume,
        rs_percentile=snapshot.rs_rating,
        sector_rs_percentile=snapshot.sector_rs_percentile,
        mansfield_rs_series=mansfield_rs_series,
    )
    # `stage_transition.py` (Fase 3) es, por ahora, el único detector
    # registrado en `SETUP_DETECTORS` - esta llamada devolvía siempre `[]`
    # hasta entonces (§28.2). Cablearla desde antes de tener ningún detector
    # real dejó que los tests de este job cubrieran la construcción de
    # `SetupContext` contra datos de verdad, no solo el stub sintético de
    # `test_setups_registry.py`.
    # `[]`, no `None`: "sin coincidencias hoy" es un resultado real y
    # esperado (la mayoría de tickers la mayoría de días no cumplen ningún
    # setup), mismo criterio que `gate_conditions` - `None` queda reservado
    # para una fila anterior a esta columna, no para "no se encontró nada".
    # `order_by_rank` (Parte 0/6, §28.8) deja el setup titular en el índice
    # 0 - "el mejor gana, los demás se muestran como contexto" - sin
    # descartar ninguno.
    ordered_setups = setups_arbitration.order_by_rank(setups_registry.detect_all(setup_ctx))
    setups_list = [setup_match_to_dict(m) for m in ordered_setups]

    gate = le.evaluate_gate(
        price=snapshot.price,
        trend=snapshot.trend,
        atr14=atr14,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        weekly_stage=weekly_stage,
        liquidity_ok=liquidity_ok,
        fast_pair_bearish_signal=fast_pair_veto,
        next_earnings_date=next_earnings_date,
        as_of=trade_date,
        ema21=ema21,
        ema55=ema55,
    )
    trigger = gate.entry_trigger
    stop_target = gate.stop_and_target

    # Parte 5.3 (later pass): the exact same "is there even a viable
    # geometry to grade" precondition `ticker_analysis_service.compute_core_signals`
    # uses for "Analizar activo" - see levels_engine.compute_grade's own
    # docstring for why `grade=None` (never a fabricated grade on top of a
    # non-viable trade). `snapshot.relative_volume`/`sma200`/`rs_rating`/
    # `sector_rs_percentile` are already computed by the screener for this
    # exact ticker - no re-derivation, no new network call.
    grade_dict: dict | None = None
    if gate.entry_trigger is not None and gate.entry_geometry is not None and gate.entry_geometry.viable:
        weekly_bullish = mtf.timeframe_bias(multi_timeframe.weekly) == "bullish"
        grade_result = le.compute_grade(
            price=snapshot.price,
            atr14=atr14,
            entry_trigger=gate.entry_trigger,
            geometry=gate.entry_geometry,
            weekly_bullish=weekly_bullish,
            relative_volume=snapshot.relative_volume,
            rs_percentile=snapshot.rs_rating,
            sma200=snapshot.sma200,
            sector_rs_percentile=snapshot.sector_rs_percentile,
        )
        # Parte 6 (§28.x): los modificadores de contexto de la biblioteca de
        # setups ajustan este MISMO grado - no un concepto paralelo - con la
        # lista de setups que este ticker ya calculó arriba
        # (`ordered_setups`). No necesitan cartera, así que corren aquí, no
        # en el endpoint de lectura (a diferencia de
        # `apply_portfolio_grade_modifiers`).
        grade_result = setups_context_modifiers.apply_context_modifiers(grade_result, setup_ctx, ordered_setups)
        grade_dict = {
            "grade": grade_result.grade.value if grade_result.grade is not None else None,
            "reasons": grade_result.reasons,
        }

    return TickerDailyState(
        id=None,
        region=region,
        ticker=snapshot.ticker,
        trade_date=trade_date,
        computed_at=computed_at,
        price=snapshot.price,
        currency=snapshot.currency,
        trend=snapshot.trend.value,
        stage=snapshot.stage.value if snapshot.stage else None,
        rs_rating=snapshot.rs_rating,
        adx14=snapshot.adx14,
        atr_multiple=snapshot.atr_multiple,
        rsi14=snapshot.rsi14,
        gate_passes=gate.passes,
        gate_conditions=[{"label": c.label, "passed": c.passed} for c in gate.conditions],
        gate_version=le.GATE_VERSION,
        entry_trigger_type=trigger.trigger_type if trigger else None,
        entry_trigger_price=trigger.trigger_price if trigger else None,
        entry_already_triggered=trigger.already_triggered if trigger else False,
        stop_loss=stop_target.stop_loss,
        take_profit=stop_target.take_profit,
        take_profit_method=stop_target.take_profit_method,
        risk_reward=stop_target.risk_reward,
        entry_geometry=geometry_to_dict(gate.entry_geometry) if gate.entry_geometry is not None else None,
        grade=grade_dict,
        setups=setups_list,
    )


def ticker_trigger_events(
    previous: TickerDailyState | None, new: TickerDailyState, now: datetime
) -> list[TriggerEvent]:
    """A `TriggerEvent` per meaningful change between yesterday's precomputed
    state and today's - `[]` when there's nothing to compare against yet
    (the very first run ever tracking this ticker) or when `previous` is
    itself from today (a same-day retry, not a real day-over-day
    transition)."""
    if previous is None or previous.trade_date == new.trade_date:
        return []
    events: list[TriggerEvent] = []
    if previous.gate_passes != new.gate_passes:
        events.append(
            TriggerEvent(
                id=None,
                entity_type="ticker",
                entity_key=new.ticker,
                event_type="gate_passed" if new.gate_passes else "gate_failed",
                previous_value=str(previous.gate_passes),
                new_value=str(new.gate_passes),
                occurred_at=now,
                details={"price": new.price},
            )
        )
    if not previous.entry_already_triggered and new.entry_already_triggered:
        events.append(
            TriggerEvent(
                id=None,
                entity_type="ticker",
                entity_key=new.ticker,
                event_type="entry_triggered",
                previous_value=previous.entry_trigger_type,
                new_value=new.entry_trigger_type,
                occurred_at=now,
                details={"price": new.price, "trigger_price": new.entry_trigger_price},
            )
        )
    return events


def position_daily_state_from_risk(
    risk: PositionRisk, portfolio_id: int, trade_date: date, computed_at: datetime
) -> PositionDailyState | None:
    """`None` when `risk` carries no exit-engine read at all - happens when
    the ticker has no open `TradePlan` yet (e.g. a transaction was recorded
    but `trade_plan_service.ensure_trade_plan` couldn't reconstruct one).
    Nothing to persist for "what to do with this holding today" without a
    plan to judge it against."""
    if risk.exit_urgency is None:
        return None
    return PositionDailyState(
        id=None,
        portfolio_id=portfolio_id,
        ticker=risk.ticker,
        trade_date=trade_date,
        computed_at=computed_at,
        urgency=risk.exit_urgency,
        reasons=risk.exit_reasons,
        price=risk.price,
        r_multiple=risk.r_multiple,
        current_stop=risk.trade_plan.current_stop if risk.trade_plan else None,
        engine_version=le.GATE_VERSION,
    )


def position_trigger_events(
    previous: PositionDailyState | None, new: PositionDailyState, now: datetime
) -> list[TriggerEvent]:
    if previous is None or previous.trade_date == new.trade_date or previous.urgency == new.urgency:
        return []
    return [
        TriggerEvent(
            id=None,
            entity_type="position",
            entity_key=f"{new.portfolio_id}:{new.ticker}",
            event_type="exit_urgency_changed",
            previous_value=previous.urgency,
            new_value=new.urgency,
            occurred_at=now,
            details={"price": new.price, "r_multiple": new.r_multiple},
        )
    ]


def build_daily_brief(
    portfolio_id: int,
    trade_date: date,
    computed_at: datetime,
    position_states: list[PositionDailyState],
    new_entry_triggers: int,
    new_gate_passes: int,
) -> DailyBrief:
    needing_action = [s for s in position_states if s.urgency != ee.ExitUrgency.HOLD.value]
    if needing_action:
        headline = f"{len(needing_action)} posición(es) necesitan atención hoy."
    elif new_entry_triggers:
        headline = f"{new_entry_triggers} activo(s) del universo dispararon su entrada hoy."
    else:
        headline = "Sin acciones urgentes hoy."
    return DailyBrief(
        id=None,
        portfolio_id=portfolio_id,
        brief_date=trade_date,
        computed_at=computed_at,
        positions_needing_action=len(needing_action),
        new_entry_triggers=new_entry_triggers,
        new_gate_passes=new_gate_passes,
        headline=headline,
    )


@dataclass(frozen=True, slots=True)
class DailyCloseResult:
    job_run: JobRun
    rows_processed: int
    new_gate_passes: int
    new_entry_triggers: int


def run_daily_close(
    db: Session,
    market_data: MarketDataService,
    screener: MarketScreenerService,
    regions: tuple[str, ...] = REGIONS,
) -> DailyCloseResult:
    job_repo = JobRunRepository(db)
    ticker_repo = TickerDailyStateRepository(db)
    position_repo = PositionDailyStateRepository(db)
    trigger_repo = TriggerEventRepository(db)
    brief_repo = DailyBriefRepository(db)
    portfolio_repo = PortfolioRepository(db)
    trade_plan_repo = TradePlanRepository(db)
    position_signal_snapshot_repo = PositionSignalSnapshotRepository(db)

    job = job_repo.start("daily_close")
    trade_date = date.today()
    now = datetime.now(UTC)
    rows_processed = 0
    new_gate_passes = 0
    new_entry_triggers = 0

    try:
        universe_snapshot: list[TickerSnapshot] = []
        for region in regions:
            snapshot = screener.get_universe_snapshot(region, db=db)
            universe_snapshot.extend(snapshot)
            ohlcv_by_ticker = screener.get_cached_ohlcv(region)
            # Biblioteca de setups del Radar (`stage1_rs_turning`, §28): el
            # mismo benchmark de la región que `get_universe_snapshot` ya
            # descargó en el lote batcheado (`market_screener_service.py`) -
            # una sola búsqueda por región, no por ticker.
            benchmark_df = ohlcv_by_ticker.get(benchmark_for_region(region))
            benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
            for ts in snapshot:
                df = ohlcv_by_ticker.get(ts.ticker)
                if df is None:
                    continue
                # Parte 6.2's `no_event_risk` - one network call per ticker,
                # accepted here (unlike in any live request path) because
                # this job runs once a night, off the request path entirely
                # (Parte 4.1) - the exact distinction CLAUDE.md's "no
                # llamadas de red por ticker en los caminos calientes" rule
                # draws between a job and an endpoint.
                next_earnings_date = market_data.get_next_earnings_date(ts.ticker)
                state = build_ticker_daily_state(
                    ts, region, df, trade_date, now, next_earnings_date, benchmark_close
                )
                if state is None:
                    continue
                previous = ticker_repo.latest_for_ticker(ts.ticker)
                ticker_repo.upsert(state)
                rows_processed += 1
                for event in ticker_trigger_events(previous, state, now):
                    trigger_repo.record(event)
                    if event.event_type == "gate_passed":
                        new_gate_passes += 1
                    elif event.event_type == "entry_triggered":
                        new_entry_triggers += 1

        for portfolio in portfolio_repo.list_all():
            transactions = portfolio_repo.get_transactions(portfolio.id)
            tickers = sorted({tx.ticker for tx in transactions if tx.ticker is not None})
            position_states: list[PositionDailyState] = []
            if tickers:
                previous_by_ticker = {s.ticker: s for s in position_repo.latest_for_portfolio(portfolio.id)}
                risks = get_portfolio_positions_risk(
                    tickers,
                    market_data,
                    universe_snapshot,
                    portfolio_id=portfolio.id,
                    transactions=transactions,
                    trade_plan_repo=trade_plan_repo,
                    position_signal_snapshot_repo=position_signal_snapshot_repo,
                )
                for risk in risks:
                    state = position_daily_state_from_risk(risk, portfolio.id, trade_date, now)
                    if state is None:
                        continue
                    previous = previous_by_ticker.get(risk.ticker)
                    position_repo.upsert(state)
                    rows_processed += 1
                    position_states.append(state)
                    for event in position_trigger_events(previous, state, now):
                        trigger_repo.record(event)

            brief = build_daily_brief(
                portfolio.id, trade_date, now, position_states, new_entry_triggers, new_gate_passes
            )
            brief_repo.upsert(brief)
            rows_processed += 1

        finished = job_repo.finish(job.id, status="success", rows_processed=rows_processed, error_message=None)
        return DailyCloseResult(finished, rows_processed, new_gate_passes, new_entry_triggers)
    except Exception as exc:
        logger.exception("daily_close failed")
        job_repo.finish(job.id, status="failed", rows_processed=rows_processed, error_message=str(exc)[:2000])
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region", choices=REGIONS, action="append", dest="regions", help="Limit to this region (repeatable)"
    )
    args = parser.parse_args()
    regions = tuple(args.regions) if args.regions else REGIONS

    market_data = MarketDataService(YFinanceProvider())
    db = SessionLocal()
    try:
        screener = MarketScreenerService(market_data)
        result = run_daily_close(db, market_data, screener, regions=regions)
        print(
            f"[daily_close] {result.job_run.status} - {result.rows_processed} filas procesadas, "
            f"{result.new_gate_passes} gates nuevos, {result.new_entry_triggers} entradas disparadas"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
