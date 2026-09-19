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
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime

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
from app.infrastructure.db.repositories.setup_performance_repository import SetupPerformanceRepository
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.infrastructure.db.repositories.trade_plan_repository import TradePlanRepository
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.services import exit_engine as ee
from app.services import levels_engine as le
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import benchmark_for_region
from app.services.portfolio_risk_service import PositionRisk, get_portfolio_positions_risk
from app.services.setups.types import SetupStage
from app.services.ticker_daily_state_builder import build_ticker_daily_state

logger = logging.getLogger(__name__)

REGIONS = ("us", "europe")


def ticker_trigger_events(
    previous: TickerDailyState | None, new: TickerDailyState, now: datetime
) -> list[TriggerEvent]:
    """A `TriggerEvent` per meaningful change between yesterday's precomputed
    state and today's - `[]` when there's nothing to compare against yet
    (the very first run ever tracking this ticker) or when `previous` is
    itself from today (a same-day retry, not a real day-over-day
    transition).

    `setup_ready`/`setup_triggered` (biblioteca de setups, Parte 11.4 -
    "registra los que alcanzaron READY, si dispararon, si el propietario
    entró, y cómo acabaron") son un evento por (ticker, nombre de setup) que
    llega a `SetupStage.READY`/`TRIGGERED` desde cualquier otro estado (o
    desde no existir ayer) - comparados por nombre, no por posición en la
    lista, porque el orden de `setups` puede cambiar de un día a otro
    (`arbitration.order_by_rank`). `setup_ready` es el análogo de
    `gate_passed` (un estado más temprano, no algo que el propietario "actúa"
    sobre directamente - por eso no es elegible para `taken` en
    `trigger_performance_service.py`); `setup_triggered` es el análogo de
    `entry_triggered` (el momento accionable, sí elegible para `taken`).
    Reutiliza el mismo `TriggerEvent`/`compute_trigger_outcomes` que ya mide
    `gate_passed`/`entry_triggered` - ningún esquema ni agregación nuevos,
    ver `trigger_performance_service.py`."""
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
    previous_setup_stage = {s["name"]: s.get("stage") for s in (previous.setups or [])}
    for setup in new.setups or []:
        stage = setup.get("stage")
        prior_stage = previous_setup_stage.get(setup["name"])
        if stage not in (SetupStage.READY.value, SetupStage.TRIGGERED.value):
            continue
        if prior_stage == stage:
            continue  # ya estaba en ese mismo estado ayer - no es una transición nueva
        events.append(
            TriggerEvent(
                id=None,
                entity_type="ticker",
                entity_key=new.ticker,
                event_type="setup_ready" if stage == SetupStage.READY.value else "setup_triggered",
                previous_value=prior_stage,
                new_value=stage,
                occurred_at=now,
                details={"price": new.price, "setup_name": setup["name"], "family": setup.get("family")},
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
    detail: dict


def _today_trigger_counts(trigger_repo: TriggerEventRepository, trade_date: date) -> tuple[int, int]:
    """Auditoria del Radar, bloque B4: `new_gate_passes`/`new_entry_triggers`
    ya NO se acumulan localmente mientras se recorre el universo - un
    reintento del mismo día vería `ticker_trigger_events` devolver `[]` para
    todos (la propia guarda de "previous.trade_date == new.trade_date"), así
    que el acumulador local se leería en cero aunque los disparos reales ya
    hubieran ocurrido en un intento anterior ese mismo día, machacando
    `daily_briefs` con un resumen falso. En vez de eso, se cuenta desde el
    propio log de `TriggerEvent` (append-only, nunca duplica - ver el
    docstring de `TriggerEventRepositoryPort.record`) filtrado a hoy: el
    mismo número sale independientemente de cuántas veces se haya corrido
    `daily_close.py` hoy mismo, que es justo lo que "idempotente" significa
    aquí."""
    today_start = datetime.combine(trade_date, datetime.min.time(), tzinfo=UTC)
    today_events = trigger_repo.list_since(today_start, entity_type="ticker")
    new_gate_passes = sum(1 for e in today_events if e.event_type == "gate_passed")
    new_entry_triggers = sum(1 for e in today_events if e.event_type == "entry_triggered")
    return new_gate_passes, new_entry_triggers


def run_daily_close(
    db: Session,
    market_data: MarketDataService,
    screener: MarketScreenerService,
    regions: tuple[str, ...] = REGIONS,
) -> DailyCloseResult:
    """Auditoria del Radar (bloque C), tras el diagnóstico del bloque A:
    aisla cada ticker y cada cartera (B2/B3 - un fallo puntual nunca debe
    tumbar el resto del universo ni dejar `job_runs` colgado en "running"),
    fuerza el precio de verdad de cierre en vez de la caché durable de 3h
    que puede llevar precios intradía (B1), y deja un resumen ejecutable
    persistido en `job_runs.detail` (bloque C.1) - no solo en logs, que
    Render no retiene indefinidamente."""
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
    started_monotonic = time.monotonic()
    rows_processed = 0
    tickers_processed = 0
    tickers_failed_by_type: dict[str, int] = {}
    gate_passes_today = 0
    setups_by_family: dict[str, int] = {}
    portfolios_processed = 0
    portfolios_failed_by_type: dict[str, int] = {}

    # Parte 10.3 (§28.x): la foto completa de `setup_performance` (si
    # `scripts/setup_replay_study.py` ya corrió alguna vez), leída una sola
    # vez para todo el job - no por ticker. Solo la fila SIN segmentar de
    # cada nombre (ver `setup_replay.apply_measured_confidence`); `{}` si
    # la tabla todavía está vacía, dejando todo en UNVALIDATED como hasta
    # ahora, nunca un error.
    setup_performance_by_name = {
        row.setup_name: row
        for row in SetupPerformanceRepository(db).all()
        if row.grade is None and row.market_regime is None
    }

    try:
        universe_snapshot: list[TickerSnapshot] = []
        for region in regions:
            # B1: `force_refresh=True` - este job es la propia definición de
            # "el cierre". Sin esto, la caché durable de 3h (compartida con
            # el servicio web vía `durable_cache.py`) puede colarle precios
            # intradía de hace un par de horas como si fueran el cierre real,
            # si algún usuario ya disparó un recálculo esta tarde.
            snapshot = screener.get_universe_snapshot(region, force_refresh=True, db=db)
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
                # B2/B3: un ticker con un borde no cubierto (dato corrupto,
                # una excepción numérica, un fallo de red puntual en
                # `get_next_earnings_date`) no debe tumbar el resto del
                # universo - misma disciplina que
                # `portfolio_risk_service._safe_assess_position_risk` ya
                # aplica un nivel más arriba. `db.rollback()` ANTES de seguir
                # con el siguiente ticker: si el fallo dejó la sesión en un
                # estado de transacción abortada (típico de un error de BD a
                # mitad de `upsert`), cualquier lectura/escritura posterior
                # en la misma sesión lanzaría `PendingRollbackError` en
                # cascada sin este rollback - el mecanismo exacto por el que
                # un solo ticker malo también podía inutilizar el `finally`
                # de más abajo.
                try:
                    # Parte 6.2's `no_event_risk` - one network call per
                    # ticker, accepted here (unlike in any live request path)
                    # because this job runs once a night, off the request
                    # path entirely (Parte 4.1) - the exact distinction
                    # CLAUDE.md's "no llamadas de red por ticker en los
                    # caminos calientes" rule draws between a job and an
                    # endpoint.
                    next_earnings_date = market_data.get_next_earnings_date(ts.ticker)
                    state = build_ticker_daily_state(
                        ts, region, df, trade_date, now, next_earnings_date, benchmark_close,
                        setup_performance_by_name,
                    )
                    if state is None:
                        continue
                    previous = ticker_repo.latest_for_ticker(ts.ticker)
                    ticker_repo.upsert(state)
                    rows_processed += 1
                    tickers_processed += 1
                    if state.gate_passes:
                        gate_passes_today += 1
                    for setup in state.setups or []:
                        family = setup.get("family", "?")
                        setups_by_family[family] = setups_by_family.get(family, 0) + 1
                    for event in ticker_trigger_events(previous, state, now):
                        trigger_repo.record(event)
                except Exception as ticker_exc:
                    db.rollback()
                    error_type = type(ticker_exc).__name__
                    tickers_failed_by_type[error_type] = tickers_failed_by_type.get(error_type, 0) + 1
                    logger.exception("daily_close: fallo procesando %s (%s)", ts.ticker, region)

        # B4 (idempotencia): el conteo del día se deriva del log de eventos,
        # no de un acumulador local - ver `_today_trigger_counts`.
        new_gate_passes, new_entry_triggers = _today_trigger_counts(trigger_repo, trade_date)

        for portfolio in portfolio_repo.list_all():
            # Mismo aislamiento que B2/B3 para el universo, aplicado aquí a
            # cartera completa: una cartera con datos inconsistentes (un
            # `TradePlan` huérfano, una transacción corrupta) no debe
            # impedir que el resto de carteras reciban su brief de hoy.
            try:
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
                portfolios_processed += 1
            except Exception as portfolio_exc:
                db.rollback()
                error_type = type(portfolio_exc).__name__
                portfolios_failed_by_type[error_type] = portfolios_failed_by_type.get(error_type, 0) + 1
                logger.exception("daily_close: fallo procesando cartera %s", portfolio.id)

        duration_seconds = round(time.monotonic() - started_monotonic, 1)
        detail = {
            "regions": list(regions),
            "tickers_processed": tickers_processed,
            "tickers_failed": tickers_failed_by_type,
            "gate_passes_today": gate_passes_today,
            "new_gate_passes_today": new_gate_passes,
            "new_entry_triggers_today": new_entry_triggers,
            "setups_by_family": setups_by_family,
            "portfolios_processed": portfolios_processed,
            "portfolios_failed": portfolios_failed_by_type,
            "duration_seconds": duration_seconds,
        }
        finished = job_repo.finish(
            job.id, status="success", rows_processed=rows_processed, error_message=None, detail=detail
        )
        return DailyCloseResult(finished, rows_processed, new_gate_passes, new_entry_triggers, detail)
    except Exception as exc:
        logger.exception("daily_close failed")
        # B2/B3: rollback ANTES de `finish` - si la excepción no capturada
        # (una fuera de los dos bucles aislados de arriba, p. ej. al listar
        # el universo en sí) dejó la sesión en transacción abortada, `finish`
        # (que hace `db.get` + `db.commit`) lanzaría `PendingRollbackError`
        # en vez de persistir "failed", dejando la fila en "running" para
        # siempre - el bug exacto que este rollback cierra.
        db.rollback()
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
        d = result.detail
        print(
            f"[daily_close] {result.job_run.status} - {result.rows_processed} filas procesadas "
            f"en {d['duration_seconds']}s"
        )
        print(
            f"  tickers: {d['tickers_processed']} procesados, "
            f"{sum(d['tickers_failed'].values())} fallidos {d['tickers_failed'] or '{}'}"
        )
        print(f"  gate: {d['gate_passes_today']} pasan hoy, {result.new_gate_passes} nuevos")
        print(f"  entradas disparadas hoy: {result.new_entry_triggers}")
        print(f"  setups por familia: {d['setups_by_family'] or '{}'}")
        print(
            f"  carteras: {d['portfolios_processed']} procesadas, "
            f"{sum(d['portfolios_failed'].values())} fallidas {d['portfolios_failed'] or '{}'}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
