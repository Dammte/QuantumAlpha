"""Reconstruction (2026-09), Fase 2: end-to-end orchestration test for
scripts/daily_close.py's `run_daily_close` - the pure per-row logic
(build_ticker_daily_state/ticker_trigger_events/position_daily_state_from_risk/
position_trigger_events/build_daily_brief) already has its own hand-built
unit tests (tests/unit/test_daily_close.py); this verifies the plumbing that
wires them together against a real (SQLite, in-memory) session and the same
deterministic FakeMarketDataProvider every other integration test uses -
never real yfinance/network."""

from datetime import date, datetime, time, timedelta

from sqlalchemy.orm import Session

import scripts.daily_close as dc
from app.domain.models.transaction import TransactionType
from app.infrastructure.db.models import PortfolioORM
from app.infrastructure.db.repositories.daily_brief_repository import DailyBriefRepository
from app.infrastructure.db.repositories.job_run_repository import JobRunRepository
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.db.repositories.position_daily_state_repository import PositionDailyStateRepository
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from tests.integration.conftest import FakeMarketDataProvider


def test_run_daily_close_populates_ticker_states_for_the_us_universe(db_session: Session):
    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    result = dc.run_daily_close(db_session, market_data, screener, regions=("us",))

    assert result.job_run.status == "success"
    assert result.rows_processed > 0

    ticker_repo = TickerDailyStateRepository(db_session)
    # Not pinned to one hardcoded ticker: FakeMarketDataProvider's random walk
    # is seeded per-ticker (deterministic, but not tuned to clear the
    # screener's own liquidity/price floor for every single symbol) - what
    # matters here is that *some* of the universe made it through the whole
    # pipeline into a real, well-formed row, not which specific one did.
    rows = ticker_repo.latest_by_region("us")
    assert len(rows) > 0
    sample = rows[0]
    assert sample.region == "us"
    assert sample.gate_version is not None
    assert isinstance(sample.gate_passes, bool)


def test_run_daily_close_records_a_successful_job_run(db_session: Session):
    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    dc.run_daily_close(db_session, market_data, screener, regions=("us",))

    job_repo = JobRunRepository(db_session)
    latest = job_repo.latest("daily_close")
    assert latest is not None
    assert latest.status == "success"
    assert latest.finished_at is not None


def test_run_daily_close_builds_a_daily_brief_for_every_portfolio(db_session: Session):
    portfolio = PortfolioORM(name="Cartera de prueba", base_currency="USD")
    db_session.add(portfolio)
    db_session.commit()
    db_session.refresh(portfolio)

    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    dc.run_daily_close(db_session, market_data, screener, regions=("us",))

    brief_repo = DailyBriefRepository(db_session)
    brief = brief_repo.latest_for_portfolio(portfolio.id)
    assert brief is not None
    assert brief.brief_date == date.today()


def test_run_daily_close_writes_a_position_daily_state_for_a_held_ticker_with_a_trade_plan(db_session: Session):
    portfolio_repo = PortfolioRepository(db_session)
    portfolio = portfolio_repo.create("Cartera con posición")
    entry_date = date.today() - timedelta(days=30)
    portfolio_repo.add_transaction(
        portfolio.id,
        TransactionType.BUY,
        quantity=10.0,
        ticker="AAPL",
        price=100.0,
        executed_at=datetime.combine(entry_date, time()),
    )

    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    dc.run_daily_close(db_session, market_data, screener, regions=("us",))

    position_repo = PositionDailyStateRepository(db_session)
    states = position_repo.latest_for_portfolio(portfolio.id)
    assert len(states) == 1
    assert states[0].ticker == "AAPL"
    assert states[0].urgency in {"hold", "watch", "tighten_stop", "reduce", "exit_now"}


def test_run_daily_close_handles_a_portfolio_with_no_transactions(db_session: Session):
    portfolio_repo = PortfolioRepository(db_session)
    empty_portfolio = portfolio_repo.create("Cartera vacía")

    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    result = dc.run_daily_close(db_session, market_data, screener, regions=("us",))
    assert result.job_run.status == "success"

    brief_repo = DailyBriefRepository(db_session)
    brief = brief_repo.latest_for_portfolio(empty_portfolio.id)
    assert brief is not None
    assert brief.positions_needing_action == 0


# --- Auditoria del Radar, bloque B2/B3: aislamiento por ticker --------------


def test_run_daily_close_isolates_a_ticker_that_raises_and_still_finishes_successfully(
    db_session: Session, monkeypatch
):
    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    real_build = dc.build_ticker_daily_state
    call_count = {"n": 0}

    def _flaky_build(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("dato corrupto simulado")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(dc, "build_ticker_daily_state", _flaky_build)

    result = dc.run_daily_close(db_session, market_data, screener, regions=("us",))

    # El job termina en éxito, no se propaga la excepción del primer ticker.
    assert result.job_run.status == "success"
    assert result.job_run.finished_at is not None
    assert call_count["n"] > 1  # el resto del universo se siguió procesando
    assert result.detail["tickers_failed"] == {"ValueError": 1}
    # Y el resto de tickers sí quedaron persistidos - no solo "no crasheó".
    ticker_repo = TickerDailyStateRepository(db_session)
    assert len(ticker_repo.latest_by_region("us")) > 0


def test_run_daily_close_leaves_job_runs_not_stuck_running_after_an_unhandled_exception(
    db_session: Session, monkeypatch
):
    # Una excepción FUERA de los bucles aislados (aquí, simulada al listar el
    # universo) debe dejar la fila de job_runs en "failed", nunca en
    # "running" para siempre - el bug exacto de B2/B3 (sin el db.rollback()
    # antes de job_repo.finish, ese propio finish podía lanzar
    # PendingRollbackError y dejar la fila colgada).
    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    def _boom(*args, **kwargs):
        raise RuntimeError("fallo simulado listando el universo")

    monkeypatch.setattr(screener, "get_universe_snapshot", _boom)

    try:
        dc.run_daily_close(db_session, market_data, screener, regions=("us",))
    except RuntimeError:
        pass

    job_repo = JobRunRepository(db_session)
    latest = job_repo.latest("daily_close")
    assert latest is not None
    assert latest.status == "failed"
    assert latest.finished_at is not None


# --- Auditoria del Radar, bloque B4: idempotencia de daily_briefs -----------


def test_run_daily_close_twice_the_same_day_does_not_zero_out_the_real_brief(db_session: Session):
    # "Un reintento pone los contadores a cero y machaca el brief real" - el
    # bug reportado. new_entry_triggers/new_gate_passes ahora se derivan del
    # log de TriggerEvent (idempotente), no de un acumulador local que se lee
    # en cero en un reintento del mismo día.
    portfolio_repo = PortfolioRepository(db_session)
    portfolio = portfolio_repo.create("Cartera de prueba idempotencia")

    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)

    first = dc.run_daily_close(db_session, market_data, screener, regions=("us",))
    second = dc.run_daily_close(db_session, market_data, screener, regions=("us",))

    assert first.job_run.status == "success"
    assert second.job_run.status == "success"
    # El segundo intento del MISMO día ve el mismo conteo de disparos que el
    # primero - nunca menor (nunca "vuelve a cero").
    assert second.new_gate_passes == first.new_gate_passes
    assert second.new_entry_triggers == first.new_entry_triggers

    brief_repo = DailyBriefRepository(db_session)
    brief = brief_repo.latest_for_portfolio(portfolio.id)
    assert brief is not None
    assert brief.new_gate_passes == first.new_gate_passes
