"""Reconstruction (2026-09), Fase 2: the 6 new precompute repositories have
no caller yet (daily_close.py/intraday_refresh.py don't exist until later in
this Fase) - these are lightweight round-trip smoke tests against a real
(SQLite, in-memory) session, verifying the upsert/query logic in its own
right rather than shipping it completely unverified until a future slice
happens to exercise it end-to-end."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.domain.models.daily_brief import DailyBrief
from app.domain.models.position_daily_state import PositionDailyState
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_intraday_state import TickerIntradayState
from app.domain.models.trigger_event import TriggerEvent
from app.infrastructure.db.models import PortfolioORM
from app.infrastructure.db.repositories.daily_brief_repository import DailyBriefRepository
from app.infrastructure.db.repositories.job_run_repository import JobRunRepository
from app.infrastructure.db.repositories.position_daily_state_repository import PositionDailyStateRepository
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.infrastructure.db.repositories.ticker_intraday_state_repository import TickerIntradayStateRepository
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository


@pytest.fixture()
def portfolio_id(db_session: Session) -> int:
    portfolio = PortfolioORM(name="Test Portfolio", base_currency="USD")
    db_session.add(portfolio)
    db_session.commit()
    db_session.refresh(portfolio)
    return portfolio.id


def _daily_state(ticker: str, trade_date: date, **overrides) -> TickerDailyState:
    defaults = dict(
        id=None,
        region="us",
        ticker=ticker,
        trade_date=trade_date,
        computed_at=datetime.now(UTC),
        price=100.0,
        currency="USD",
        trend="uptrend",
        stage="stage2",
        rs_rating=80,
        adx14=28.0,
        atr_multiple=1.2,
        rsi14=55.0,
        gate_passes=True,
        gate_conditions=[{"label": "Tendencia alcista", "passed": True}],
        gate_version="2026-09-levels-v1",
        entry_trigger_type="breakout",
        entry_trigger_price=105.0,
        entry_already_triggered=False,
        stop_loss=95.0,
        take_profit=115.0,
        take_profit_method="objetivo 2:1 sobre el riesgo",
        risk_reward=2.0,
    )
    defaults.update(overrides)
    return TickerDailyState(**defaults)


# --- JobRunRepository --------------------------------------------------------


def test_job_run_start_creates_a_running_row(db_session: Session):
    repo = JobRunRepository(db_session)
    run = repo.start("daily_close")
    assert run.id is not None
    assert run.status == "running"
    assert run.finished_at is None


def test_job_run_finish_updates_the_same_row(db_session: Session):
    repo = JobRunRepository(db_session)
    run = repo.start("daily_close")
    finished = repo.finish(run.id, status="success", rows_processed=217, error_message=None)
    assert finished.id == run.id
    assert finished.status == "success"
    assert finished.rows_processed == 217
    assert finished.finished_at is not None


def test_job_run_latest_returns_the_most_recently_started_run(db_session: Session):
    repo = JobRunRepository(db_session)
    first = repo.start("daily_close")
    repo.finish(first.id, "success", 1, None)
    second = repo.start("daily_close")
    latest = repo.latest("daily_close")
    assert latest.id == second.id


def test_job_run_latest_none_when_job_never_ran(db_session: Session):
    repo = JobRunRepository(db_session)
    assert repo.latest("intraday_refresh") is None


# --- TickerDailyStateRepository ----------------------------------------------


def test_ticker_daily_state_upsert_then_get_round_trips(db_session: Session):
    repo = TickerDailyStateRepository(db_session)
    saved = repo.upsert(_daily_state("AAPL", date(2026, 9, 10)))
    fetched = repo.get("us", "AAPL", date(2026, 9, 10))
    assert fetched is not None
    assert fetched.ticker == "AAPL"
    assert fetched.gate_passes is True
    assert fetched.gate_conditions == [{"label": "Tendencia alcista", "passed": True}]
    assert saved.id == fetched.id


def test_ticker_daily_state_entry_geometry_round_trips_through_the_json_column(db_session: Session):
    # Parte 7 (later pass): `daily_close.py` persists `entry_geometry` as a
    # plain JSON dict (`trade_geometry.geometry_to_dict`) - this must survive
    # a real DB round trip byte-for-byte, `None` included for its sizing
    # fields (never computed by this job, see that function's own docstring).
    geometry = {
        "entry_price": 100.0, "stop_price": 95.0, "stop_basis": "bajo el soporte en 95.00",
        "entry_type": "pullback_support", "risk_pct": 0.05, "risk_atr": 1.25, "risk_ceiling_pct": 0.07,
        "target_price": 110.0, "target_basis": "objetivo 2:1 sobre el riesgo", "reward_pct": 0.10,
        "risk_reward_gross": 2.0, "risk_reward_net": 1.9,
        "shares_for_risk_budget": None, "position_value": None, "pct_of_portfolio": None,
        "viable": True, "rejection_reason": None,
    }
    repo = TickerDailyStateRepository(db_session)
    repo.upsert(_daily_state("AAPL", date(2026, 9, 10), entry_geometry=geometry))
    fetched = repo.get("us", "AAPL", date(2026, 9, 10))
    assert fetched.entry_geometry == geometry


def test_ticker_daily_state_entry_geometry_defaults_to_none(db_session: Session):
    repo = TickerDailyStateRepository(db_session)
    repo.upsert(_daily_state("MSFT", date(2026, 9, 10)))
    fetched = repo.get("us", "MSFT", date(2026, 9, 10))
    assert fetched.entry_geometry is None


def test_ticker_daily_state_upsert_overwrites_same_day_not_duplicates(db_session: Session):
    repo = TickerDailyStateRepository(db_session)
    repo.upsert(_daily_state("AAPL", date(2026, 9, 10), price=100.0))
    repo.upsert(_daily_state("AAPL", date(2026, 9, 10), price=101.5))
    rows = repo.for_region_and_date("us", date(2026, 9, 10))
    assert len(rows) == 1
    assert rows[0].price == pytest.approx(101.5)


def test_ticker_daily_state_latest_for_ticker_picks_the_max_date(db_session: Session):
    repo = TickerDailyStateRepository(db_session)
    repo.upsert(_daily_state("AAPL", date(2026, 9, 9), price=99.0))
    repo.upsert(_daily_state("AAPL", date(2026, 9, 10), price=100.0))
    latest = repo.latest_for_ticker("AAPL")
    assert latest.trade_date == date(2026, 9, 10)
    assert latest.price == pytest.approx(100.0)


def test_ticker_daily_state_latest_by_region_one_row_per_ticker(db_session: Session):
    repo = TickerDailyStateRepository(db_session)
    repo.upsert(_daily_state("AAPL", date(2026, 9, 9)))
    repo.upsert(_daily_state("AAPL", date(2026, 9, 10)))  # AAPL's newer row should win
    repo.upsert(_daily_state("MSFT", date(2026, 9, 10)))
    rows = repo.latest_by_region("us")
    by_ticker = {r.ticker: r for r in rows}
    assert set(by_ticker) == {"AAPL", "MSFT"}
    assert by_ticker["AAPL"].trade_date == date(2026, 9, 10)


def test_ticker_daily_state_for_region_and_date_is_exact_not_latest(db_session: Session):
    repo = TickerDailyStateRepository(db_session)
    repo.upsert(_daily_state("AAPL", date(2026, 9, 9)))
    repo.upsert(_daily_state("AAPL", date(2026, 9, 10)))
    rows = repo.for_region_and_date("us", date(2026, 9, 9))
    assert [r.trade_date for r in rows] == [date(2026, 9, 9)]


# --- TickerIntradayStateRepository --------------------------------------------


def test_ticker_intraday_state_upsert_then_get(db_session: Session):
    repo = TickerIntradayStateRepository(db_session)
    state = TickerIntradayState(
        ticker="AAPL", region="us", updated_at=datetime.now(UTC), price=106.0, entry_already_triggered=True
    )
    repo.upsert(state)
    fetched = repo.get("AAPL")
    assert fetched.price == pytest.approx(106.0)
    assert fetched.entry_already_triggered is True


def test_ticker_intraday_state_upsert_overwrites_in_place(db_session: Session):
    repo = TickerIntradayStateRepository(db_session)
    repo.upsert(TickerIntradayState("AAPL", "us", datetime.now(UTC), 100.0, False))
    repo.upsert(TickerIntradayState("AAPL", "us", datetime.now(UTC), 108.0, True))
    rows = repo.list_by_region("us")
    assert len(rows) == 1
    assert rows[0].price == pytest.approx(108.0)
    assert rows[0].entry_already_triggered is True


# --- PositionDailyStateRepository ---------------------------------------------


def test_position_daily_state_upsert_and_latest_for_portfolio(db_session: Session, portfolio_id: int):
    repo = PositionDailyStateRepository(db_session)
    repo.upsert(
        PositionDailyState(
            id=None,
            portfolio_id=portfolio_id,
            ticker="AAPL",
            trade_date=date(2026, 9, 9),
            computed_at=datetime.now(UTC),
            urgency="hold",
            reasons=[],
            price=100.0,
            r_multiple=1.2,
            current_stop=95.0,
            engine_version="2026-09-audit-v7",
        )
    )
    repo.upsert(
        PositionDailyState(
            id=None,
            portfolio_id=portfolio_id,
            ticker="AAPL",
            trade_date=date(2026, 9, 10),
            computed_at=datetime.now(UTC),
            urgency="reduce",
            reasons=["Estancada sin progreso"],
            price=101.0,
            r_multiple=0.3,
            current_stop=96.0,
            engine_version="2026-09-audit-v7",
        )
    )
    latest = repo.latest_for_portfolio(portfolio_id)
    assert len(latest) == 1
    assert latest[0].urgency == "reduce"
    assert latest[0].trade_date == date(2026, 9, 10)


def test_position_daily_state_for_portfolio_and_date_is_exact(db_session: Session, portfolio_id: int):
    repo = PositionDailyStateRepository(db_session)
    repo.upsert(
        PositionDailyState(
            None, portfolio_id, "AAPL", date(2026, 9, 9), datetime.now(UTC), "hold", [], 100.0, 1.0, 95.0, "v7"
        )
    )
    rows = repo.for_portfolio_and_date(portfolio_id, date(2026, 9, 9))
    assert len(rows) == 1
    rows_other_day = repo.for_portfolio_and_date(portfolio_id, date(2026, 9, 10))
    assert rows_other_day == []


# --- TriggerEventRepository ----------------------------------------------------


def test_trigger_event_record_appends_without_deduping(db_session: Session):
    repo = TriggerEventRepository(db_session)
    event = TriggerEvent(
        id=None,
        entity_type="ticker",
        entity_key="AAPL",
        event_type="gate_passed",
        previous_value="false",
        new_value="true",
        occurred_at=datetime.now(UTC),
        details={"price": 105.0},
    )
    first = repo.record(event)
    second = repo.record(event)
    assert first.id != second.id  # two calls, two rows - no implicit dedup


def test_trigger_event_list_since_filters_by_time_and_entity_type(db_session: Session):
    repo = TriggerEventRepository(db_session)
    now = datetime.now(UTC)
    old_event = TriggerEvent(None, "ticker", "AAPL", "gate_passed", None, "true", now - timedelta(days=5), {})
    recent_ticker_event = TriggerEvent(None, "ticker", "MSFT", "gate_passed", None, "true", now, {})
    recent_position_event = TriggerEvent(
        None, "position", "1:AAPL", "exit_urgency_escalated", "hold", "reduce", now, {}
    )
    repo.record(old_event)
    repo.record(recent_ticker_event)
    repo.record(recent_position_event)

    since = now - timedelta(hours=1)
    all_recent = repo.list_since(since)
    assert len(all_recent) == 2

    ticker_only = repo.list_since(since, entity_type="ticker")
    assert len(ticker_only) == 1
    assert ticker_only[0].entity_key == "MSFT"


# --- DailyBriefRepository -------------------------------------------------------


def test_daily_brief_upsert_and_latest_for_portfolio(db_session: Session, portfolio_id: int):
    repo = DailyBriefRepository(db_session)
    repo.upsert(
        DailyBrief(
            id=None,
            portfolio_id=portfolio_id,
            brief_date=date(2026, 9, 9),
            computed_at=datetime.now(UTC),
            positions_needing_action=1,
            new_entry_triggers=2,
            new_gate_passes=3,
            headline="1 posición necesita atención hoy.",
        )
    )
    repo.upsert(
        DailyBrief(
            id=None,
            portfolio_id=portfolio_id,
            brief_date=date(2026, 9, 10),
            computed_at=datetime.now(UTC),
            positions_needing_action=0,
            new_entry_triggers=1,
            new_gate_passes=0,
            headline="Todo tranquilo hoy.",
        )
    )
    latest = repo.latest_for_portfolio(portfolio_id)
    assert latest.brief_date == date(2026, 9, 10)
    assert latest.headline == "Todo tranquilo hoy."


def test_daily_brief_upsert_overwrites_same_day(db_session: Session, portfolio_id: int):
    repo = DailyBriefRepository(db_session)
    repo.upsert(
        DailyBrief(None, portfolio_id, date(2026, 9, 10), datetime.now(UTC), 0, 0, 0, "Primera versión")
    )
    repo.upsert(
        DailyBrief(None, portfolio_id, date(2026, 9, 10), datetime.now(UTC), 5, 5, 5, "Versión corregida")
    )
    latest = repo.latest_for_portfolio(portfolio_id)
    assert latest.headline == "Versión corregida"
    assert latest.positions_needing_action == 5
