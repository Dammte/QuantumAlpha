"""Reconstruction (2026-09), Fase 2: end-to-end orchestration test for
scripts/intraday_refresh.py's `run_intraday_refresh` - see
test_daily_close_job.py's own docstring for why this is a thin plumbing
check on top of the already-unit-tested pure functions
(tests/unit/test_intraday_refresh.py)."""

from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

import scripts.intraday_refresh as ir
from app.domain.models.ticker_daily_state import TickerDailyState
from app.infrastructure.db.repositories.job_run_repository import JobRunRepository
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.infrastructure.db.repositories.ticker_intraday_state_repository import TickerIntradayStateRepository
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository
from app.services.market_data_service import MarketDataService
from tests.integration.conftest import FakeMarketDataProvider


def _seed_daily_state(db: Session, **overrides) -> TickerDailyState:
    defaults = dict(
        id=None,
        region="us",
        ticker="AAPL",
        trade_date=date.today(),
        computed_at=datetime.now(UTC),
        price=120.0,
        currency="USD",
        trend="uptrend",
        stage="stage2",
        rs_rating=85,
        adx14=28.0,
        atr_multiple=1.2,
        rsi14=55.0,
        gate_passes=True,
        gate_conditions=[],
        gate_version="2026-09-levels-v1",
        entry_trigger_type="breakout",
        entry_trigger_price=125.0,  # FakeMarketDataProvider quotes every ticker at 150.0 - clears this
        entry_already_triggered=False,
        stop_loss=110.0,
        take_profit=140.0,
        take_profit_method="objetivo 2:1 sobre el riesgo",
        risk_reward=2.0,
    )
    defaults.update(overrides)
    return TickerDailyStateRepository(db).upsert(TickerDailyState(**defaults))


def test_run_intraday_refresh_flags_a_fresh_breakout(db_session: Session):
    _seed_daily_state(db_session)
    market_data = MarketDataService(FakeMarketDataProvider())

    result = ir.run_intraday_refresh(db_session, market_data, regions=("us",))

    assert result.job_status == "success"
    assert result.rows_processed == 1
    assert result.new_triggers == 1

    intraday_repo = TickerIntradayStateRepository(db_session)
    state = intraday_repo.get("AAPL")
    assert state is not None
    assert state.price == 150.0
    assert state.entry_already_triggered is True

    events = TriggerEventRepository(db_session).list_since(datetime.now(UTC).replace(year=2000))
    entry_events = [e for e in events if e.event_type == "entry_triggered"]
    assert len(entry_events) == 1
    assert entry_events[0].entity_key == "AAPL"
    assert entry_events[0].details.get("source") == "intraday"


def test_run_intraday_refresh_records_job_run(db_session: Session):
    _seed_daily_state(db_session)
    market_data = MarketDataService(FakeMarketDataProvider())

    ir.run_intraday_refresh(db_session, market_data, regions=("us",))

    latest = JobRunRepository(db_session).latest("intraday_refresh")
    assert latest is not None
    assert latest.status == "success"


def test_run_intraday_refresh_skips_tickers_with_no_active_trigger(db_session: Session):
    _seed_daily_state(db_session, ticker="MSFT", entry_trigger_type=None, entry_trigger_price=None)
    market_data = MarketDataService(FakeMarketDataProvider())

    result = ir.run_intraday_refresh(db_session, market_data, regions=("us",))

    assert result.rows_processed == 0
    assert TickerIntradayStateRepository(db_session).get("MSFT") is None


def test_run_intraday_refresh_no_new_trigger_when_already_triggered_yesterday(db_session: Session):
    _seed_daily_state(db_session, entry_already_triggered=True)
    market_data = MarketDataService(FakeMarketDataProvider())

    result = ir.run_intraday_refresh(db_session, market_data, regions=("us",))

    assert result.rows_processed == 1
    assert result.new_triggers == 0  # was already triggered at close - nothing new to report
