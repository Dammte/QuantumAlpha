"""Unit tests for the pure pieces of scripts/intraday_refresh.py - see
test_daily_close.py's own docstring for the import-path note
(`scripts.intraday_refresh`, no manual sys.path hack, relies on pytest's
`pythonpath = ["."]`)."""

from datetime import UTC, date, datetime, timedelta

import scripts.intraday_refresh as ir
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_intraday_state import TickerIntradayState


def _daily_state(**overrides) -> TickerDailyState:
    defaults = dict(
        id=None,
        region="us",
        ticker="AAPL",
        trade_date=date(2026, 9, 10),
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
        entry_trigger_price=125.0,
        entry_already_triggered=False,
        stop_loss=110.0,
        take_profit=140.0,
        take_profit_method="objetivo 2:1 sobre el riesgo",
        risk_reward=2.0,
    )
    defaults.update(overrides)
    return TickerDailyState(**defaults)


# --- compute_intraday_state ---------------------------------------------------


def test_breakout_triggered_when_price_clears_the_level():
    state = _daily_state(entry_trigger_type="breakout", entry_trigger_price=125.0)
    result = ir.compute_intraday_state(state, price=126.0, updated_at=datetime.now(UTC))
    assert result.entry_already_triggered is True
    assert result.price == 126.0
    assert result.ticker == "AAPL"
    assert result.region == "us"


def test_breakout_not_triggered_when_price_below_the_level():
    state = _daily_state(entry_trigger_type="breakout", entry_trigger_price=125.0)
    result = ir.compute_intraday_state(state, price=124.0, updated_at=datetime.now(UTC))
    assert result.entry_already_triggered is False


def test_breakout_triggered_at_the_exact_boundary():
    state = _daily_state(entry_trigger_type="breakout", entry_trigger_price=125.0)
    result = ir.compute_intraday_state(state, price=125.0, updated_at=datetime.now(UTC))
    assert result.entry_already_triggered is True


def test_pullback_bounce_carries_forward_the_daily_value_regardless_of_price():
    state = _daily_state(entry_trigger_type="pullback_bounce", entry_already_triggered=True)
    # A price far below the pullback level shouldn't flip this to False -
    # pullback re-checking isn't this job's concern, see module docstring.
    result = ir.compute_intraday_state(state, price=1.0, updated_at=datetime.now(UTC))
    assert result.entry_already_triggered is True


# --- previous_already_triggered ------------------------------------------------


def test_uses_existing_intraday_row_when_it_is_from_today():
    today = date(2026, 9, 10)
    daily = _daily_state(entry_already_triggered=False)
    existing = TickerIntradayState(
        ticker="AAPL", region="us", updated_at=datetime.combine(today, datetime.min.time(), tzinfo=UTC),
        price=126.0, entry_already_triggered=True,
    )
    assert ir.previous_already_triggered(daily, existing, today) is True


def test_falls_back_to_daily_state_when_intraday_row_is_from_a_prior_day():
    today = date(2026, 9, 10)
    yesterday = today - timedelta(days=1)
    daily = _daily_state(entry_already_triggered=False)
    stale = TickerIntradayState(
        ticker="AAPL", region="us", updated_at=datetime.combine(yesterday, datetime.min.time(), tzinfo=UTC),
        price=126.0, entry_already_triggered=True,
    )
    assert ir.previous_already_triggered(daily, stale, today) is False


def test_falls_back_to_daily_state_when_no_intraday_row_exists_yet():
    today = date(2026, 9, 10)
    daily = _daily_state(entry_already_triggered=False)
    assert ir.previous_already_triggered(daily, None, today) is False


# --- intraday_trigger_event -----------------------------------------------------


def test_no_event_when_already_triggered_before():
    assert ir.intraday_trigger_event("AAPL", True, True, 126.0, datetime.now(UTC)) is None


def test_no_event_when_still_not_triggered():
    assert ir.intraday_trigger_event("AAPL", False, False, 120.0, datetime.now(UTC)) is None


def test_fires_on_false_to_true_transition():
    now = datetime.now(UTC)
    event = ir.intraday_trigger_event("AAPL", False, True, 126.0, now)
    assert event is not None
    assert event.entity_type == "ticker"
    assert event.entity_key == "AAPL"
    assert event.event_type == "entry_triggered"
    assert event.previous_value == "false"
    assert event.new_value == "true"
    assert event.details == {"price": 126.0, "source": "intraday"}


def test_no_event_on_a_true_to_false_reversal():
    # Job A's own nightly recompute is the authoritative record for this
    # direction - see the function's own docstring.
    assert ir.intraday_trigger_event("AAPL", True, False, 118.0, datetime.now(UTC)) is None
