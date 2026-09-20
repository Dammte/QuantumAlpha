from datetime import UTC, date, datetime

import pytest

from app.domain.models.ticker_daily_state import TickerDailyState
from app.services import market_regime_service as mrs


def _state(ticker: str, weekly_price_vs_ma: str | None) -> TickerDailyState:
    timeframe_strip = (
        {"weekly": {"price_vs_ma": weekly_price_vs_ma}, "daily": {}, "monthly": {}}
        if weekly_price_vs_ma is not None
        else None
    )
    return TickerDailyState(
        id=None,
        region="us",
        ticker=ticker,
        trade_date=date(2026, 9, 19),
        computed_at=datetime(2026, 9, 19, tzinfo=UTC),
        price=100.0,
        currency="USD",
        trend="uptrend",
        stage="stage2",
        rs_rating=None,
        adx14=None,
        atr_multiple=None,
        rsi14=None,
        gate_passes=True,
        gate_conditions=[],
        gate_version="v1",
        entry_trigger_type=None,
        entry_trigger_price=None,
        entry_already_triggered=False,
        stop_loss=None,
        take_profit=None,
        take_profit_method=None,
        risk_reward=None,
        timeframe_strip=timeframe_strip,
    )


# --- weekly_breadth_pct -------------------------------------------------------


def test_weekly_breadth_pct_none_without_any_reading():
    assert mrs.weekly_breadth_pct([_state("A", None)]) is None


def test_weekly_breadth_pct_none_for_an_empty_universe():
    assert mrs.weekly_breadth_pct([]) is None


def test_weekly_breadth_pct_counts_above_fraction():
    states = [_state("A", "above"), _state("B", "above"), _state("C", "below"), _state("D", "below")]
    assert mrs.weekly_breadth_pct(states) == 0.5


def test_weekly_breadth_pct_ignores_states_without_a_reading():
    states = [_state("A", "above"), _state("B", None)]
    assert mrs.weekly_breadth_pct(states) == 1.0


# --- assess_trend_regime -------------------------------------------------------


def test_unknown_when_index_reading_is_missing():
    states = [_state("A", "above")]
    result = mrs.assess_trend_regime(states, None, index_above_weekly_ma30=None)
    assert result.status == mrs.REGIME_UNKNOWN


def test_unknown_when_breadth_cannot_be_computed():
    result = mrs.assess_trend_regime([], None, index_above_weekly_ma30=True)
    assert result.status == mrs.REGIME_UNKNOWN


def test_bearish_when_index_is_below_its_own_weekly_ma30():
    states = [_state("A", "above"), _state("B", "above")]  # amplitud perfecta, pero el índice manda
    result = mrs.assess_trend_regime(states, None, index_above_weekly_ma30=False)
    assert result.status == mrs.REGIME_BEARISH
    assert "media de 30 semanas" in result.headline


def test_bearish_when_breadth_is_below_the_threshold_even_if_the_index_holds():
    states = [_state("A", "below"), _state("B", "below"), _state("C", "above")]  # 33% breadth
    result = mrs.assess_trend_regime(states, None, index_above_weekly_ma30=True)
    assert result.status == mrs.REGIME_BEARISH
    assert result.breadth_pct == pytest.approx(1 / 3)


def test_bullish_when_index_holds_and_breadth_clears_the_threshold():
    states = [_state("A", "above"), _state("B", "above"), _state("C", "below")]  # 67% breadth
    result = mrs.assess_trend_regime(states, None, index_above_weekly_ma30=True)
    assert result.status == mrs.REGIME_BULLISH


def test_breadth_change_5d_is_the_difference_from_the_prior_snapshot():
    today = [_state("A", "above"), _state("B", "above")]  # 100%
    five_days_ago = [_state("A", "below"), _state("B", "above")]  # 50%
    result = mrs.assess_trend_regime(today, five_days_ago, index_above_weekly_ma30=True)
    assert result.breadth_change_5d == pytest.approx(0.5)


def test_breadth_change_5d_is_none_without_a_prior_snapshot():
    result = mrs.assess_trend_regime([_state("A", "above")], None, index_above_weekly_ma30=True)
    assert result.breadth_change_5d is None
