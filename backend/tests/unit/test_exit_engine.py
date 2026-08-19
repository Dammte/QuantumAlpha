"""Every case here is a hand-built, obviously-correct scenario (per the
project's own testing philosophy) - none of these fight synthetic price
series to hit an exact SMA boundary, they construct the already-computed
reads exit_engine actually consumes and check the urgency/reasons that come
out."""

import ast
from datetime import date
from pathlib import Path

import pytest

from app.services import exit_engine as ee
from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta


def _timeframe_read(
    timeframe: str = "daily",
    trend: ta.TrendState = ta.TrendState.SIDEWAYS,
    stage: ta.Stage | None = None,
    ma_cross_50_200: str | None = None,
    cross_quality_20_50: ta.CrossQuality | None = None,
    imminent_cross_50_200: ta.ImminentCross | None = None,
    imminent_cross_20_50: ta.ImminentCross | None = None,
    macd_cross: str | None = None,
    price_vs_sma20: str | None = None,
    price_vs_sma50: str | None = None,
) -> mtf.TimeframeRead:
    return mtf.TimeframeRead(
        timeframe=timeframe,
        trend=trend,
        stage=stage,
        ma_cross_50_200=ma_cross_50_200,
        ma_cross_20_50=cross_quality_20_50.direction if cross_quality_20_50 else None,
        cross_quality_20_50=cross_quality_20_50,
        imminent_cross_50_200=imminent_cross_50_200,
        imminent_cross_20_50=imminent_cross_20_50,
        macd_cross=macd_cross,
        macd_histogram_turning=None,
        rsi14=None,
        adx14=None,
        di_bias=None,
        price_vs_sma20=price_vs_sma20,
        price_vs_sma50=price_vs_sma50,
        price_vs_sma200=None,
        bars_since_cross=None,
    )


def _mtf(
    daily: mtf.TimeframeRead | None = None,
    weekly: mtf.TimeframeRead | None = None,
    alignment: str = "transitioning",
) -> mtf.MultiTimeframeRead:
    return mtf.MultiTimeframeRead(
        weekly=weekly,
        daily=daily or _timeframe_read(),
        intraday=None,
        alignment=alignment,
        alignment_score=0.0,
        conflicts=[],
    )


def _position(
    initial_stop: float | None = None,
    current_stop: float | None = None,
    initial_target: float | None = None,
    r_multiple: float | None = None,
    bars_held: int = 10,
) -> ee.PositionContext:
    return ee.PositionContext(
        ticker="TEST",
        average_cost=100.0,
        quantity=10.0,
        opened_at=date(2024, 1, 1),
        initial_stop=initial_stop,
        current_stop=current_stop,
        initial_target=initial_target,
        highest_close_since_entry=100.0,
        unrealized_pnl_pct=None,
        r_multiple=r_multiple,
        bars_held=bars_held,
    )


def _evaluate(
    price: float = 100.0,
    position: ee.PositionContext | None = None,
    multi_timeframe: mtf.MultiTimeframeRead | None = None,
    consecutive_closes_below_daily_sma50: int = 0,
    consecutive_closes_below_daily_sma_fast: int = 0,
    nearest_support: ta.PriceLevel | None = None,
    nearest_resistance: ta.PriceLevel | None = None,
    obv_divergence: str | None = None,
    relative_volume: float | None = None,
    rsi14: float | None = None,
    rsi_recent_max: float | None = None,
    adx14: float | None = None,
    adx_recent_max: float | None = None,
    atr_multiple: float | None = None,
    candlestick_pattern: str | None = None,
) -> ee.ExitAssessment:
    return ee.evaluate_exit(
        price=price,
        position=position or _position(),
        multi_timeframe=multi_timeframe or _mtf(),
        consecutive_closes_below_daily_sma50=consecutive_closes_below_daily_sma50,
        consecutive_closes_below_daily_sma_fast=consecutive_closes_below_daily_sma_fast,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        obv_divergence=obv_divergence,
        relative_volume=relative_volume,
        rsi14=rsi14,
        rsi_recent_max=rsi_recent_max,
        adx14=adx14,
        adx_recent_max=adx_recent_max,
        atr_multiple=atr_multiple,
        candlestick_pattern=candlestick_pattern,
    )


def test_hold_when_nothing_is_triggered():
    result = _evaluate()
    assert result.urgency == ee.ExitUrgency.HOLD
    assert len(result.reasons) == 1


# --- EXIT_NOW ------------------------------------------------------------


def test_exit_now_when_price_breaks_the_current_stop():
    result = _evaluate(price=95.0, position=_position(current_stop=98.0))
    assert result.urgency == ee.ExitUrgency.EXIT_NOW
    assert any("perforado el stop" in r for r in result.reasons)


def test_no_exit_now_when_price_is_still_above_the_stop():
    result = _evaluate(price=99.0, position=_position(current_stop=98.0))
    assert result.urgency != ee.ExitUrgency.EXIT_NOW


def test_exit_now_two_consecutive_closes_below_sma50_with_weekly_not_bullish():
    weekly = _timeframe_read("weekly", trend=ta.TrendState.DOWNTREND)
    result = _evaluate(
        multi_timeframe=_mtf(weekly=weekly, alignment="conflicted"),
        consecutive_closes_below_daily_sma50=2,
    )
    assert result.urgency == ee.ExitUrgency.EXIT_NOW
    assert any("2 sesiones consecutivas" in r for r in result.reasons)


def test_no_exit_now_from_sma50_break_when_weekly_is_bullish():
    # Same daily break, but the weekly is still clearly bullish - the rule
    # requires "cuando la tendencia semanal ya no es alcista".
    weekly = _timeframe_read("weekly", trend=ta.TrendState.UPTREND)
    result = _evaluate(
        multi_timeframe=_mtf(weekly=weekly, alignment="conflicted"),
        consecutive_closes_below_daily_sma50=2,
    )
    assert not any("SMA50 diaria" in r for r in result.reasons)


def test_exit_now_one_close_below_sma50_with_strong_volume_and_weekly_not_bullish():
    weekly = _timeframe_read("weekly", trend=ta.TrendState.DOWNTREND)
    result = _evaluate(
        multi_timeframe=_mtf(weekly=weekly, alignment="conflicted"),
        consecutive_closes_below_daily_sma50=1,
        relative_volume=2.0,
    )
    assert result.urgency == ee.ExitUrgency.EXIT_NOW
    assert any("volumen relativo elevado" in r for r in result.reasons)


def test_no_exit_now_one_close_below_sma50_without_volume_confirmation():
    weekly = _timeframe_read("weekly", trend=ta.TrendState.DOWNTREND)
    result = _evaluate(
        multi_timeframe=_mtf(weekly=weekly, alignment="conflicted"),
        consecutive_closes_below_daily_sma50=1,
        relative_volume=1.1,
    )
    assert result.urgency != ee.ExitUrgency.EXIT_NOW


def test_exit_now_when_alignment_is_bearish_aligned():
    result = _evaluate(multi_timeframe=_mtf(alignment="bearish_aligned"))
    assert result.urgency == ee.ExitUrgency.EXIT_NOW
    assert any("Alineación bajista total" in r for r in result.reasons)


def test_exit_now_confirmed_death_cross_20_50_with_quality_below_both_mas():
    quality = ta.CrossQuality(
        direction="death", bars_since=1, separation_atr=1.0, fast_slope=-0.02, slow_slope=0.0,
        volume_confirmation=1.5, quality="strong",
    )
    daily = _timeframe_read(cross_quality_20_50=quality, price_vs_sma20="below", price_vs_sma50="below")
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency == ee.ExitUrgency.EXIT_NOW
    assert any("Death cross SMA21/SMA50 confirmado" in r for r in result.reasons)


def test_no_exit_now_death_cross_20_50_when_quality_is_noise():
    quality = ta.CrossQuality(
        direction="death", bars_since=1, separation_atr=0.05, fast_slope=0.0, slow_slope=0.0,
        volume_confirmation=1.0, quality="noise",
    )
    daily = _timeframe_read(cross_quality_20_50=quality, price_vs_sma20="below", price_vs_sma50="below")
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert not any("Death cross SMA21/SMA50 confirmado" in r for r in result.reasons)


def test_exit_now_when_last_relevant_support_is_broken():
    support = ta.PriceLevel(price=95.0, kind="support", strength=2, distance_pct=-0.02)
    result = _evaluate(price=94.0, nearest_support=support)
    assert result.urgency == ee.ExitUrgency.EXIT_NOW
    assert any("Rotura del último soporte" in r for r in result.reasons)


def test_no_exit_now_when_price_is_still_above_support():
    support = ta.PriceLevel(price=95.0, kind="support", strength=2, distance_pct=-0.02)
    result = _evaluate(price=96.0, nearest_support=support)
    assert not any("Rotura del último soporte" in r for r in result.reasons)


# --- REDUCE ----------------------------------------------------------------


def test_reduce_imminent_death_cross_50_200_with_high_confidence_and_daily_not_uptrend():
    imminent = ta.ImminentCross(direction="death", bars_until=3, r_squared=0.85)
    daily = _timeframe_read(trend=ta.TrendState.SIDEWAYS, imminent_cross_50_200=imminent)
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency == ee.ExitUrgency.REDUCE
    assert any("proyectado en ~3 sesiones" in r for r in result.reasons)


def test_no_reduce_imminent_death_cross_50_200_when_daily_is_still_uptrend():
    # The REDUCE rule itself doesn't fire (still requires the WATCH-tier
    # fallback text below, not the REDUCE-specific "tendencia diaria ya no
    # alcista" wording) - a weaker WATCH-level mention is fine and expected.
    imminent = ta.ImminentCross(direction="death", bars_until=3, r_squared=0.85)
    daily = _timeframe_read(trend=ta.TrendState.UPTREND, imminent_cross_50_200=imminent)
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency != ee.ExitUrgency.REDUCE
    assert not any("tendencia diaria ya no alcista" in r for r in result.reasons)


def test_no_reduce_imminent_death_cross_50_200_below_r2_threshold():
    imminent = ta.ImminentCross(direction="death", bars_until=3, r_squared=0.5)
    daily = _timeframe_read(trend=ta.TrendState.SIDEWAYS, imminent_cross_50_200=imminent)
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency != ee.ExitUrgency.REDUCE
    assert not any("tendencia diaria ya no alcista" in r for r in result.reasons)


def test_reduce_bearish_obv_divergence_with_rsi_falling_from_overbought_and_growing_volume():
    result = _evaluate(obv_divergence="bearish", rsi14=65.0, rsi_recent_max=78.0, relative_volume=1.4)
    assert result.urgency == ee.ExitUrgency.REDUCE
    assert any("Divergencia bajista de OBV" in r for r in result.reasons)


def test_no_reduce_obv_divergence_without_rsi_falling_from_overbought():
    result = _evaluate(obv_divergence="bearish", rsi14=65.0, rsi_recent_max=68.0, relative_volume=1.4)
    assert not any("Divergencia bajista de OBV" in r for r in result.reasons)


def test_reduce_when_original_target_is_reached():
    result = _evaluate(price=120.0, position=_position(initial_target=118.0))
    assert result.urgency == ee.ExitUrgency.REDUCE
    assert any("Objetivo original alcanzado" in r for r in result.reasons)


def test_reduce_extended_beyond_4_atr_while_in_profit():
    result = _evaluate(atr_multiple=4.5, position=_position(r_multiple=1.2))
    assert result.urgency == ee.ExitUrgency.REDUCE
    assert any("Extensión parabólica" in r for r in result.reasons)


def test_no_reduce_extended_beyond_4_atr_when_not_in_profit():
    result = _evaluate(atr_multiple=4.5, position=_position(r_multiple=-0.3))
    assert not any("Extensión parabólica" in r for r in result.reasons)


def test_reduce_when_position_has_stalled_without_progress():
    # r_multiple below +1R after 20+ sessions, still above its stop (hasn't
    # stopped out, just isn't going anywhere).
    result = _evaluate(
        price=101.0, position=_position(current_stop=95.0, r_multiple=0.2, bars_held=25)
    )
    assert result.urgency == ee.ExitUrgency.REDUCE
    assert any("sin progreso tras 25 sesiones" in r for r in result.reasons)


def test_no_stalled_reduce_before_max_bars_without_progress():
    result = _evaluate(price=101.0, position=_position(current_stop=95.0, r_multiple=0.2, bars_held=10))
    assert not any("sin progreso" in r for r in result.reasons)


def test_no_stalled_reduce_once_plus_1r_is_reached():
    result = _evaluate(price=101.0, position=_position(current_stop=95.0, r_multiple=1.2, bars_held=25))
    assert not any("sin progreso" in r for r in result.reasons)


def test_no_stalled_reduce_without_a_resolvable_r_multiple():
    # No initial_stop -> no r_multiple denominator -> nothing to call "stalled".
    result = _evaluate(price=101.0, position=_position(current_stop=None, r_multiple=None, bars_held=25))
    assert not any("sin progreso" in r for r in result.reasons)


def test_reduce_on_a_fresh_confirmed_break_below_the_fast_daily_sma():
    # Real-world motivating case: a held position scoring well on the buy
    # checklist (would otherwise stay ADD_CANDIDATE) closes below its own
    # SMA21 for the first time - the propietario's own primary short-term
    # timing signal. consecutive_closes_below_daily_sma_fast == 1 means
    # today is the break itself, not an old, already-known one.
    result = _evaluate(consecutive_closes_below_daily_sma_fast=1)
    assert result.urgency == ee.ExitUrgency.REDUCE
    assert any(f"por debajo de la SMA{mtf.FAST_MA_PERIOD} diaria" in r for r in result.reasons)


def test_fast_sma_break_reduce_overrides_add_candidate_via_portfolio_risk_service_precedence():
    # Confirms the actual bug report this fixes: REDUCE (unlike TIGHTEN_STOP)
    # is one of the two urgencies portfolio_risk_service.py lets override an
    # ADD_CANDIDATE badge to EXIT_WARNING - see its precedence comment. This
    # test only checks the urgency tier itself; the override behavior is
    # exercised in test_portfolio_risk_service.py.
    result = _evaluate(consecutive_closes_below_daily_sma_fast=1)
    assert result.urgency in (ee.ExitUrgency.EXIT_NOW, ee.ExitUrgency.REDUCE)


def test_no_reduce_when_fast_sma_break_is_not_fresh():
    # Already known/reported on a prior evaluation (3 consecutive sessions,
    # not the day of the break) - stays visible at WATCH, doesn't re-fire REDUCE.
    result = _evaluate(consecutive_closes_below_daily_sma_fast=3)
    assert result.urgency == ee.ExitUrgency.WATCH
    assert not any("Cierre confirmado por debajo" in r for r in result.reasons)


def test_watch_when_price_has_stayed_below_the_fast_sma_for_several_sessions():
    result = _evaluate(consecutive_closes_below_daily_sma_fast=4)
    assert result.urgency == ee.ExitUrgency.WATCH
    assert any(f"sigue por debajo de la SMA{mtf.FAST_MA_PERIOD} diaria" in r for r in result.reasons)
    assert any("4 sesiones consecutivas" in r for r in result.reasons)


def test_no_fast_sma_watch_or_reduce_when_price_is_above_it():
    result = _evaluate(consecutive_closes_below_daily_sma_fast=0)
    assert result.urgency == ee.ExitUrgency.HOLD


# --- TIGHTEN_STOP ------------------------------------------------------------


def test_tighten_stop_imminent_death_cross_20_50():
    imminent = ta.ImminentCross(direction="death", bars_until=4, r_squared=0.65)
    daily = _timeframe_read(imminent_cross_20_50=imminent)
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency == ee.ExitUrgency.TIGHTEN_STOP
    assert any("corto plazo (SMA21/SMA50) proyectado" in r for r in result.reasons)


def test_no_tighten_stop_imminent_death_cross_20_50_below_r2_threshold():
    imminent = ta.ImminentCross(direction="death", bars_until=4, r_squared=0.4)
    daily = _timeframe_read(imminent_cross_20_50=imminent)
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency == ee.ExitUrgency.HOLD


def test_tighten_stop_when_adx_falls_from_a_real_trend():
    result = _evaluate(adx14=18.0, adx_recent_max=30.0)
    assert result.urgency == ee.ExitUrgency.TIGHTEN_STOP
    assert any("ADX cayendo" in r for r in result.reasons)


def test_no_tighten_stop_when_adx_was_never_really_trending():
    result = _evaluate(adx14=18.0, adx_recent_max=22.0)
    assert result.urgency == ee.ExitUrgency.HOLD


def test_tighten_stop_on_confirmed_daily_macd_bearish_cross():
    daily = _timeframe_read(macd_cross="bearish")
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency == ee.ExitUrgency.TIGHTEN_STOP
    assert any("Cruce bajista de MACD confirmado" in r for r in result.reasons)


def test_no_tighten_stop_on_bullish_macd_cross():
    daily = _timeframe_read(macd_cross="bullish")
    result = _evaluate(multi_timeframe=_mtf(daily=daily))
    assert result.urgency == ee.ExitUrgency.HOLD


def test_tighten_stop_bearish_engulfing_at_resistance():
    resistance = ta.PriceLevel(price=101.0, kind="resistance", strength=2, distance_pct=0.01)
    result = _evaluate(nearest_resistance=resistance, candlestick_pattern="bearish_engulfing")
    assert result.urgency == ee.ExitUrgency.TIGHTEN_STOP
    assert any("Vela envolvente bajista justo en un nivel de resistencia" in r for r in result.reasons)


def test_tighten_stop_when_position_is_up_more_than_1_5r():
    result = _evaluate(position=_position(r_multiple=2.0))
    assert result.urgency == ee.ExitUrgency.TIGHTEN_STOP
    assert any("beneficio de 2.0R" in r for r in result.reasons)


def test_no_tighten_stop_at_exactly_1_5r():
    result = _evaluate(position=_position(r_multiple=1.5))
    assert result.urgency == ee.ExitUrgency.HOLD


# --- WATCH -------------------------------------------------------------------


def test_watch_when_price_is_near_support():
    support = ta.PriceLevel(price=99.0, kind="support", strength=1, distance_pct=-0.01)
    result = _evaluate(price=100.0, nearest_support=support)
    assert result.urgency == ee.ExitUrgency.WATCH


def test_watch_bearish_engulfing_away_from_resistance():
    result = _evaluate(candlestick_pattern="bearish_engulfing")
    assert result.urgency == ee.ExitUrgency.WATCH
    assert any("Vela envolvente bajista en la última sesión" in r for r in result.reasons)


# --- severity ordering: the most severe tier wins, but every reason across
# every tier that fired is still reported -------------------------------------


def test_exit_now_outranks_a_simultaneously_triggered_tighten_stop_and_all_reasons_are_kept():
    result = _evaluate(
        price=95.0,
        position=_position(current_stop=98.0, r_multiple=2.0),  # EXIT_NOW + TIGHTEN_STOP both fire
    )
    assert result.urgency == ee.ExitUrgency.EXIT_NOW
    assert any("perforado el stop" in r for r in result.reasons)
    assert any("beneficio de 2.0R" in r for r in result.reasons)


# --- the acceptance test: this is the scenario the whole exercise is about ---


def test_weekly_bearish_plus_projected_20_50_death_cross_plus_below_sma50_never_holds():
    """D3's acceptance test: an asset with a bearish weekly, a high-confidence
    imminent SMA21/50 death cross, and price already confirmed below the
    daily SMA50 must return EXIT_NOW or REDUCE - never HOLD - *even with* an
    RS Rating of 85 and excellent fundamentals. Those two are not simulated
    as "passed but ignored": evaluate_exit's signature has no rs_rating,
    revenue_growth, profit_margins, or debt_to_equity parameter at all, so
    there is no way for them to reach this decision, structurally, not just
    by convention - the same thing `grep -r recommendation_engine
    app/services/exit_engine.py` (empty) confirms about the buy-side score
    itself.
    """
    weekly = _timeframe_read("weekly", trend=ta.TrendState.DOWNTREND)
    imminent_20_50 = ta.ImminentCross(direction="death", bars_until=3, r_squared=0.85)
    daily = _timeframe_read(
        trend=ta.TrendState.SIDEWAYS,
        imminent_cross_20_50=imminent_20_50,
        price_vs_sma50="below",
    )
    result = _evaluate(
        multi_timeframe=_mtf(daily=daily, weekly=weekly, alignment="conflicted"),
        consecutive_closes_below_daily_sma50=2,
    )
    assert result.urgency in (ee.ExitUrgency.EXIT_NOW, ee.ExitUrgency.REDUCE)
    assert result.urgency != ee.ExitUrgency.HOLD


def test_exit_engine_never_imports_recommendation_engine():
    """The acceptance criterion from the brief, enforced automatically (not
    just checkable by a manual grep): exit_engine.py must never import
    build_recommendation or anything else from recommendation_engine.py, so a
    held position's exit urgency can never be influenced by the factors that
    justified the original buy (RS Rating, fundamentals, Minervini, 52-week
    range). Checked via the AST's actual import nodes, not a text search -
    the module's own docstring legitimately *names*
    recommendation_engine.py in prose to explain this very guarantee, which
    a blind grep would misfire on."""
    tree = ast.parse(Path(ee.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any("recommendation_engine" in module for module in imported_modules)


@pytest.mark.parametrize("urgency", list(ee.ExitUrgency))
def test_every_urgency_level_is_a_valid_string_value(urgency: ee.ExitUrgency):
    # Cheap guard against a typo silently breaking the API/frontend contract
    # (SIGNAL_LABELS-style maps in the frontend key off these exact strings).
    assert urgency.value == urgency.value.lower()
