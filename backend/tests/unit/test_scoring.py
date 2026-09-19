"""Auditoria del Radar, bloque E2 (docs/quant_methodology.md §29): score
compuesto del Radar. Cada componente se prueba aislado (todos los demás
inputs neutralizados) antes de probar el total, para poder atribuir un
cambio de puntuación a un componente concreto - el propio objetivo del
bloque ("necesito saber por qué un valor está el primero")."""

from datetime import date

import pytest

from app.core import trading_params as tp
from app.services.setups import scoring as sc

_AS_OF = date(2026, 9, 19)


def _input(**overrides) -> sc.RadarScoreInput:
    defaults = dict(
        grade=None, setup_confidence=None, rs_rating=None, sector_rs_percentile=None,
        distance_atr=None, risk_reward_net=None, entry_type=None, relative_volume=None,
        setup_stage=None, horizon=None, next_earnings_date=None, as_of=_AS_OF,
        atr_pct=None, atr_pct_p90_in_universe=None,
    )
    defaults.update(overrides)
    return sc.RadarScoreInput(**defaults)


# --- setup_quality -----------------------------------------------------------


def test_setup_quality_grade_a_measured_is_full_weight():
    result = sc.compute_radar_score(_input(grade="A", setup_confidence="measured"))
    assert result.setup_quality == pytest.approx(tp.RADAR_SCORE_WEIGHT_SETUP_QUALITY)


def test_setup_quality_unvalidated_confidence_penalizes_not_rewards():
    measured = sc.compute_radar_score(_input(grade="A", setup_confidence="measured")).setup_quality
    unvalidated = sc.compute_radar_score(_input(grade="A", setup_confidence="unvalidated")).setup_quality
    assert unvalidated < measured
    assert unvalidated == pytest.approx(tp.RADAR_SCORE_WEIGHT_SETUP_QUALITY * 0.5)


def test_setup_quality_no_grade_is_zero():
    result = sc.compute_radar_score(_input(grade=None, setup_confidence="measured"))
    assert result.setup_quality == 0.0


def test_setup_quality_ranks_a_above_b_above_c():
    a = sc.compute_radar_score(_input(grade="A", setup_confidence="measured")).setup_quality
    b = sc.compute_radar_score(_input(grade="B", setup_confidence="measured")).setup_quality
    c = sc.compute_radar_score(_input(grade="C", setup_confidence="measured")).setup_quality
    assert a > b > c > 0


# --- relative_strength ---------------------------------------------------


def test_relative_strength_max_at_100_and_100():
    result = sc.compute_radar_score(_input(rs_rating=100, sector_rs_percentile=100))
    assert result.relative_strength == pytest.approx(tp.RADAR_SCORE_WEIGHT_RELATIVE_STRENGTH)


def test_relative_strength_weighs_ticker_more_than_sector():
    ticker_strong = sc.compute_radar_score(_input(rs_rating=100, sector_rs_percentile=0)).relative_strength
    sector_strong = sc.compute_radar_score(_input(rs_rating=0, sector_rs_percentile=100)).relative_strength
    assert ticker_strong > sector_strong


def test_relative_strength_missing_values_treated_as_zero_not_crash():
    result = sc.compute_radar_score(_input(rs_rating=None, sector_rs_percentile=None))
    assert result.relative_strength == 0.0


# --- trigger_proximity ---------------------------------------------------


def test_trigger_proximity_at_zero_distance_is_full_weight():
    result = sc.compute_radar_score(_input(distance_atr=0.0))
    assert result.trigger_proximity == pytest.approx(tp.RADAR_SCORE_WEIGHT_TRIGGER_PROXIMITY)


def test_trigger_proximity_beyond_the_scale_is_zero_not_negative():
    result = sc.compute_radar_score(_input(distance_atr=10.0))
    assert result.trigger_proximity == 0.0


def test_trigger_proximity_closer_scores_higher():
    near = sc.compute_radar_score(_input(distance_atr=0.3)).trigger_proximity
    far = sc.compute_radar_score(_input(distance_atr=1.5)).trigger_proximity
    assert near > far


def test_trigger_proximity_none_distance_is_zero():
    assert sc.compute_radar_score(_input(distance_atr=None)).trigger_proximity == 0.0


# --- geometry_quality ------------------------------------------------------


def test_geometry_quality_real_level_anchor_beats_ma_anchor_at_same_risk_reward():
    level = sc.compute_radar_score(_input(risk_reward_net=2.5, entry_type="breakout")).geometry_quality
    ma = sc.compute_radar_score(_input(risk_reward_net=2.5, entry_type="pullback_ema21")).geometry_quality
    assert level > ma


def test_geometry_quality_at_minimum_risk_reward_is_only_the_anchor_component():
    result = sc.compute_radar_score(
        _input(risk_reward_net=tp.MIN_RISK_REWARD_NET, entry_type="breakout")
    )
    expected = tp.RADAR_SCORE_REAL_LEVEL_ANCHOR_SCORE * tp.RADAR_SCORE_ANCHOR_SUBWEIGHT
    expected *= tp.RADAR_SCORE_WEIGHT_GEOMETRY_QUALITY
    assert result.geometry_quality == pytest.approx(expected)


def test_geometry_quality_higher_risk_reward_scores_higher():
    low = sc.compute_radar_score(_input(risk_reward_net=1.6, entry_type="breakout")).geometry_quality
    high = sc.compute_radar_score(_input(risk_reward_net=4.0, entry_type="breakout")).geometry_quality
    assert high > low


def test_geometry_quality_no_entry_type_and_no_risk_reward_is_zero():
    result = sc.compute_radar_score(_input(risk_reward_net=None, entry_type=None))
    assert result.geometry_quality == 0.0


# --- volume_confirmation ---------------------------------------------------


def test_volume_confirmation_high_volume_on_trigger_scores_high():
    result = sc.compute_radar_score(_input(relative_volume=2.5, setup_stage="triggered"))
    assert result.volume_confirmation == pytest.approx(tp.RADAR_SCORE_WEIGHT_VOLUME_CONFIRMATION)


def test_volume_confirmation_low_volume_on_trigger_scores_low():
    result = sc.compute_radar_score(_input(relative_volume=0.4, setup_stage="triggered"))
    assert result.volume_confirmation == 0.0


def test_volume_confirmation_low_volume_while_forming_scores_high():
    # Contracción de volumen mientras la base se forma - lo deseable.
    result = sc.compute_radar_score(_input(relative_volume=0.3, setup_stage="forming"))
    assert result.volume_confirmation == pytest.approx(tp.RADAR_SCORE_WEIGHT_VOLUME_CONFIRMATION)


def test_volume_confirmation_high_volume_while_forming_scores_low():
    result = sc.compute_radar_score(_input(relative_volume=2.0, setup_stage="forming"))
    assert result.volume_confirmation == 0.0


def test_volume_confirmation_missing_data_is_neutral_not_zero():
    result = sc.compute_radar_score(_input(relative_volume=None, setup_stage="triggered"))
    assert result.volume_confirmation == pytest.approx(tp.RADAR_SCORE_WEIGHT_VOLUME_CONFIRMATION * 0.5)


# --- earnings_penalty ------------------------------------------------------


def test_earnings_penalty_applies_within_window_on_short_horizon():
    result = sc.compute_radar_score(
        _input(horizon="short", next_earnings_date=date(2026, 9, 25))  # 6 días
    )
    assert result.earnings_penalty == -tp.RADAR_SCORE_EARNINGS_PENALTY_SHORT


def test_earnings_penalty_absent_beyond_the_window():
    result = sc.compute_radar_score(
        _input(horizon="short", next_earnings_date=date(2026, 11, 1))
    )
    assert result.earnings_penalty == 0.0


def test_earnings_penalty_never_applies_on_medium_horizon():
    # "En medio plazo, solo aviso" - literal, sin penalización numérica.
    result = sc.compute_radar_score(
        _input(horizon="medium", next_earnings_date=date(2026, 9, 20))
    )
    assert result.earnings_penalty == 0.0


def test_earnings_penalty_none_date_never_applies():
    result = sc.compute_radar_score(_input(horizon="short", next_earnings_date=None))
    assert result.earnings_penalty == 0.0


def test_earnings_penalty_a_past_earnings_date_does_not_apply():
    result = sc.compute_radar_score(
        _input(horizon="short", next_earnings_date=date(2026, 9, 1))
    )
    assert result.earnings_penalty == 0.0


# --- high_atr_penalty -------------------------------------------------------


def test_high_atr_penalty_applies_above_the_universe_p90():
    result = sc.compute_radar_score(_input(atr_pct=0.08, atr_pct_p90_in_universe=0.05))
    assert result.high_atr_penalty == -tp.RADAR_SCORE_HIGH_ATR_PENALTY


def test_high_atr_penalty_absent_at_or_below_the_p90():
    result = sc.compute_radar_score(_input(atr_pct=0.05, atr_pct_p90_in_universe=0.05))
    assert result.high_atr_penalty == 0.0


def test_high_atr_penalty_absent_without_a_universe_reference():
    result = sc.compute_radar_score(_input(atr_pct=0.20, atr_pct_p90_in_universe=None))
    assert result.high_atr_penalty == 0.0


# --- total -------------------------------------------------------------


def test_total_is_clamped_to_zero_when_penalties_would_push_it_negative():
    result = sc.compute_radar_score(
        _input(
            grade="C", setup_confidence="unvalidated", rs_rating=0, sector_rs_percentile=0,
            distance_atr=10.0, risk_reward_net=None, entry_type=None, relative_volume=3.0,
            setup_stage="forming", horizon="short", next_earnings_date=date(2026, 9, 20),
            atr_pct=0.5, atr_pct_p90_in_universe=0.05,
        )
    )
    assert result.total == 0.0


def test_total_is_the_sum_of_every_component_and_penalty():
    inp = _input(
        grade="A", setup_confidence="measured", rs_rating=90, sector_rs_percentile=80,
        distance_atr=0.2, risk_reward_net=3.0, entry_type="breakout", relative_volume=2.0,
        setup_stage="triggered", horizon="short", next_earnings_date=None,
        atr_pct=0.03, atr_pct_p90_in_universe=0.06,
    )
    result = sc.compute_radar_score(inp)
    expected = (
        result.setup_quality + result.relative_strength + result.trigger_proximity
        + result.geometry_quality + result.volume_confirmation
        + result.earnings_penalty + result.high_atr_penalty
    )
    assert result.total == pytest.approx(expected)


def test_a_fully_neutral_input_does_not_crash_and_only_scores_the_neutral_volume_component():
    # Sin ningún dato salvo el volumen (que trata "sin dato" como neutral,
    # no como cero - ver test_volume_confirmation_missing_data_is_neutral_not_zero)
    # el total es exactamente ese componente neutral, no cero.
    result = sc.compute_radar_score(_input())
    assert result.total == pytest.approx(tp.RADAR_SCORE_WEIGHT_VOLUME_CONFIRMATION * 0.5)
