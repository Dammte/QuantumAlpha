import pytest

from app.services import recommendation_engine as re
from app.services.markov_chain_model import MarkovChainResult
from app.services.technical_analysis import PriceLevel, Stage, TrendState
from app.services.volatility_model import GarchResult


def _markov(prob_bullish_21d: float, sequence_looks_random: bool = False) -> MarkovChainResult:
    return MarkovChainResult(
        current_state=3,
        current_state_label="alcista",
        state_labels=["fuerte bajista", "bajista", "lateral", "alcista", "fuerte alcista"],
        transition_matrix=[[0.2] * 5] * 5,
        state_mean_returns=[0.0] * 5,
        stationary_distribution=[0.2] * 5,
        forecast_5d_return=0.01,
        forecast_21d_return=0.03,
        forecast_21d_distribution=[0.1, 0.1, 0.1, 0.35, 0.35],
        prob_bullish_21d=prob_bullish_21d,
        runs_test_z=0.5,
        sequence_looks_random=sequence_looks_random,
        order2_justified=False,
        order2_p_value=0.5,
    )


def _garch(regime: str) -> GarchResult:
    return GarchResult(
        omega=0.00001,
        alpha=0.08,
        beta=0.88,
        persistence=0.96,
        unconditional_vol_annualized=0.25,
        current_vol_annualized=0.30,
        forecast_vol_21d_annualized=0.28,
        vol_percentile=0.9 if regime == "alta" else 0.4,
        regime=regime,
    )


def test_strong_bullish_setup_recommends_buy_with_stop_and_target():
    support = PriceLevel(price=97.0, kind="support", strength=2, distance_pct=-0.03)
    resistance = PriceLevel(price=110.0, kind="resistance", strength=1, distance_pct=0.10)

    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=Stage.STAGE_2,
        ma_cross="golden",
        rsi14=55.0,
        adx14=30.0,
        plus_di=25.0,
        minus_di=10.0,
        atr14=2.0,
        atr_multiple=1.0,
        rs_rating=90,
        minervini_pass=True,
        nearest_support=support,
        nearest_resistance=resistance,
    )

    assert rec.verdict == "comprar"
    assert rec.score >= re.BUY_THRESHOLD
    assert rec.stop_loss is not None
    assert rec.stop_loss < 100.0
    assert rec.take_profit == pytest.approx(110.0)
    assert rec.take_profit_method == "resistencia más cercana"
    assert rec.risk_reward > 0


def test_strong_bearish_setup_recommends_avoid_with_no_stop():
    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.DOWNTREND,
        stage=Stage.STAGE_4,
        ma_cross="death",
        rsi14=40.0,
        adx14=30.0,
        plus_di=10.0,
        minus_di=25.0,
        atr14=2.0,
        atr_multiple=1.0,
        rs_rating=15,
        minervini_pass=False,
        nearest_support=None,
        nearest_resistance=None,
    )

    assert rec.verdict == "evitar"
    assert rec.score <= re.AVOID_THRESHOLD
    assert rec.stop_loss is None
    assert rec.take_profit is None


def test_neutral_setup_recommends_wait():
    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.SIDEWAYS,
        stage=None,
        ma_cross=None,
        rsi14=50.0,
        adx14=15.0,
        plus_di=18.0,
        minus_di=17.0,
        atr14=2.0,
        atr_multiple=0.5,
        rs_rating=50,
        minervini_pass=False,
        nearest_support=None,
        nearest_resistance=None,
    )

    assert rec.verdict == "esperar"
    assert rec.stop_loss is None


def test_buy_falls_back_to_2to1_target_when_resistance_too_far():
    support = PriceLevel(price=98.0, kind="support", strength=1, distance_pct=-0.02)
    far_resistance = PriceLevel(price=180.0, kind="resistance", strength=1, distance_pct=0.80)

    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=Stage.STAGE_2,
        ma_cross="golden",
        rsi14=55.0,
        adx14=30.0,
        plus_di=25.0,
        minus_di=10.0,
        atr14=2.0,
        atr_multiple=1.0,
        rs_rating=95,
        minervini_pass=True,
        nearest_support=support,
        nearest_resistance=far_resistance,
    )

    assert rec.verdict == "comprar"
    assert rec.take_profit_method == "objetivo 2:1 sobre el riesgo"
    risk = 100.0 - rec.stop_loss
    assert rec.take_profit == pytest.approx(100.0 + 2 * risk)


def test_buy_falls_back_to_2to1_target_when_resistance_gives_bad_reward_risk():
    # Resistance is close enough to qualify by distance, but a stop 2.5x ATR away
    # means the reward it offers is worse than 1:1 - should not be used as the target.
    support = PriceLevel(price=98.0, kind="support", strength=1, distance_pct=-0.02)
    close_resistance = PriceLevel(price=102.0, kind="resistance", strength=1, distance_pct=0.02)

    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=Stage.STAGE_2,
        ma_cross="golden",
        rsi14=55.0,
        adx14=30.0,
        plus_di=25.0,
        minus_di=10.0,
        atr14=2.0,
        atr_multiple=1.0,
        rs_rating=95,
        minervini_pass=True,
        nearest_support=support,
        nearest_resistance=close_resistance,
    )

    assert rec.verdict == "comprar"
    assert rec.take_profit_method == "objetivo 2:1 sobre el riesgo"
    assert rec.risk_reward == pytest.approx(2.0)


def test_stop_loss_never_exceeds_the_atr_ceiling_even_with_a_distant_support():
    far_support = PriceLevel(price=70.0, kind="support", strength=1, distance_pct=-0.30)

    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=Stage.STAGE_2,
        ma_cross="golden",
        rsi14=55.0,
        adx14=30.0,
        plus_di=25.0,
        minus_di=10.0,
        atr14=2.0,
        atr_multiple=1.0,
        rs_rating=90,
        minervini_pass=True,
        nearest_support=far_support,
        nearest_resistance=None,
    )

    assert rec.verdict == "comprar"
    # ATR ceiling: price - 2.5*ATR14 = 100 - 5 = 95, which is tighter (higher) than
    # the far support's 70*0.99 - the recommendation must not take on that much risk.
    assert rec.stop_loss == pytest.approx(95.0)


def test_factors_report_which_conditions_actually_triggered():
    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=None,
        ma_cross=None,
        rsi14=None,
        adx14=None,
        plus_di=None,
        minus_di=None,
        atr14=None,
        atr_multiple=None,
        rs_rating=None,
        minervini_pass=False,
        nearest_support=None,
        nearest_resistance=None,
    )
    triggered_labels = {f.label for f in rec.factors if f.triggered}
    assert triggered_labels == {"Tendencia alcista (MA20 > MA50 > MA200)"}
    assert rec.score == 2


def _neutral_kwargs() -> dict:
    return dict(
        price=100.0,
        trend=TrendState.SIDEWAYS,
        stage=None,
        ma_cross=None,
        rsi14=50.0,
        adx14=15.0,
        plus_di=18.0,
        minus_di=17.0,
        atr14=2.0,
        atr_multiple=0.5,
        rs_rating=50,
        minervini_pass=False,
        nearest_support=None,
        nearest_resistance=None,
    )


def test_markov_bullish_forecast_adds_points_when_sequence_is_not_random():
    rec = re.build_recommendation(**_neutral_kwargs(), markov=_markov(0.65, sequence_looks_random=False))
    labels = {f.label for f in rec.factors if f.triggered}
    assert "Cadena de Markov: continuidad alcista probable (secuencia no aleatoria)" in labels
    assert rec.score == 2


def test_markov_bearish_forecast_subtracts_points_when_sequence_is_not_random():
    rec = re.build_recommendation(**_neutral_kwargs(), markov=_markov(0.20, sequence_looks_random=False))
    labels = {f.label for f in rec.factors if f.triggered}
    assert "Cadena de Markov: continuidad bajista probable (secuencia no aleatoria)" in labels
    assert rec.score == -2


def test_markov_forecast_ignored_when_sequence_looks_random():
    rec = re.build_recommendation(**_neutral_kwargs(), markov=_markov(0.65, sequence_looks_random=True))
    assert not any(f.triggered for f in rec.factors if "Markov" in f.label)
    assert rec.score == 0


def test_markov_neutral_probability_triggers_neither_factor():
    rec = re.build_recommendation(**_neutral_kwargs(), markov=_markov(0.50, sequence_looks_random=False))
    assert not any(f.triggered for f in rec.factors if "Markov" in f.label)
    assert rec.score == 0


def test_garch_high_vol_regime_subtracts_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), garch=_garch("alta"))
    labels = {f.label for f in rec.factors if f.triggered}
    assert "Volatilidad condicional elevada (GARCH, percentil ≥75 de su propio historial)" in labels
    assert rec.score == -1


def test_garch_normal_vol_regime_no_penalty():
    rec = re.build_recommendation(**_neutral_kwargs(), garch=_garch("normal"))
    assert not any(f.triggered for f in rec.factors if "GARCH" in f.label)
    assert rec.score == 0
