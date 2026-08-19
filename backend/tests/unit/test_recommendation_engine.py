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


# compute_stop_and_target is exercised indirectly by the build_recommendation
# tests below (only ever called there when verdict == "comprar"), and
# directly here - it's reused standalone by trade_plan_service.py to
# reconstruct a position's stop/target from its point-in-time entry data,
# independent of what today's checklist verdict says.


def test_compute_stop_and_target_none_without_atr():
    result = re.compute_stop_and_target(price=100.0, atr14=None, nearest_support=None, nearest_resistance=None)
    assert result == re.StopAndTarget(None, None, None, None)


def test_compute_stop_and_target_uses_atr_ceiling_without_a_nearby_support():
    result = re.compute_stop_and_target(price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None)
    assert result.stop_loss == pytest.approx(100.0 - re.ATR_STOP_MULTIPLE * 2.0)
    assert result.take_profit is not None
    assert result.take_profit_method == f"objetivo {re.REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"


def test_compute_stop_and_target_prefers_the_tighter_of_support_and_atr_ceiling():
    # Support (99*0.99=98.01) is tighter (higher) than the ATR ceiling
    # (100-2.5*2=95.0) - the tighter one wins, never risking more than
    # necessary just because the ATR ceiling would allow it.
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = re.compute_stop_and_target(price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None)
    assert result.stop_loss == pytest.approx(99.0 * 0.99)


def test_compute_stop_and_target_targets_resistance_when_reward_risk_clears_the_bar():
    support = PriceLevel(price=97.0, kind="support", strength=2, distance_pct=-0.03)
    resistance = PriceLevel(price=110.0, kind="resistance", strength=1, distance_pct=0.10)
    result = re.compute_stop_and_target(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=resistance
    )
    assert result.take_profit == pytest.approx(110.0)
    assert result.take_profit_method == "resistencia más cercana"


def test_compute_stop_and_target_none_take_profit_when_stop_is_at_or_above_price():
    # A support level *above* the entry price (e.g. a fast-moving entry that
    # already cleared it) makes the "tighter of the two" stop land at or
    # above price itself - no valid risk to size a target against.
    support = PriceLevel(price=102.0, kind="support", strength=1, distance_pct=0.02)
    result = re.compute_stop_and_target(price=100.0, atr14=0.001, nearest_support=support, nearest_resistance=None)
    assert result.stop_loss is not None
    assert result.take_profit is None
    assert result.risk_reward is None


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


def test_minervini_pass_now_contributes_a_smaller_confirmation_bonus():
    # Regression guard for the double-counting fix: Minervini alone (no other
    # trend/stage/RS factors triggered) should add exactly +1, not the old +2.
    rec = re.build_recommendation(**{**_neutral_kwargs(), "minervini_pass": True})
    assert rec.score == 1


def test_minervini_range_confirmed_scores_independently_of_the_8_of_8_gate():
    # The empirically-best-validated factor (see factor_ablation_study.py):
    # must score its own +2 even when the full Minervini 8/8 gate fails (e.g.
    # RS Rating below 70 or another unrelated criterion) - that's the whole
    # point of pulling it out of the AND-gate.
    rec = re.build_recommendation(
        **{**_neutral_kwargs(), "minervini_pass": False, "minervini_range_confirmed": True}
    )
    assert rec.score == 2
    labels = {f.label for f in rec.factors if f.triggered}
    assert any("Movimiento confirmado" in label for label in labels)


def test_minervini_range_confirmed_and_full_pass_both_contribute():
    rec = re.build_recommendation(
        **{**_neutral_kwargs(), "minervini_pass": True, "minervini_range_confirmed": True}
    )
    assert rec.score == 3  # +1 (8/8 bonus) + 2 (range-confirmed)


def test_minervini_range_confirmed_defaults_to_false():
    rec = re.build_recommendation(**_neutral_kwargs())
    assert rec.score == 0


def test_obv_bearish_divergence_subtracts_points():
    rec = re.build_recommendation(**_neutral_kwargs(), obv_divergence="bearish")
    labels = {f.label for f in rec.factors if f.triggered}
    assert any("Divergencia bajista de volumen" in label for label in labels)
    assert rec.score == -2


def test_obv_bullish_divergence_adds_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), obv_divergence="bullish")
    labels = {f.label for f in rec.factors if f.triggered}
    assert any("Divergencia alcista de volumen" in label for label in labels)
    assert rec.score == 1


def test_obv_divergence_none_triggers_no_volume_factor():
    rec = re.build_recommendation(**_neutral_kwargs(), obv_divergence=None)
    assert not any("volumen" in f.label.lower() for f in rec.factors if f.triggered)


def test_strong_revenue_growth_adds_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), revenue_growth=0.20)
    assert rec.score == 1


def test_contracting_revenue_subtracts_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), revenue_growth=-0.05)
    assert rec.score == -1


def test_healthy_profit_margin_adds_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), profit_margins=0.22)
    assert rec.score == 1


def test_negative_profit_margin_subtracts_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), profit_margins=-0.10)
    assert rec.score == -1


def test_high_leverage_subtracts_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), debt_to_equity=250.0)
    assert rec.score == -1


def test_moderate_leverage_no_penalty():
    rec = re.build_recommendation(**_neutral_kwargs(), debt_to_equity=80.0)
    assert rec.score == 0


def test_fundamentals_none_by_default_contributes_nothing():
    # Every fundamentals param defaults to None - a caller that doesn't pass
    # them (portfolio risk, premium watchlist) must see zero contribution,
    # not an error or a silent zero-as-bad-fundamentals penalty.
    rec = re.build_recommendation(**_neutral_kwargs())
    assert not any(
        label in {f.label for f in rec.factors if f.triggered}
        for label in [
            "Crecimiento de ingresos sólido (≥15% interanual)",
            "Ingresos en contracción (crecimiento interanual negativo)",
            "Margen neto saludable (≥15%)",
            "Empresa no rentable (margen neto negativo)",
            "Apalancamiento elevado (deuda/patrimonio > 200%)",
        ]
    )
    assert rec.score == 0


def test_all_new_factors_combine_additively_without_interfering():
    rec = re.build_recommendation(
        **_neutral_kwargs(),
        obv_divergence="bullish",
        revenue_growth=0.20,
        profit_margins=0.22,
        debt_to_equity=80.0,
    )
    assert rec.score == 3  # +1 obv, +1 growth, +1 margin, +0 leverage (below threshold)


def test_build_recommendation_has_no_market_regime_params():
    # Regression guard for a deliberate reversal, not an oversight: an earlier
    # version of this audit scored a benchmark-below-SMA200 / VIX-panic
    # penalty here. scripts/factor_ablation_study.py tested it and found the
    # *opposite* sign, significant after Benjamini-Hochberg correction at
    # both 21d and 126d horizons - so it was removed from scoring rather than
    # kept on the strength of the citation alone. See the module docstring
    # and docs/quant_methodology.md. market_regime_inputs()/vix_regime() in
    # technical_analysis.py still exist for MarketContextService and the
    # ablation study; this just asserts they're not silently reintroduced
    # here without a fresh evidence-based decision.
    import inspect

    params = inspect.signature(re.build_recommendation).parameters
    assert "market_trend" not in params
    assert "vix_regime" not in params


def test_overbought_not_penalized_inside_a_strong_confirmed_uptrend():
    # An external audit correctly flagged this exact conflict: penalizing RSI
    # overbought unconditionally fights the trend factors in a genuinely
    # strong, ADX-confirmed uptrend, where staying "overbought" for weeks is
    # normal and healthy, not a warning sign.
    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=None,
        ma_cross=None,
        rsi14=85.0,
        adx14=30.0,
        plus_di=25.0,
        minus_di=10.0,
        atr14=2.0,
        atr_multiple=0.5,
        rs_rating=None,
        minervini_pass=False,
        nearest_support=None,
        nearest_resistance=None,
    )
    assert not any("Sobrecompra" in f.label for f in rec.factors if f.triggered)


def test_overbought_still_penalized_outside_a_strong_trend():
    rec = re.build_recommendation(**{**_neutral_kwargs(), "rsi14": 85.0})
    labels = {f.label for f in rec.factors if f.triggered}
    assert any("Sobrecompra" in label for label in labels)
    assert rec.score == -1


def test_mean_reverting_structure_subtracts_a_point():
    rec = re.build_recommendation(**_neutral_kwargs(), mean_reverting_structure=True)
    assert rec.score == -1
    labels = {f.label for f in rec.factors if f.triggered}
    assert any("reversión a la media" in label for label in labels)


def test_mean_reverting_structure_false_by_default():
    rec = re.build_recommendation(**_neutral_kwargs())
    assert rec.score == 0


def test_overbought_still_penalized_in_a_weak_uptrend_without_strong_adx():
    # Uptrend by MA alignment, but ADX doesn't confirm strength - RSI 85 here
    # is still flagged, unlike the strong-trend case above.
    rec = re.build_recommendation(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=None,
        ma_cross=None,
        rsi14=85.0,
        adx14=15.0,
        plus_di=18.0,
        minus_di=17.0,
        atr14=2.0,
        atr_multiple=0.5,
        rs_rating=None,
        minervini_pass=False,
        nearest_support=None,
        nearest_resistance=None,
    )
    assert any("Sobrecompra" in f.label for f in rec.factors if f.triggered)
