import numpy as np
import pandas as pd
import pytest

from app.services import technical_analysis as ta
from app.services import walk_forward_backtest as wf


def _synthetic_regime_series(n: int = 6400, block: int = 400, seed: int = 123) -> pd.Series:
    """Alternating long uptrend/downtrend regimes, each long enough (relative
    to the 200-day SMA's memory) for `classify_trend` to cleanly read the
    regime it's actually in - by construction, "comprar"-favorable conditions
    should correlate with genuinely higher forward returns and vice versa."""
    rng = np.random.default_rng(seed)
    returns = []
    n_blocks = n // block + 1
    for k in range(n_blocks):
        drift = 0.0025 if k % 2 == 0 else -0.0025
        returns.extend(rng.normal(drift, 0.008, block))
    prices = 100 * np.cumprod(1 + np.array(returns[:n]))
    return pd.Series(prices)


def _indicator_bundle(close: pd.Series) -> dict:
    high = close * 1.005
    low = close * 0.995
    return {
        "close": close,
        "sma20": ta.sma(close, 20),
        "sma50": ta.sma(close, 50),
        "sma150": ta.sma(close, 150),
        "sma200": ta.sma(close, 200),
        "rsi14": ta.rsi(close),
        "adx14": ta.adx(high, low, close),
        "plus_di": ta.dmi(high, low, close)[0],
        "minus_di": ta.dmi(high, low, close)[1],
        "atr14": ta.atr(high, low, close),
    }


def test_replay_verdict_none_before_smas_are_valid():
    close = pd.Series(np.linspace(100, 110, 50))
    bundle = _indicator_bundle(close)
    verdict = wf._replay_verdict_at(10, **bundle)
    assert verdict is None


def test_replay_verdict_works_without_volume_backward_compatible():
    # volume is optional (defaults to None) precisely so existing callers/tests
    # that don't pass it keep working unchanged.
    close = _synthetic_regime_series(n=400)
    bundle = _indicator_bundle(close)
    verdict = wf._replay_verdict_at(300, **bundle)
    assert verdict in ("comprar", "esperar", "evitar")


def test_replay_verdict_incorporates_minervini_range_confirmation():
    # A steady climb from a clear base gives a bar deep in "confirmed range"
    # territory (>=25% above its 52w low, within 25% of its 52w high) well
    # before the series ends - this must not crash and must still return a
    # valid verdict once the range-confirmation replay logic is wired in.
    close = _synthetic_regime_series(n=800, block=800)  # one long uptrend block
    bundle = _indicator_bundle(close)
    verdict = wf._replay_verdict_at(700, **bundle)
    assert verdict in ("comprar", "esperar", "evitar")


def test_replay_recommendation_at_returns_none_before_smas_are_valid():
    close = pd.Series(np.linspace(100, 110, 50))
    bundle = _indicator_bundle(close)
    assert wf.replay_recommendation_at(10, **bundle) is None


def test_replay_recommendation_at_carries_a_stop_loss_for_a_comprar_verdict():
    # D7: the whole point of this refactor - a replayed "comprar" verdict
    # must carry the stop_loss/take_profit build_recommendation actually
    # proposed at that point, not just the bare verdict string.
    close = _synthetic_regime_series(n=800, block=800)  # one long uptrend block
    bundle = _indicator_bundle(close)
    rec = wf.replay_recommendation_at(700, **bundle)
    assert rec is not None
    if rec.verdict == "comprar":
        assert rec.stop_loss is not None
        assert rec.stop_loss < close.iloc[700]
        assert rec.take_profit is not None


def test_replay_recommendation_at_matches_replay_verdict_at():
    # _replay_verdict_at is now a thin wrapper - confirms it stays consistent
    # with the fuller function it delegates to.
    close = _synthetic_regime_series(n=400)
    bundle = _indicator_bundle(close)
    rec = wf.replay_recommendation_at(300, **bundle)
    verdict = wf._replay_verdict_at(300, **bundle)
    assert rec is not None
    assert rec.verdict == verdict


def test_replay_verdict_incorporates_obv_divergence_when_volume_given():
    close = _synthetic_regime_series(n=400)
    bundle = _indicator_bundle(close)
    flat_volume = pd.Series([1000.0] * len(close))
    # Same price path, only volume differs - a real point-in-time OBV
    # divergence read requires actual volume data, not a placeholder.
    with_volume = wf._replay_verdict_at(300, **bundle, volume=flat_volume)
    without_volume = wf._replay_verdict_at(300, **bundle)
    # A perfectly flat volume series produces no divergence signal either way
    # (OBV moves in lockstep with price direction, same relative position) -
    # this asserts the plumbing runs without error and returns a valid verdict,
    # not a specific different outcome.
    assert with_volume in ("comprar", "esperar", "evitar")
    assert without_volume in ("comprar", "esperar", "evitar")


def test_run_walk_forward_backtest_none_for_insufficient_history():
    close = pd.Series(np.linspace(100, 110, 100))
    bundle = _indicator_bundle(close)
    assert wf.run_walk_forward_backtest(**bundle) is None


def test_run_walk_forward_backtest_detects_genuine_edge_on_synthetic_regime_data():
    close = _synthetic_regime_series()
    bundle = _indicator_bundle(close)
    result = wf.run_walk_forward_backtest(**bundle, horizon_days=21)

    assert result is not None
    assert result.n_samples >= wf.MIN_BUCKET_SIZE

    comprar = next(b for b in result.bucket_stats if b.verdict == "comprar")
    evitar = next(b for b in result.bucket_stats if b.verdict == "evitar")
    assert comprar.n > 0
    assert evitar.n > 0
    assert comprar.mean_return > evitar.mean_return

    test = next(t for t in result.significance_tests if t.comparison == "comprar vs evitar")
    assert test.significant_at_5pct
    assert test.mean_difference > 0
    assert "insuficiente" not in result.interpretation


def test_bucket_stats_counts_and_rates_are_consistent():
    df = pd.DataFrame(
        {
            "verdict": ["comprar"] * 10 + ["evitar"] * 10,
            "fwd_return": [0.05] * 8 + [-0.02] * 2 + [-0.03] * 9 + [0.01] * 1,
        }
    )
    stats = wf._bucket_stats(df)
    comprar = next(b for b in stats if b.verdict == "comprar")
    evitar = next(b for b in stats if b.verdict == "evitar")
    assert comprar.n == 10
    assert comprar.win_rate == pytest.approx(0.8)
    assert evitar.n == 10
    assert evitar.win_rate == pytest.approx(0.1)


def test_significance_tests_skipped_when_bucket_too_small():
    df = pd.DataFrame(
        {
            "verdict": ["comprar"] * 3 + ["evitar"] * 20,
            "fwd_return": list(np.linspace(0.01, 0.05, 3)) + list(np.linspace(-0.05, -0.01, 20)),
        }
    )
    tests = wf._significance_tests(df)
    assert all(t.comparison != "comprar vs evitar" for t in tests)


def test_permutation_test_high_p_value_for_identical_distributions():
    rng = np.random.default_rng(1)
    sample_a = rng.normal(0, 0.02, 40)
    sample_b = rng.normal(0, 0.02, 40)
    p = wf._permutation_test(sample_a, sample_b, n_permutations=2000, seed=1)
    assert p > 0.05


def test_permutation_test_low_p_value_for_clearly_different_means():
    rng = np.random.default_rng(2)
    sample_a = rng.normal(0.05, 0.01, 40)
    sample_b = rng.normal(-0.05, 0.01, 40)
    p = wf._permutation_test(sample_a, sample_b, n_permutations=2000, seed=2)
    assert p < 0.01


def test_interpret_flags_insufficient_data_when_bucket_empty():
    bucket_stats = [
        wf.VerdictBucketStats("comprar", 0, None, None, None),
        wf.VerdictBucketStats("esperar", 20, 0.5, 0.01, 0.01),
        wf.VerdictBucketStats("evitar", 0, None, None, None),
    ]
    message = wf._interpret(bucket_stats, [])
    assert "insuficiente" in message


def test_interpret_warns_when_evitar_outperforms_comprar():
    bucket_stats = [
        wf.VerdictBucketStats("comprar", 20, 0.4, -0.01, -0.01),
        wf.VerdictBucketStats("esperar", 20, 0.5, 0.0, 0.0),
        wf.VerdictBucketStats("evitar", 20, 0.6, 0.03, 0.03),
    ]
    tests = [
        wf.SignificanceTest(
            comparison="comprar vs evitar",
            mean_difference=-0.04,
            t_stat=-3.0,
            p_value=0.001,
            p_value_bonferroni=0.003,
            permutation_p_value=0.002,
            significant_at_5pct=True,
        )
    ]
    message = wf._interpret(bucket_stats, tests)
    assert "Advertencia" in message
