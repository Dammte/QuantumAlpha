"""Walk-forward backtest of the recommendation engine's own scoring logic,
replayed point-in-time against this specific ticker's history, to answer the
one question a rule-based system can't answer about itself: has "comprar"
actually outperformed "evitar" on THIS ticker historically, or is the recipe
just plausible-sounding rules with no real edge here? This is the main
mechanism for keeping the overall tool objective rather than a confirmation
machine - it is willing to report that its own bullish factors have NOT
historically paid off on a given name.

Every indicator series involved (SMAs, RSI, ADX/DMI, ATR) is already computed
causally by `ticker_analysis_service` - pandas `.rolling`/`.ewm` never look
ahead, so indexing into the precomputed series at bar `i` is exactly what was
knowable on that day, with no lookahead bias and no need to recompute anything
per step. Two point-in-time-unavailable factors from the live recommendation
engine are deliberately left out of the replay: RS Rating and the Minervini
RS>=70 sub-criterion (both need a cross-sectional universe snapshot that isn't
available at arbitrary past dates), and nearest support/resistance (its pivot
scan is O(n) per call and re-running it at every historical bar is
prohibitively expensive). The replayed score is therefore a conservative,
not identical, proxy for the live recommendation - but it reuses
`build_recommendation` itself (same thresholds, same remaining factors), so
it is a faithful test of the trend/momentum core of the system rather than a
separate, hand-tuned scoring function.

Samples are taken on a NON-OVERLAPPING grid (stride == forecast horizon)
specifically to avoid the classic overlapping-window autocorrelation trap:
sampling every day and looking `h` days ahead produces heavy pseudo-
replication (adjacent samples share almost all of their return path), which
inflates apparent statistical significance. Non-overlapping sampling costs
sample size but keeps the significance tests honest. Both a Welch's t-test
(parametric) and a permutation test (nonparametric, no distributional
assumptions) are reported for each pairwise verdict comparison, with a
Bonferroni correction across the three possible pairwise comparisons.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from app.services.recommendation_engine import build_recommendation
from app.services.technical_analysis import (
    classify_stage,
    classify_trend,
    detect_recent_cross,
    obv_divergence,
    rolling_extreme_price,
)

MIN_OBSERVATIONS = 300
WARMUP_BARS = 260
MIN_BUCKET_SIZE = 8
N_PERMUTATIONS = 5000
VERDICTS = ("comprar", "esperar", "evitar")


def _replay_verdict_at(
    i: int,
    close: pd.Series,
    sma20: pd.Series,
    sma50: pd.Series,
    sma150: pd.Series,
    sma200: pd.Series,
    rsi14: pd.Series,
    adx14: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    atr14: pd.Series,
    volume: pd.Series | None = None,
) -> str | None:
    """Rebuilds the recommendation verdict using only values knowable at bar
    `i` - either a scalar reading at `i`, or (for the functions that need a
    lookback window - `classify_stage`, `detect_recent_cross`, `obv_divergence`)
    a slice `[:i+1]`, still using no information beyond bar `i`."""
    price = close.iloc[i]
    s20, s50, s200 = sma20.iloc[i], sma50.iloc[i], sma200.iloc[i]
    if pd.isna(s20) or pd.isna(s50) or pd.isna(s200):
        return None

    trend = classify_trend(price, s20, s50, s200)
    s150 = sma150.iloc[i]
    stage = classify_stage(price, sma150.iloc[: i + 1]) if not pd.isna(s150) else None
    ma_cross = detect_recent_cross(sma50.iloc[: i + 1], sma200.iloc[: i + 1])
    obv_div = obv_divergence(close.iloc[: i + 1], volume.iloc[: i + 1]) if volume is not None else None

    # Unlike the other 7 Minervini criteria (excluded here - they need a
    # point-in-time RS Rating this replay doesn't have), the 52-week-range
    # criteria are pure price history and fully causal to compute at any bar.
    # See recommendation_engine.py's comment on `minervini_range_confirmed`
    # for why this one is scored independently of the 8/8 gate.
    price_52w_low = rolling_extreme_price(close.iloc[: i + 1], 252, "low")
    price_52w_high = rolling_extreme_price(close.iloc[: i + 1], 252, "high")
    range_confirmed = (
        price_52w_low is not None
        and price_52w_low > 0
        and price >= price_52w_low * 1.25
        and price_52w_high is not None
        and price_52w_high > 0
        and price >= price_52w_high * 0.75
    )

    atr_t = atr14.iloc[i]
    has_atr = not pd.isna(atr_t) and atr_t != 0
    atr_multiple = float((price - s50) / atr_t) if has_atr else None

    rec = build_recommendation(
        price=price,
        trend=trend,
        stage=stage,
        ma_cross=ma_cross,
        rsi14=None if pd.isna(rsi14.iloc[i]) else float(rsi14.iloc[i]),
        adx14=None if pd.isna(adx14.iloc[i]) else float(adx14.iloc[i]),
        plus_di=None if pd.isna(plus_di.iloc[i]) else float(plus_di.iloc[i]),
        minus_di=None if pd.isna(minus_di.iloc[i]) else float(minus_di.iloc[i]),
        atr14=float(atr_t) if has_atr else None,
        atr_multiple=atr_multiple,
        rs_rating=None,
        minervini_pass=False,
        nearest_support=None,
        nearest_resistance=None,
        minervini_range_confirmed=range_confirmed,
        obv_divergence=obv_div,
    )
    return rec.verdict


def _permutation_test(
    sample_a: np.ndarray, sample_b: np.ndarray, n_permutations: int = N_PERMUTATIONS, seed: int = 0
) -> float:
    """Empirical two-sided p-value for the difference in means, under the null
    that verdict labels carry no information (shuffled relative to the
    pooled returns). Doesn't assume normality, unlike the t-test."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([sample_a, sample_b])
    n_a = len(sample_a)
    observed = float(sample_a.mean() - sample_b.mean())

    shuffle_idx = np.argsort(rng.random((n_permutations, len(pooled))), axis=1)
    permuted = pooled[shuffle_idx]
    diffs = permuted[:, :n_a].mean(axis=1) - permuted[:, n_a:].mean(axis=1)
    count = int((np.abs(diffs) >= abs(observed)).sum())
    return (count + 1) / (n_permutations + 1)


@dataclass(frozen=True, slots=True)
class VerdictBucketStats:
    verdict: str
    n: int
    win_rate: float | None
    mean_return: float | None
    median_return: float | None


@dataclass(frozen=True, slots=True)
class SignificanceTest:
    comparison: str
    mean_difference: float
    t_stat: float
    p_value: float
    p_value_bonferroni: float
    permutation_p_value: float
    significant_at_5pct: bool


@dataclass(frozen=True, slots=True)
class WalkForwardBacktestResult:
    horizon_days: int
    n_samples: int
    bucket_stats: list[VerdictBucketStats]
    significance_tests: list[SignificanceTest]
    overall_mean_return: float
    interpretation: str


def _bucket_stats(df: pd.DataFrame) -> list[VerdictBucketStats]:
    stats_by_verdict = []
    for verdict in VERDICTS:
        subset = df.loc[df.verdict == verdict, "fwd_return"]
        if len(subset) == 0:
            stats_by_verdict.append(VerdictBucketStats(verdict, 0, None, None, None))
        else:
            stats_by_verdict.append(
                VerdictBucketStats(
                    verdict=verdict,
                    n=len(subset),
                    win_rate=float((subset > 0).mean()),
                    mean_return=float(subset.mean()),
                    median_return=float(subset.median()),
                )
            )
    return stats_by_verdict


def _significance_tests(df: pd.DataFrame) -> list[SignificanceTest]:
    pairs = list(combinations(VERDICTS, 2))
    tests = []
    for a, b in pairs:
        sample_a = df.loc[df.verdict == a, "fwd_return"].to_numpy()
        sample_b = df.loc[df.verdict == b, "fwd_return"].to_numpy()
        if len(sample_a) < MIN_BUCKET_SIZE or len(sample_b) < MIN_BUCKET_SIZE:
            continue
        t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        perm_p = _permutation_test(sample_a, sample_b)
        p_bonf = min(float(p_value) * len(pairs), 1.0)
        tests.append(
            SignificanceTest(
                comparison=f"{a} vs {b}",
                mean_difference=float(sample_a.mean() - sample_b.mean()),
                t_stat=float(t_stat),
                p_value=float(p_value),
                p_value_bonferroni=p_bonf,
                permutation_p_value=perm_p,
                significant_at_5pct=bool(p_bonf < 0.05),
            )
        )
    return tests


def _interpret(bucket_stats: list[VerdictBucketStats], tests: list[SignificanceTest]) -> str:
    comprar = next((b for b in bucket_stats if b.verdict == "comprar"), None)
    evitar = next((b for b in bucket_stats if b.verdict == "evitar"), None)
    test = next((t for t in tests if t.comparison == "comprar vs evitar"), None)

    if test is None or comprar is None or evitar is None or not comprar.n or not evitar.n:
        return (
            "Historial insuficiente de señales 'comprar' y 'evitar' en este activo para "
            "validar el sistema estadísticamente - interpretar el veredicto actual con cautela."
        )

    if test.significant_at_5pct and test.mean_difference > 0:
        return (
            f"El sistema muestra ventaja estadísticamente significativa en este activo: las señales "
            f"'comprar' ({comprar.n} casos, retorno medio {comprar.mean_return:+.1%}) superaron a las "
            f"'evitar' ({evitar.n} casos, retorno medio {evitar.mean_return:+.1%}), diferencia "
            f"{test.mean_difference:+.1%} (p={test.p_value_bonferroni:.3f})."
        )
    if test.significant_at_5pct and test.mean_difference <= 0:
        return (
            f"Advertencia: en el historial de este activo, las señales 'evitar' rindieron mejor que las "
            f"'comprar' ({evitar.mean_return:+.1%} vs {comprar.mean_return:+.1%}, p="
            f"{test.p_value_bonferroni:.3f}) - el sistema no muestra ventaja aquí; tratar el veredicto "
            f"actual con escepticismo en este activo específico."
        )
    return (
        f"Sin diferencia estadísticamente significativa entre 'comprar' ({comprar.mean_return:+.1%}) y "
        f"'evitar' ({evitar.mean_return:+.1%}) en el historial de este activo (p="
        f"{test.p_value_bonferroni:.3f}) - la ventaja del sistema en este activo específico no está "
        f"confirmada; combinar con las demás señales (Markov, Monte Carlo) en vez de apoyarse solo en esta."
    )


def run_walk_forward_backtest(
    close: pd.Series,
    sma20: pd.Series,
    sma50: pd.Series,
    sma150: pd.Series,
    sma200: pd.Series,
    rsi14: pd.Series,
    adx14: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    atr14: pd.Series,
    horizon_days: int = 21,
    volume: pd.Series | None = None,
) -> WalkForwardBacktestResult | None:
    n = len(close)
    if n < MIN_OBSERVATIONS:
        return None

    last_valid_start = n - horizon_days
    if last_valid_start <= WARMUP_BARS:
        return None

    indices = range(WARMUP_BARS, last_valid_start, horizon_days)
    records = []
    for i in indices:
        verdict = _replay_verdict_at(
            i, close, sma20, sma50, sma150, sma200, rsi14, adx14, plus_di, minus_di, atr14, volume
        )
        if verdict is None:
            continue
        fwd_return = float(close.iloc[i + horizon_days] / close.iloc[i] - 1)
        records.append((verdict, fwd_return))

    if len(records) < MIN_BUCKET_SIZE:
        return None

    df = pd.DataFrame(records, columns=["verdict", "fwd_return"])
    bucket_stats = _bucket_stats(df)
    tests = _significance_tests(df)

    return WalkForwardBacktestResult(
        horizon_days=horizon_days,
        n_samples=len(df),
        bucket_stats=bucket_stats,
        significance_tests=tests,
        overall_mean_return=float(df.fwd_return.mean()),
        interpretation=_interpret(bucket_stats, tests),
    )
