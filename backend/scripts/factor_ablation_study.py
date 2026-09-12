"""Cross-sectional factor ablation study: does each individual recommendation-
engine factor actually correlate with forward returns, measured independently
AND jointly, pooled across the *entire* curated universe (US + Europe, ~217
tickers) - not just one ticker at a time.

Why this exists (see `recommendation_engine.py`'s module docstring): the
existing `walk_forward_backtest.py`/`backtest_engine.py` validate the
*combined* verdict on one ticker at a time, which answers "does the system
work on THIS stock" but not "which specific factor is actually pulling its
weight, and which is dead weight (or worse, wrong-signed)". This script
answers that, pooling across ~217 tickers instead of one gives roughly
20,000+ samples instead of ~100-150 per ticker.

**Fase 5 rewrite (agosto 2026) - four real methodological gaps the first
version had, each addressed directly:**

1. **Triple-barrier labeling, not a fixed-horizon return.** The original
   `fwd_return = close[i+h]/close[i] - 1` had exactly the same D7 problem
   `walk_forward_backtest.py` did - buy-and-hold to a fixed horizon, ignoring
   any stop/target. Every sample here is now labeled with
   `backtest_engine.label_triple_barrier` (trailing Chandelier stop
   included), sized with `recommendation_engine.py`'s own
   `ATR_STOP_MULTIPLE`/`REWARD_RISK_RATIO` constants - the same discipline
   the live system actually trades under, applied uniformly to every sample
   regardless of which factor is being tested (a factor can't have its own
   bespoke stop/target; that would make cross-factor comparison meaningless).
2. **Cross-sectional demeaning.** Every return is expressed relative to its
   own calendar-month bucket's mean return across the whole sampled universe
   before any statistic is computed - this is *why* `vix_stress` measured
   +4.35pp in the very first study (docs/quant_methodology.md §6.1): that
   was market beta in a decade dominated by a bull market, not a factor
   edge. Demeaning strips exactly that out.
3. **Multivariate regression alongside the univariate test.** Trend/stage/RS
   Rating/Minervini all measure some version of "is this in a confirmed
   uptrend" (recommendation_engine.py's own docstring says so) - a
   univariate test on collinear factors credits the same underlying fact
   several times over. An OLS regression with every factor as a
   simultaneous predictor (`run_multivariate_regression`) answers "does this
   factor still matter once the others are already accounted for".
4. **Information Coefficient.** `compute_information_coefficient` reports
   the industry-standard factor metric - the Spearman rank correlation
   between a factor's trigger and demeaned forward returns, computed
   separately per calendar-month bucket and then averaged (mean IC) with its
   own dispersion (IC / std(IC) = "IC IR") - more informative than one
   pooled significance test because it shows whether an edge is consistent
   bucket to bucket or driven by one or two lucky periods.

Also new: **regime segmentation** (`segment_by_regime`) - the same pooled
univariate test, rerun separately within "market above/below its own
SMA200" and "VIX calm/stress" subsets, because a factor can work in one
regime and fail in another; a single pooled average hides both. And the
**default horizons are now 5/10/21 sessions** (this portfolio's actual
holding horizon), not 21 alone - 63/126 are still available via `--horizons`
for anyone checking horizon sensitivity against the momentum literature
(Jegadeesh & Titman 1993), same as before.

**Fase 8 reorientation (reconstrucción, septiembre 2026) - measuring the new
gate instead of the retired checklist:** `recommendation_engine.py`'s
checklist no longer decides anything live (`levels_engine.evaluate_gate`
does - docs/quant_methodology.md §25), so the factors worth measuring here
changed too. `compute_triggers_at` now also calls
`levels_engine.replay_gate_at` - the exact same point-in-time replay
function the live "Analizar activo" backtest uses, not a second hand-rolled
approximation of the gate - and exposes five of its six conditions as their
own factors (`gate_trend_or_stage2`, `gate_not_parabolic`,
`gate_not_overbought_outside_strong_trend`, `gate_no_obv_bearish_divergence`,
`gate_no_fast_pair_veto`), plus the compound `gate_passes`. Three of the old
checklist's factors were *removed*, not kept alongside their gate
equivalent: `atr_parabolic`, `rsi_overbought_outside_strong_trend`, and
`obv_bearish` are each the exact logical negation of a gate factor above
(same predicate, opposite boolean) - keeping both would hand
`run_multivariate_regression` two perfectly collinear columns, which is a
real defect (a singular design matrix), not just redundant reporting. The
sixth gate condition (reward:risk >= 1.5) is deliberately **not** exposed as
a factor here: `replay_gate_at` has no point-in-time support/resistance
(its own documented simplification), so `compute_stop_and_target` always
falls back to the fixed ATR-stop/2:1-target formula, making that condition
a constant `True` for every sample - a zero-variance column is collinear
with the regression's own intercept, which would corrupt every other
factor's coefficient in the same fit, not just that one's. That condition
is instead measured properly, against real daily precomputed
support/resistance, by `trigger_performance_service.py`
(`GET /system/signal-performance`) - see that module's own docstring.
`golden_cross`/`death_cross`/`rsi_oversold_bounce`/`minervini_range_position`/
`trend_down`/`stage4` are informational signals the live system still
surfaces (imminent-cross badges, context) even though none of them gate
anything - kept as-is, still worth knowing whether they correlate with
anything real.

**What this script deliberately does NOT do**: it does not change any
weight in `recommendation_engine.py`, and it does not implement D9's
cross-sectional percentile threshold (that's a live-system, request-time
change - "what percentile is today's universe in" needs infrastructure this
offline script doesn't have, and touching it is exactly the kind of
recalibration decision that needs a human reviewing real, freshly-run
output first, not an automated action). Running this script produces
evidence; acting on it is a separate, deliberate step - see "cómo usar los
resultados" below.

**Cómo usar los resultados** (antes de cambiar un solo peso en
`recommendation_engine.py`):
- Un factor que no supera significación BH-ajustada fuera de muestra no
  cambia de peso a partir de un único resultado dentro de muestra.
- No inviertas el signo de un factor con décadas de literatura detrás salvo
  evidencia fuerte y consistente en varios regímenes (ver la segmentación
  por régimen antes de decidir esto).
- Los pesos deben quedar en enteros pequeños y redondos - una sugerencia de
  +1,73 se convierte en +2, nunca en +1,73.
- Compara el coeficiente multivariante con el univariante: si un factor
  pierde toda su significación en la regresión conjunta, es colinealidad con
  otro factor ya puntuado, no una señal nueva independiente.

Usage:
    python scripts/factor_ablation_study.py [--horizons 5 10 21] [--regions us europe]

Takes several minutes (mostly yfinance download time for ~217 tickers x 10
years of daily bars) - this is an offline research/calibration script, not
something that runs as part of the API.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.db.repositories.universe_membership_repository import (  # noqa: E402
    UniverseMembershipRepository,
)
from app.infrastructure.db.session import SessionLocal  # noqa: E402
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider  # noqa: E402
from app.services import backtest_engine as be  # noqa: E402
from app.services import dynamic_universe_service as dus  # noqa: E402
from app.services import levels_engine as le  # noqa: E402
from app.services import technical_analysis as ta  # noqa: E402
from app.services import watchlist_service as wl  # noqa: E402
from app.services.market_data_service import MarketDataService  # noqa: E402
from app.services.market_universe import VIX_TICKER, benchmark_for_ticker, universe_tickers  # noqa: E402
from app.services.recommendation_engine import ATR_STOP_MULTIPLE, REWARD_RISK_RATIO  # noqa: E402

WARMUP_BARS = 260  # enough for SMA200 + its 25-bar slope lookback
MIN_BARS_REQUIRED = WARMUP_BARS + 100
HISTORY_YEARS = 10
N_PERMUTATIONS = 5000
DEFAULT_HORIZONS = (5, 10, 21)  # this portfolio's real holding horizon - see docs/quant_methodology.md Fase 5
MIN_GROUP_SIZE = 30  # per side of a comparison, before trusting a statistic at all
MIN_BUCKET_SIZE_FOR_IC = 5  # per calendar-month bucket, before that bucket contributes to a factor's IC

# Segunda auditoría, Bloque 5: calibrate 2016-2022 / validate 2023-2026 - the
# brief's own explicit split. Never mixed: a factor's calibrate-period result
# is reported separately from its validate-period one, never pooled into a
# single number.
TEMPORAL_SPLIT_CUTOFF = pd.Timestamp("2023-01-01")


@dataclass(frozen=True, slots=True)
class FactorSample:
    ticker: str
    date: pd.Timestamp
    fwd_return: float  # raw triple-barrier return, pre-demeaning
    demeaned_return: float  # fwd_return minus its own calendar-month bucket's cross-sectional mean
    triggers: dict[str, bool]
    # Tercera auditoría, Bloque F-6: backtest_engine.TripleBarrierLabel's own
    # per-sample fields, propagated instead of discarded - segment_by_setup_type
    # partitions samples by setup, but until now every statistic computed
    # within a segment was a recommendation_engine *factor* stat (mean_difference,
    # IC, ...), never the setup's own realized outcome ("how much do I
    # typically make with this setup" - see compute_setup_outcome_stats).
    exit_reason: str  # "stop" | "target" | "vertical" -> win rate
    bars_held: int  # -> median holding period
    mae_pct: float  # Maximum Adverse Excursion, <=0 -> where a stop routinely gets tested
    mfe_pct: float  # Maximum Favorable Excursion, >=0
    risk_pct: float  # (entry_price - stop) / entry_price at this sample's own entry - "1R" in % terms,
    # so fwd_return/risk_pct is this sample's own return expressed in R multiples (expectancy_r's input)


@dataclass(frozen=True, slots=True)
class FactorResult:
    factor: str
    current_points: int
    n_triggered: int
    n_not_triggered: int
    mean_return_triggered: float
    mean_return_not_triggered: float
    mean_difference: float
    t_stat: float
    p_value: float
    permutation_p_value: float
    permutation_p_value_bh: float  # Benjamini-Hochberg-adjusted (false discovery rate)
    significant_at_1pct: bool  # raw p-value - kept for comparison, don't trust alone
    significant_at_1pct_bh: bool  # BH-adjusted - the one that accounts for testing many factors at once
    directionally_consistent: bool  # does the sign of the measured effect match the factor's current point sign?
    mean_ic: float | None  # mean Spearman IC across calendar-month buckets
    ic_ir: float | None  # mean_ic / std(ic) across buckets - None if fewer than 2 buckets qualified
    n_ic_buckets: int
    multivariate_coef_pct: float | None  # OLS coefficient (all factors simultaneously), in percentage points
    multivariate_p_value: float | None


# Mirrors the point values in recommendation_engine.py at the time this was
# run, purely for the "directionally consistent" sanity check below - not
# imported directly since several of these factors (trend_down, stage4, etc.)
# don't have a single clean boolean predicate exposed by the engine itself.
# 2026-09 (Fase 8): the gate_* entries have no real point value at all (the
# gate is a boolean AND, not a weighted score) - +1 here is a nominal stand-in
# for "this condition was designed to be a bullish/protective signal", the
# same sign every condition in evaluate_gate() is framed with, just so
# `directionally_consistent` has something to compare the measured sign
# against. `atr_parabolic`/`rsi_overbought_outside_strong_trend`/
# `obv_bearish` are gone - see the module docstring's Fase 8 section for why
# (each is the exact negation of a gate_* factor below; keeping both would be
# a real collinearity defect, not just redundant reporting).
CURRENT_POINTS = {
    "trend_up": 2,
    "trend_down": -3,
    "stage2": 2,
    "stage4": -3,
    "golden_cross": 1,
    "death_cross": -2,
    "adx_strong_trend": 1,
    "rsi_oversold_bounce": 1,
    "obv_bullish": 1,
    "minervini_range_position": 1,  # the non-RS-dependent half of the +1 confirmation bonus
    "market_below_sma200": -2,  # not scored live (§6.1) - kept as a regime segmentation key, not a live factor
    "vix_stress": -2,  # same - regime segmentation key, not a live factor
    "gate_passes": 1,
    "gate_trend_or_stage2": 1,
    "gate_not_parabolic": 1,
    "gate_not_overbought_outside_strong_trend": 1,
    "gate_no_obv_bearish_divergence": 1,
    "gate_no_fast_pair_veto": 1,
}


def compute_triggers_at(
    i: int,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    sma20: pd.Series,
    sma50: pd.Series,
    sma150: pd.Series,
    sma200: pd.Series,
    rsi14: pd.Series,
    adx14: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    atr14: pd.Series,
    benchmark_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
) -> dict[str, bool] | None:
    price = close.iloc[i]
    s20, s50, s200 = sma20.iloc[i], sma50.iloc[i], sma200.iloc[i]
    if pd.isna(s20) or pd.isna(s50) or pd.isna(s200):
        return None

    trend = ta.classify_trend(price, s20, s50, s200)
    s150 = sma150.iloc[i]
    stage = ta.classify_stage(price, sma150.iloc[: i + 1]) if not pd.isna(s150) else None
    ma_cross = ta.detect_recent_cross(sma50.iloc[: i + 1], sma200.iloc[: i + 1])
    obv_div = ta.obv_divergence(close.iloc[: i + 1], volume.iloc[: i + 1])

    adx_t, plus_t, minus_t = adx14.iloc[i], plus_di.iloc[i], minus_di.iloc[i]
    strong_trend = (
        not pd.isna(adx_t) and adx_t >= 25 and not pd.isna(plus_t) and not pd.isna(minus_t) and plus_t > minus_t
    )

    rsi_t = rsi14.iloc[i]
    rsi_oversold_bounce = not pd.isna(rsi_t) and rsi_t <= 30 and trend != ta.TrendState.DOWNTREND

    price_52w_low = ta.rolling_extreme_price(close.iloc[: i + 1], 252, "low")
    price_52w_high = ta.rolling_extreme_price(close.iloc[: i + 1], 252, "high")
    minervini_range_position = (
        price_52w_low is not None
        and price_52w_low > 0
        and price >= price_52w_low * 1.25
        and price_52w_high is not None
        and price_52w_high > 0
        and price >= price_52w_high * 0.75
    )

    market_trend, vix_regime_label = ta.market_regime_inputs(
        benchmark_close.iloc[: i + 1] if benchmark_close is not None else None,
        vix_close.iloc[: i + 1] if vix_close is not None else None,
    )

    # Segunda auditoría, Bloque 5: which of watchlist_service.py's four
    # short-term setup types would have matched at this historical point -
    # same exact thresholds as that module's own detectors (reusing its
    # named constants where it has them), duplicated here rather than
    # imported directly because watchlist_service operates on a
    # TickerSnapshot (a live, cross-sectional object this script never
    # builds), not a raw indicator series at an arbitrary past bar. These are
    # setup *memberships*, not recommendation_engine.py factors - excluded
    # from the regular per-factor reports (see `factor_names` below), used
    # only to segment samples (`segment_by_setup_type`).
    change_1d = ta.pct_change_over(close.iloc[: i + 1], 1)
    change_1w = ta.pct_change_over(close.iloc[: i + 1], 5)
    relative_volume = ta.relative_volume(volume.iloc[: i + 1])
    dist_52w_high = ta.distance_to_rolling_extreme(close.iloc[: i + 1], 252, "high")

    setup_oversold_bounce = not pd.isna(rsi_t) and rsi_t <= 35 and change_1d is not None and change_1d > 0
    setup_breakout_volume = (
        dist_52w_high is not None
        and dist_52w_high >= -0.02
        and relative_volume is not None
        and relative_volume >= 1.3
    )
    setup_trend_continuation = strong_trend and change_1w is not None and change_1w > 0
    setup_pullback_to_support = (
        not pd.isna(s50)
        and not pd.isna(s200)
        and s50 > s200
        and price >= s50
        and (price - s50) / s50 <= wl.PULLBACK_MAX_DISTANCE_ABOVE_SMA50
        and not pd.isna(rsi_t)
        and rsi_t > wl.PULLBACK_MIN_RSI
    )

    # Fase 8 reorientation: the actual live gate, replayed point-in-time -
    # see the module docstring's "Fase 8 reorientation" section for why this
    # calls the real `levels_engine.replay_gate_at` instead of re-deriving
    # each condition by hand. Guaranteed non-None here - this function's own
    # guard above (`pd.isna(s20) or pd.isna(s50) or pd.isna(s200)`) is the
    # exact same one `replay_gate_at` makes internally.
    gate = le.replay_gate_at(
        i, close, sma20, sma50, sma150, sma200, rsi14, adx14, plus_di, minus_di, atr14, volume
    )
    assert gate is not None
    # Positional, not by label - evaluate_gate()'s six `add(...)` calls are in
    # a fixed, documented order (see that function). The sixth (reward:risk)
    # is intentionally not unpacked into its own factor - see the module
    # docstring.
    (
        gate_trend_or_stage2,
        gate_not_parabolic,
        gate_not_overbought_outside_strong_trend,
        gate_no_obv_bearish_divergence,
        gate_no_fast_pair_veto,
        _gate_reward_risk_ok,
    ) = (c.passed for c in gate.conditions)

    return {
        "trend_up": trend == ta.TrendState.UPTREND,
        "trend_down": trend == ta.TrendState.DOWNTREND,
        "stage2": stage == ta.Stage.STAGE_2,
        "stage4": stage == ta.Stage.STAGE_4,
        "golden_cross": ma_cross == "golden",
        "death_cross": ma_cross == "death",
        "adx_strong_trend": strong_trend,
        "rsi_oversold_bounce": rsi_oversold_bounce,
        "obv_bullish": obv_div == "bullish",
        "minervini_range_position": minervini_range_position,
        "market_below_sma200": market_trend == ta.TrendState.DOWNTREND,
        "vix_stress": vix_regime_label in ("pánico", "crisis"),
        "gate_passes": gate.passes,
        "gate_trend_or_stage2": gate_trend_or_stage2,
        "gate_not_parabolic": gate_not_parabolic,
        "gate_not_overbought_outside_strong_trend": gate_not_overbought_outside_strong_trend,
        "gate_no_obv_bearish_divergence": gate_no_obv_bearish_divergence,
        "gate_no_fast_pair_veto": gate_no_fast_pair_veto,
        "setup_oversold_bounce": setup_oversold_bounce,
        "setup_breakout_volume": setup_breakout_volume,
        "setup_trend_continuation": setup_trend_continuation,
        "setup_pullback_to_support": setup_pullback_to_support,
    }


def collect_samples_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    horizon_days: int,
    benchmark_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
) -> list[FactorSample]:
    """Non-overlapping sampling grid (stride == horizon), same as before -
    but each sample's return now comes from `backtest_engine.label_triple_barrier`
    (trailing Chandelier stop, sized with the live system's own
    ATR_STOP_MULTIPLE/REWARD_RISK_RATIO), not a naive fixed-horizon return.
    `demeaned_return` is left at 0.0 here - filled in by
    `demean_cross_sectionally` once every ticker's samples for this horizon
    are pooled (demeaning needs the whole cross-section, not one ticker at a
    time)."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    open_ = df["open"]
    n = len(close)
    if n < MIN_BARS_REQUIRED:
        return []

    sma20_s, sma50_s = ta.sma(close, 20), ta.sma(close, 50)
    sma150_s, sma200_s = ta.sma(close, 150), ta.sma(close, 200)
    rsi_s = ta.rsi(close)
    adx_s = ta.adx(high, low, close)
    plus_di_s, minus_di_s = ta.dmi(high, low, close)
    atr_s = ta.atr(high, low, close)

    # Realigned once, up front (not per bar) - a benchmark/VIX series can have
    # minor calendar differences (holidays) from the ticker's own calendar.
    aligned_benchmark = benchmark_close.reindex(close.index).ffill() if benchmark_close is not None else None
    aligned_vix = vix_close.reindex(close.index).ffill() if vix_close is not None else None

    last_valid_start = n - horizon_days - 1
    if last_valid_start <= WARMUP_BARS:
        return []

    samples = []
    for i in range(WARMUP_BARS, last_valid_start, horizon_days):
        triggers = compute_triggers_at(
            i, close, high, low, volume, sma20_s, sma50_s, sma150_s, sma200_s,
            rsi_s, adx_s, plus_di_s, minus_di_s, atr_s, aligned_benchmark, aligned_vix,
        )
        if triggers is None:
            continue
        atr_t = atr_s.iloc[i]
        if pd.isna(atr_t) or atr_t <= 0:
            continue
        entry_price = float(close.iloc[i])
        stop = entry_price - ATR_STOP_MULTIPLE * float(atr_t)
        if stop >= entry_price:
            continue
        target = entry_price + REWARD_RISK_RATIO * (entry_price - stop)
        # vol_regime=None: a per-bar GARCH refit for 217 tickers x 10 years is
        # the same computational cost that already excludes Markov/GARCH from
        # this study (§2 "Excluidos deliberadamente") - the trailing stop
        # falls back to CHANDELIER_MULTIPLIER_DEFAULT, a reasonable, uniform
        # choice applied identically to every sample.
        label = be.label_triple_barrier(
            close, high, low, i, stop, target, horizon_days, trailing=True, atr14=atr_s, vol_regime=None,
            open_=open_,
        )
        if label is None:
            continue
        samples.append(
            FactorSample(
                ticker=ticker, date=close.index[i], fwd_return=label.return_pct, demeaned_return=0.0,
                triggers=triggers, exit_reason=label.exit_reason, bars_held=label.bars_held,
                mae_pct=label.mae_pct, mfe_pct=label.mfe_pct,
                risk_pct=(entry_price - stop) / entry_price,
            )
        )
    return samples


def demean_cross_sectionally(samples: list[FactorSample]) -> list[FactorSample]:
    """Subtracts each calendar-month bucket's own cross-sectional mean return
    from every sample in it - isolates a factor's own contribution from pure
    market beta (see this module's docstring: this is why `vix_stress`
    measured +4.35pp in the original study - that was beta from a decade
    dominated by a bull market, not a factor edge). Bucketed by month, not
    exact date: the non-overlapping sampling grid starts at each ticker's own
    `WARMUP_BARS`-th bar, so sample dates across tickers aren't perfectly
    calendar-aligned bar for bar - a month is coarse enough to tolerate that
    drift while still grouping samples that saw broadly the same market."""
    buckets: dict[pd.Period, list[float]] = defaultdict(list)
    for s in samples:
        buckets[s.date.to_period("M")].append(s.fwd_return)
    bucket_means = {period: float(np.mean(returns)) for period, returns in buckets.items()}
    return [replace(s, demeaned_return=s.fwd_return - bucket_means[s.date.to_period("M")]) for s in samples]


def compute_information_coefficient(
    factor: str, samples: list[FactorSample]
) -> tuple[float | None, float | None, int]:
    """Mean Information Coefficient and its Information Ratio (mean IC /
    std(IC)) for one factor: per calendar-month bucket, the Spearman rank
    correlation between the (boolean, 0/1) trigger and that bucket's
    demeaned forward returns, averaged across buckets. More informative than
    one pooled significance test because it shows whether an edge is
    consistent bucket to bucket or driven by one or two lucky periods -
    the standard factor-evaluation metric in the industry."""
    buckets: dict[pd.Period, list[tuple[float, float]]] = defaultdict(list)
    for s in samples:
        buckets[s.date.to_period("M")].append((float(s.triggers[factor]), s.demeaned_return))

    ics = []
    for pairs in buckets.values():
        if len(pairs) < MIN_BUCKET_SIZE_FOR_IC:
            continue
        trigger_values = np.array([p[0] for p in pairs])
        return_values = np.array([p[1] for p in pairs])
        if len(set(trigger_values.tolist())) < 2:
            continue  # no variation in the trigger this bucket - correlation undefined
        ic, _ = stats.spearmanr(trigger_values, return_values)
        if not np.isnan(ic):
            ics.append(float(ic))

    if not ics:
        return None, None, 0
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else None
    ic_ir = mean_ic / std_ic if std_ic and std_ic > 0 else None
    return mean_ic, ic_ir, len(ics)


def run_multivariate_regression(
    samples: list[FactorSample], factor_names: list[str]
) -> dict[str, tuple[float, float]]:
    """All factors as simultaneous OLS predictors of demeaned forward return
    - answers "does this factor still matter once every other (possibly
    collinear) factor is accounted for", not just "does it correlate on its
    own". Trend/stage/RS Rating/Minervini all measure some version of "is
    this in a confirmed uptrend" (recommendation_engine.py's own docstring
    says so) - a univariate-only test on collinear factors credits the same
    underlying fact several times over. Returns {factor: (coef, p_value)}."""
    import statsmodels.api as sm

    design = pd.DataFrame({f: [float(s.triggers[f]) for s in samples] for f in factor_names})
    design = sm.add_constant(design)
    target = np.array([s.demeaned_return for s in samples])
    model = sm.OLS(target, design, missing="drop").fit()
    return {f: (float(model.params[f]), float(model.pvalues[f])) for f in factor_names}


def segment_by_regime(samples: list[FactorSample]) -> dict[str, list[FactorSample]]:
    """The same pooled sample set, split by the two market-regime dimensions
    this project already estimates (Faber SMA200 gate, VIX stress) - a
    factor can work in one regime and fail in another, and a single pooled
    average hides both (see docs/quant_methodology.md §6.1 for why this
    project treats "regime" as something to segment by, not score by)."""
    return {
        "market_above_sma200": [s for s in samples if not s.triggers.get("market_below_sma200", False)],
        "market_below_sma200": [s for s in samples if s.triggers.get("market_below_sma200", False)],
        "vix_calm": [s for s in samples if not s.triggers.get("vix_stress", False)],
        "vix_stress": [s for s in samples if s.triggers.get("vix_stress", False)],
    }


SETUP_TRIGGER_KEYS = (
    "setup_oversold_bounce", "setup_breakout_volume", "setup_trend_continuation", "setup_pullback_to_support",
)


def segment_by_setup_type(samples: list[FactorSample]) -> dict[str, list[FactorSample]]:
    """The same pooled sample set, split by which of watchlist_service.py's
    four short-term setup types applies at each point (Segunda auditoría,
    Bloque 3/5) - a factor's effect can differ by setup the same way it can
    differ by market regime (see `segment_by_regime`). Not mutually
    exclusive the way regime segments are - a sample can match more than one
    setup, or none, and still appears in every segment it matches."""
    return {
        key.removeprefix("setup_"): [s for s in samples if s.triggers.get(key, False)]
        for key in SETUP_TRIGGER_KEYS
    }


@dataclass(frozen=True, slots=True)
class SetupOutcomeStats:
    """Tercera auditoría, Bloque F-6: "how much do I typically make with this
    setup", not another read of recommendation_engine's own factors. Every
    field comes straight from `backtest_engine.label_triple_barrier`'s own
    per-sample output (`FactorSample.exit_reason`/`bars_held`/`mae_pct`/`risk_pct`),
    which `segment_by_setup_type`'s samples always had - just never
    aggregated into this before. Not demeaned (unlike FactorResult) - a win
    rate/expectancy is meant to read as the setup's own real-world number,
    not relative to the cross-section's mean that day."""

    setup: str
    n: int
    win_rate: float  # fraction with exit_reason == "target" (stop/vertical both count as not-a-win)
    expectancy_r: float  # mean(fwd_return / risk_pct) - this sample's own ATR-based "1R", not a shared constant
    median_bars_held: float
    mae_p80_pct: float  # 80th percentile of |mae_pct| - where a stop this size routinely gets tested


def compute_setup_outcome_stats(samples: list[FactorSample]) -> dict[str, SetupOutcomeStats]:
    """One `SetupOutcomeStats` per setup segment with at least `MIN_GROUP_SIZE`
    samples - thinner segments are omitted rather than reported on too little
    data to trust."""
    stats: dict[str, SetupOutcomeStats] = {}
    for setup_name, setup_samples in segment_by_setup_type(samples).items():
        if len(setup_samples) < MIN_GROUP_SIZE:
            continue
        wins = sum(1 for s in setup_samples if s.exit_reason == "target")
        r_multiples = np.array([s.fwd_return / s.risk_pct for s in setup_samples if s.risk_pct > 0])
        bars_held = np.array([s.bars_held for s in setup_samples])
        mae_abs_pct = np.abs(np.array([s.mae_pct for s in setup_samples]))
        stats[setup_name] = SetupOutcomeStats(
            setup=setup_name,
            n=len(setup_samples),
            win_rate=wins / len(setup_samples),
            expectancy_r=float(np.mean(r_multiples)) if len(r_multiples) else 0.0,
            median_bars_held=float(np.median(bars_held)),
            mae_p80_pct=float(np.percentile(mae_abs_pct, 80)),
        )
    return stats


def split_samples_by_date(
    samples: list[FactorSample], cutoff: pd.Timestamp = TEMPORAL_SPLIT_CUTOFF
) -> tuple[list[FactorSample], list[FactorSample]]:
    """(calibrate, validate) - calibrate is strictly before `cutoff`,
    validate on/after - reuses `backtest_engine.split_by_date`'s own
    boundary convention (Segunda auditoría, Bloque 5). Never mixed: report
    each side separately, never pool them into one statistic."""
    if not samples:
        return [], []
    dates = pd.DatetimeIndex([s.date for s in samples])
    calibrate_mask, validate_mask = be.split_by_date(dates, cutoff)
    calibrate = [s for s, m in zip(samples, calibrate_mask, strict=True) if m]
    validate = [s for s, m in zip(samples, validate_mask, strict=True) if m]
    return calibrate, validate


def benjamini_hochberg_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate-adjusted p-values. Standard
    step-up procedure: sort ascending, adjust each by n/rank, then enforce
    monotonicity by taking a running minimum from the largest p-value down -
    without that last step, adjusted p-values wouldn't necessarily preserve
    the original ordering, which would make no sense for a p-value."""
    n = len(p_values)
    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        candidate = p_values[idx] * n / (rank + 1)
        running_min = min(running_min, candidate)
        adjusted[idx] = min(running_min, 1.0)
    return adjusted


def _permutation_test(
    sample_a: np.ndarray, sample_b: np.ndarray, n_permutations: int = N_PERMUTATIONS, seed: int = 0
) -> float:
    """Empirical two-sided p-value for the difference in means, under the null
    that the two samples carry no real difference (shuffled relative to the
    pooled values). Doesn't assume normality, unlike the t-test - reported
    alongside it in `analyze_factor` below for exactly that reason.

    Moved here unchanged (2026-09, reconstruction Fase 4) from the retired
    `walk_forward_backtest.py` - this script was always its only real
    consumer of this specific helper (a generic statistics routine, no
    dependency on that module's own retired scoring replay), see
    docs/quant_methodology.md for the retirement note."""
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
class _RawFactorStats:
    factor: str
    mean_triggered: float
    mean_not_triggered: float
    mean_diff: float
    t_stat: float
    p_value: float
    perm_p_value: float
    n_triggered: int
    n_not_triggered: int


def analyze_factor(factor: str, samples: list[FactorSample]) -> _RawFactorStats | None:
    """Raw stats for one factor, on demeaned returns - t-test *and*
    permutation test, same rigor as before. The Benjamini-Hochberg
    adjustment is computed separately, across *all* factors at once, in
    `run_study_for_horizon` (it needs every factor's p-value together, not
    one at a time)."""
    triggered = np.array([s.demeaned_return for s in samples if s.triggers[factor]])
    not_triggered = np.array([s.demeaned_return for s in samples if not s.triggers[factor]])
    if len(triggered) < MIN_GROUP_SIZE or len(not_triggered) < MIN_GROUP_SIZE:
        return None

    t_stat, p_value = stats.ttest_ind(triggered, not_triggered, equal_var=False)
    perm_p = _permutation_test(triggered, not_triggered, n_permutations=N_PERMUTATIONS, seed=42)
    return _RawFactorStats(
        factor=factor,
        mean_triggered=float(triggered.mean()),
        mean_not_triggered=float(not_triggered.mean()),
        mean_diff=float(triggered.mean() - not_triggered.mean()),
        t_stat=float(t_stat),
        p_value=float(p_value),
        perm_p_value=perm_p,
        n_triggered=len(triggered),
        n_not_triggered=len(not_triggered),
    )


def resolve_universe_tickers(regions: list[str], use_dynamic_universe: bool) -> dict[str, str]:
    """ticker -> region (Tercera auditoría, Bloque F-1: used to return a flat,
    region-erased list - `filter_samples_by_point_in_time_membership` below
    needs to know which region's own snapshot history to check each ticker
    against, since US/Europe are separate, never-blended universes).

    The curated `market_universe.py` dict (default, unchanged behavior) or
    the point-in-time `universe_memberships` table (Segunda auditoría,
    Bloque 5 - `--use-dynamic-universe`) per region, with an explicit,
    logged fallback to the curated list for any region that hasn't been
    refreshed yet (`scripts/refresh_universe_membership.py`) rather than
    silently returning nothing for it."""
    ticker_region: dict[str, str] = {}
    if not use_dynamic_universe:
        for region in regions:
            for ticker in universe_tickers(region):
                ticker_region[ticker] = region
        return ticker_region

    db = SessionLocal()
    try:
        repo = UniverseMembershipRepository(db)
        for region in regions:
            dynamic = dus.read_dynamic_universe(repo, region)
            if dynamic is None:
                print(f"[{region}] no point-in-time snapshot on file - falling back to the curated universe")
                for ticker in universe_tickers(region):
                    ticker_region[ticker] = region
            else:
                print(f"[{region}] using point-in-time universe: {len(dynamic)} tickers")
                for ticker in dynamic:
                    ticker_region[ticker] = region
    finally:
        db.close()
    return ticker_region


def download_universe_ohlcv(
    regions: list[str], use_dynamic_universe: bool = False
) -> tuple[dict[str, pd.DataFrame], dict[str, str], pd.Series | None, dict[str, str]]:
    """Returns (ohlcv_by_ticker, benchmark_ticker_by_ticker, vix_close,
    ticker_region) - the benchmark map and VIX series are shared,
    single-fetch inputs every ticker's factor computation reuses for the
    market-regime factors. `ticker_region` (new, Bloque F-1) is threaded
    through to `filter_samples_by_point_in_time_membership`."""
    ticker_region = resolve_universe_tickers(regions, use_dynamic_universe)
    tickers = sorted(ticker_region)
    print(f"Universe: {len(tickers)} tickers across {regions}")

    benchmark_ticker_by_ticker = {ticker: benchmark_for_ticker(ticker) for ticker in tickers}
    fetch_list = sorted({*tickers, *set(benchmark_ticker_by_ticker.values()), VIX_TICKER})

    provider = YFinanceProvider()
    market_data = MarketDataService(provider)
    end = date.today()
    start = end - timedelta(days=365 * HISTORY_YEARS)

    print(f"Downloading {HISTORY_YEARS}y of daily OHLCV for {len(fetch_list)} tickers (this takes a while)...")
    ohlcv_by_ticker = market_data.get_bulk_ohlcv(fetch_list, start, end)
    print(f"Got data for {len(ohlcv_by_ticker)}/{len(fetch_list)} tickers")

    vix_df = ohlcv_by_ticker.get(VIX_TICKER)
    vix_close = vix_df["close"] if vix_df is not None else None
    return ohlcv_by_ticker, benchmark_ticker_by_ticker, vix_close, ticker_region


def _resolve_as_of_snapshot_date(available_dates: list[date], sample_date: date) -> date | None:
    """The latest snapshot date on or before `sample_date` - or, if the
    sample predates every snapshot on file (the honest limitation this
    module's own docstring already discloses: point-in-time data only
    exists from whenever the first refresh ran forward), the *earliest*
    available snapshot instead of none at all. Using the earliest known
    snapshot for a pre-history sample is never worse than the bug this
    replaces (today's snapshot applied blindly to 10 years of samples) and,
    unlike that, becomes genuinely point-in-time-correct for every sample
    date once enough monthly snapshots have accumulated going forward.
    `None` only when the region has no snapshot on file at all yet."""
    if not available_dates:
        return None
    on_or_before = [d for d in available_dates if d <= sample_date]
    return max(on_or_before) if on_or_before else min(available_dates)


def filter_samples_by_point_in_time_membership(
    samples: list[FactorSample], ticker_region: dict[str, str], use_dynamic_universe: bool
) -> list[FactorSample]:
    """Drops any sample whose ticker was not, per the closest available
    point-in-time snapshot, a universe member as of that *sample's own*
    date (Tercera auditoría, Bloque F-1). Without this,
    `--use-dynamic-universe` only ever checked *today's* snapshot against
    10 years of historical samples - a ticker dropped from the index in
    2020 still validated every 2016-2019 sample via today's 2026 snapshot,
    which reduces survivorship bias by exactly zero. A no-op when
    `use_dynamic_universe` is False - the curated universe makes no
    point-in-time claim to check samples against."""
    if not use_dynamic_universe or not samples:
        return samples

    db = SessionLocal()
    try:
        repo = UniverseMembershipRepository(db)
        dates_by_region = {region: repo.all_as_of_dates(region) for region in set(ticker_region.values())}
        members_cache: dict[tuple[str, date], dict[str, str | None]] = {}

        def members_as_of(region: str, sample_date: date) -> dict[str, str | None] | None:
            as_of = _resolve_as_of_snapshot_date(dates_by_region.get(region, []), sample_date)
            if as_of is None:
                return None
            key = (region, as_of)
            if key not in members_cache:
                members_cache[key] = dus.read_dynamic_universe(repo, region, as_of_date=as_of) or {}
            return members_cache[key]

        kept = []
        dropped = 0
        for sample in samples:
            region = ticker_region.get(sample.ticker)
            if region is None:
                kept.append(sample)  # unknown region (shouldn't happen) - never drop blind
                continue
            members = members_as_of(region, sample.date.date())
            if members is None or sample.ticker in members:
                kept.append(sample)
            else:
                dropped += 1
        if dropped:
            print(f"  Point-in-time membership filter: dropped {dropped}/{len(samples)} samples")
        return kept
    finally:
        db.close()


def _build_results(samples: list[FactorSample], factor_names: list[str]) -> list[FactorResult]:
    raw_results: list[_RawFactorStats] = []
    for factor in factor_names:
        result = analyze_factor(factor, samples)
        if result is not None:
            raw_results.append(result)
    if not raw_results:
        return []

    bh_adjusted = benjamini_hochberg_adjust([r.perm_p_value for r in raw_results])
    multivariate = run_multivariate_regression(samples, factor_names)

    results: list[FactorResult] = []
    for r, perm_p_bh in zip(raw_results, bh_adjusted, strict=True):
        current_points = CURRENT_POINTS.get(r.factor, 0)
        directionally_consistent = (r.mean_diff > 0 and current_points > 0) or (
            r.mean_diff < 0 and current_points < 0
        )
        mean_ic, ic_ir, n_ic_buckets = compute_information_coefficient(r.factor, samples)
        mv_coef, mv_p = multivariate.get(r.factor, (None, None))
        results.append(
            FactorResult(
                factor=r.factor,
                current_points=current_points,
                n_triggered=r.n_triggered,
                n_not_triggered=r.n_not_triggered,
                mean_return_triggered=r.mean_triggered,
                mean_return_not_triggered=r.mean_not_triggered,
                mean_difference=r.mean_diff,
                t_stat=r.t_stat,
                p_value=r.p_value,
                permutation_p_value=r.perm_p_value,
                permutation_p_value_bh=perm_p_bh,
                significant_at_1pct=bool(r.perm_p_value < 0.01),
                significant_at_1pct_bh=bool(perm_p_bh < 0.01),
                directionally_consistent=directionally_consistent,
                mean_ic=mean_ic,
                ic_ir=ic_ir,
                n_ic_buckets=n_ic_buckets,
                multivariate_coef_pct=round(mv_coef * 100, 4) if mv_coef is not None else None,
                multivariate_p_value=round(mv_p, 5) if mv_p is not None else None,
            )
        )
    return results


_REPORT_COLUMNS = [
    "factor", "current_points", "n_triggered", "n_not_triggered", "mean_return_triggered_pct",
    "mean_return_not_triggered_pct", "mean_difference_pct", "t_stat", "p_value", "permutation_p_value",
    "permutation_p_value_bh", "significant_at_1pct", "significant_at_1pct_bh", "directionally_consistent",
    "mean_ic", "ic_ir", "n_ic_buckets", "multivariate_coef_pct", "multivariate_p_value",
]


def _results_to_frame(results: list[FactorResult]) -> pd.DataFrame:
    # An empty `results` (every factor fell below MIN_GROUP_SIZE - a thin
    # sample, e.g. a short backtest window or a narrow universe) must still
    # produce a well-formed, empty-but-correctly-columned frame: pd.DataFrame([])
    # has *no* columns at all, and .sort_values on a column that doesn't
    # exist raises - better to say "no factor had enough data" explicitly
    # than to crash on a legitimate, if unhelpful, result.
    if not results:
        return pd.DataFrame(columns=_REPORT_COLUMNS)
    rows = [
        {
            "factor": r.factor,
            "current_points": r.current_points,
            "n_triggered": r.n_triggered,
            "n_not_triggered": r.n_not_triggered,
            "mean_return_triggered_pct": round(r.mean_return_triggered * 100, 3),
            "mean_return_not_triggered_pct": round(r.mean_return_not_triggered * 100, 3),
            "mean_difference_pct": round(r.mean_difference * 100, 3),
            "t_stat": round(r.t_stat, 3),
            "p_value": round(r.p_value, 5),
            "permutation_p_value": round(r.permutation_p_value, 5),
            "permutation_p_value_bh": round(r.permutation_p_value_bh, 5),
            "significant_at_1pct": r.significant_at_1pct,
            "significant_at_1pct_bh": r.significant_at_1pct_bh,
            "directionally_consistent": r.directionally_consistent,
            "mean_ic": round(r.mean_ic, 4) if r.mean_ic is not None else None,
            "ic_ir": round(r.ic_ir, 3) if r.ic_ir is not None else None,
            "n_ic_buckets": r.n_ic_buckets,
            "multivariate_coef_pct": r.multivariate_coef_pct,
            "multivariate_p_value": r.multivariate_p_value,
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("permutation_p_value_bh")


def collect_all_samples(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    benchmark_ticker_by_ticker: dict[str, str],
    vix_close: pd.Series | None,
    horizon_days: int,
) -> list[FactorSample]:
    """Every ticker's own samples, pooled - the raw material every report in
    this script (pooled, regime-segmented, setup-segmented, temporally
    split) is built from. Returns un-demeaned samples - demeaning happens
    per report, on whatever subset of samples that specific report actually
    covers (see `build_reports`): demeaning a temporal split against the
    *other* split's cross-section would leak information across the exact
    boundary this split exists to keep apart."""
    tickers_only = [t for t in ohlcv_by_ticker if t in benchmark_ticker_by_ticker]
    all_samples: list[FactorSample] = []
    for idx, ticker in enumerate(tickers_only, 1):
        df = ohlcv_by_ticker[ticker]
        benchmark_df = ohlcv_by_ticker.get(benchmark_ticker_by_ticker[ticker])
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
        samples = collect_samples_for_ticker(ticker, df, horizon_days, benchmark_close, vix_close)
        all_samples.extend(samples)
        if idx % 50 == 0:
            print(f"  [horizon={horizon_days}] processed {idx}/{len(tickers_only)} tickers")
    print(f"[horizon={horizon_days}] Total pooled samples: {len(all_samples)}")
    return all_samples


def _setup_outcome_stats_to_frame(stats: dict[str, SetupOutcomeStats]) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame(columns=["setup", "n", "win_rate", "expectancy_r", "median_bars_held", "mae_p80_pct"])
    return pd.DataFrame(
        [
            {
                "setup": s.setup,
                "n": s.n,
                "win_rate": round(s.win_rate, 4),
                "expectancy_r": round(s.expectancy_r, 4),
                "median_bars_held": s.median_bars_held,
                "mae_p80_pct": round(s.mae_p80_pct * 100, 3),
            }
            for s in stats.values()
        ]
    )


def build_reports(
    samples: list[FactorSample], horizon_days: int, label: str = ""
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """(pooled_report, regime_reports, setup_reports, setup_outcome_report)
    for one already-collected sample set - demeans internally, using only
    this set's own cross-section (see `collect_all_samples`'s docstring on
    why that must never mix with samples outside the set being measured).
    `setup_outcome_report` (Bloque F-6) is the setup's own realized trading
    outcome (win rate/expectancy in R/median duration/MAE p80) - distinct
    from `setup_reports`, which measures recommendation_engine *factors*
    within each setup segment, never the setup's own outcome. Empty
    frame/dicts when `samples` is empty (e.g. a temporal split with too
    little history on one side) rather than raising - a thin split is a
    real, reportable result, not a script failure."""
    if not samples:
        empty_outcomes = _setup_outcome_stats_to_frame({})
        return _results_to_frame([]), {}, {}, empty_outcomes

    samples = demean_cross_sectionally(samples)
    # setup_* keys are membership flags for segment_by_setup_type, not
    # recommendation_engine.py-weighted factors - excluded from the regular
    # per-factor reports so they don't show up as if they were one.
    factor_names = [f for f in samples[0].triggers if not f.startswith("setup_")]

    pooled_report = _results_to_frame(_build_results(samples, factor_names))

    regime_reports: dict[str, pd.DataFrame] = {}
    for regime_name, regime_samples in segment_by_regime(samples).items():
        results = _build_results(regime_samples, factor_names)
        if results:
            regime_reports[regime_name] = _results_to_frame(results)
        print(f"  [horizon={horizon_days}]{label} regime '{regime_name}': {len(regime_samples)} samples")

    setup_reports: dict[str, pd.DataFrame] = {}
    for setup_name, setup_samples in segment_by_setup_type(samples).items():
        results = _build_results(setup_samples, factor_names)
        if results:
            setup_reports[setup_name] = _results_to_frame(results)
        print(f"  [horizon={horizon_days}]{label} setup '{setup_name}': {len(setup_samples)} samples")

    setup_outcome_report = _setup_outcome_stats_to_frame(compute_setup_outcome_stats(samples))

    return pooled_report, regime_reports, setup_reports, setup_outcome_report


def _save_report(report: pd.DataFrame, path: str, description: str) -> None:
    report.to_csv(path, index=False)
    print(f"Saved {description} to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Forward-return horizon(s) in trading days - defaults to this portfolio's real holding "
        "horizon (5/10/21); pass --horizons 63 126 to check longer-horizon momentum sensitivity",
    )
    parser.add_argument("--regions", nargs="+", default=["us", "europe"], help="Universe regions to include")
    parser.add_argument("--out-prefix", type=str, default="factor_ablation_report", help="Output CSV path prefix")
    parser.add_argument(
        "--use-dynamic-universe",
        action="store_true",
        help="Segunda auditoría, Bloque 5: read each region's point-in-time universe "
        "(scripts/refresh_universe_membership.py) instead of the curated market_universe.py dict - "
        "falls back to curated, per region, with a logged notice, if that region has no snapshot yet.",
    )
    parser.add_argument(
        "--temporal-split",
        action="store_true",
        help="Segunda auditoría, Bloque 5: also report calibrate (before "
        f"{TEMPORAL_SPLIT_CUTOFF.date()}) and validate (on/after) separately - never mixed into "
        "the main pooled report.",
    )
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    ohlcv, benchmark_by_ticker, vix_close, ticker_region = download_universe_ohlcv(
        args.regions, args.use_dynamic_universe
    )
    for horizon in args.horizons:
        all_samples = collect_all_samples(ohlcv, benchmark_by_ticker, vix_close, horizon)
        if not all_samples:
            raise SystemExit("No samples collected - check ticker universe / data availability")
        # Tercera auditoría, Bloque F-1: without this, --use-dynamic-universe
        # only ever checked *today's* snapshot against every historical
        # sample - see filter_samples_by_point_in_time_membership's own
        # docstring for why that reduced survivorship bias by exactly zero.
        all_samples = filter_samples_by_point_in_time_membership(
            all_samples, ticker_region, args.use_dynamic_universe
        )
        if not all_samples:
            raise SystemExit("No samples survived the point-in-time membership filter")

        pooled_report, regime_reports, setup_reports, setup_outcome_report = build_reports(all_samples, horizon)
        print("\n" + "=" * 100)
        print(f"HORIZON = {horizon} trading days (pooled, cross-sectionally demeaned)")
        print("=" * 100)
        print(pooled_report.to_string(index=False))
        _save_report(pooled_report, f"{args.out_prefix}_h{horizon}.csv", "pooled report")

        for regime_name, regime_report in regime_reports.items():
            _save_report(
                regime_report, f"{args.out_prefix}_h{horizon}_regime_{regime_name}.csv",
                f"regime segment '{regime_name}'",
            )

        for setup_name, setup_report in setup_reports.items():
            _save_report(
                setup_report,
                f"{args.out_prefix}_h{horizon}_setup_{setup_name}.csv",
                f"setup segment '{setup_name}'",
            )

        print("\nSetup outcome stats (win rate / expectancy in R / median duration / MAE p80):")
        print(setup_outcome_report.to_string(index=False))
        _save_report(
            setup_outcome_report, f"{args.out_prefix}_h{horizon}_setup_outcomes.csv", "setup outcome stats"
        )

        if args.temporal_split:
            calibrate_samples, validate_samples = split_samples_by_date(all_samples)
            print(
                f"  [horizon={horizon}] temporal split: {len(calibrate_samples)} calibrate "
                f"(< {TEMPORAL_SPLIT_CUTOFF.date()}), {len(validate_samples)} validate (>= same date)"
            )
            for split_name, split_samples in (("calibrate", calibrate_samples), ("validate", validate_samples)):
                split_pooled, split_regime, split_setup, split_outcomes = build_reports(
                    split_samples, horizon, label=f" {split_name}"
                )
                _save_report(
                    split_pooled, f"{args.out_prefix}_h{horizon}_{split_name}.csv", f"{split_name} pooled report"
                )
                for regime_name, regime_report in split_regime.items():
                    _save_report(
                        regime_report,
                        f"{args.out_prefix}_h{horizon}_{split_name}_regime_{regime_name}.csv",
                        f"{split_name} regime segment '{regime_name}'",
                    )
                for setup_name, setup_report in split_setup.items():
                    _save_report(
                        setup_report,
                        f"{args.out_prefix}_h{horizon}_{split_name}_setup_{setup_name}.csv",
                        f"{split_name} setup segment '{setup_name}'",
                    )
                _save_report(
                    split_outcomes,
                    f"{args.out_prefix}_h{horizon}_{split_name}_setup_outcomes.csv",
                    f"{split_name} setup outcome stats",
                )
