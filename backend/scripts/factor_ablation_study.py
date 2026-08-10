"""Cross-sectional factor ablation study: does each individual recommendation-
engine factor actually correlate with forward returns, measured independently,
pooled across the *entire* curated universe (US + Europe, ~217 tickers) -
not just one ticker at a time.

Why this exists (see `recommendation_engine.py`'s module docstring): the
existing `walk_forward_backtest.py` validates the *combined* "comprar" vs
"evitar" verdict on one ticker at a time, which answers "does the system work
on THIS stock" but not "which specific factor is actually pulling its weight,
and which is dead weight (or worse, wrong-signed)". This script answers that,
using the same statistical rigor already established in this codebase:

- Non-overlapping sampling (stride == forecast horizon) to avoid the classic
  overlapping-window pseudo-replication trap.
- Welch's t-test AND a permutation test (no distributional assumptions) for
  every factor, exactly like `walk_forward_backtest.py`'s pairwise verdict
  tests - reusing that module's own `_permutation_test` rather than
  reimplementing it.
- Pooling across ~217 tickers instead of one gives roughly 20,000+ samples
  instead of ~100-150 per ticker, which is the actual reason this can say
  something a single-ticker backtest structurally cannot: a per-ticker test
  has to stay silent ("historial insuficiente") far more often than a
  cross-sectional one does.

Only tests factors computable causally from a plain OHLCV frame (no lookahead,
no cross-sectional info needed at each historical point) - RS Rating, Markov,
GARCH and near-support/resistance proximity are excluded for the same reasons
already documented in `walk_forward_backtest.py` (RS/Markov/GARCH need either
a point-in-time universe snapshot or a per-point refit that isn't available/
affordable at every historical bar; support/resistance's pivot scan is O(n)
per call). Market regime (benchmark below its own SMA200, VIX stress) IS
included - both are single shared series per region (SPY/STOXX + VIX), cheap
to fetch once and reuse across every ticker's replay.

Multiple-comparison correction: testing ~15 factors independently at a 1%
threshold means, under the null that none of them work, an expected ~0.15
false positives by chance alone - real, if modest, and worth controlling for
rather than ignoring (a gap in the first version of this script, caught on a
second read). `run_study_for_horizon` reports Benjamini-Hochberg-adjusted
p-values (false discovery rate control) alongside the raw ones - a factor
should clear the BH-adjusted bar, not just the raw one, before being trusted.

Usage:
    python scripts/factor_ablation_study.py [--horizons 21 63 126] [--regions us europe]

Takes several minutes (mostly yfinance download time for ~217 tickers x 10
years of daily bars) - this is an offline research/calibration script, not
something that runs as part of the API.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.market_data.yfinance_provider import YFinanceProvider  # noqa: E402
from app.services import technical_analysis as ta  # noqa: E402
from app.services.market_data_service import MarketDataService  # noqa: E402
from app.services.market_universe import VIX_TICKER, benchmark_for_ticker, universe_tickers  # noqa: E402
from app.services.walk_forward_backtest import _permutation_test  # noqa: E402

WARMUP_BARS = 260  # matches walk_forward_backtest.py - enough for SMA200 + its 25-bar slope lookback
MIN_BARS_REQUIRED = WARMUP_BARS + 100
HISTORY_YEARS = 10
N_PERMUTATIONS = 5000


@dataclass(frozen=True, slots=True)
class FactorSample:
    ticker: str
    date: pd.Timestamp
    fwd_return: float
    triggers: dict[str, bool]


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


# Mirrors the point values in recommendation_engine.py at the time this was
# run, purely for the "directionally consistent" sanity check below - not
# imported directly since several of these factors (trend_down, stage4, etc.)
# don't have a single clean boolean predicate exposed by the engine itself.
CURRENT_POINTS = {
    "trend_up": 2,
    "trend_down": -3,
    "stage2": 2,
    "stage4": -3,
    "golden_cross": 1,
    "death_cross": -2,
    "adx_strong_trend": 1,
    "rsi_overbought_outside_strong_trend": -1,
    "rsi_oversold_bounce": 1,
    "atr_parabolic": -2,
    "obv_bearish": -2,
    "obv_bullish": 1,
    "minervini_range_position": 1,  # the non-RS-dependent half of the +1 confirmation bonus
    "market_below_sma200": -2,
    "vix_stress": -2,
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
    # Matches recommendation_engine.py's regime-aware definition (2026-08 audit):
    # not penalized inside a strong, ADX-confirmed uptrend.
    rsi_overbought_outside_strong_trend = (
        not pd.isna(rsi_t) and rsi_t >= 80 and not (trend == ta.TrendState.UPTREND and strong_trend)
    )
    rsi_oversold_bounce = not pd.isna(rsi_t) and rsi_t <= 30 and trend != ta.TrendState.DOWNTREND

    atr_t = atr14.iloc[i]
    atr_multiple = float((price - s50) / atr_t) if not pd.isna(atr_t) and atr_t != 0 else None
    atr_parabolic = atr_multiple is not None and atr_multiple > 4

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

    return {
        "trend_up": trend == ta.TrendState.UPTREND,
        "trend_down": trend == ta.TrendState.DOWNTREND,
        "stage2": stage == ta.Stage.STAGE_2,
        "stage4": stage == ta.Stage.STAGE_4,
        "golden_cross": ma_cross == "golden",
        "death_cross": ma_cross == "death",
        "adx_strong_trend": strong_trend,
        "rsi_overbought_outside_strong_trend": rsi_overbought_outside_strong_trend,
        "rsi_oversold_bounce": rsi_oversold_bounce,
        "atr_parabolic": atr_parabolic,
        "obv_bearish": obv_div == "bearish",
        "obv_bullish": obv_div == "bullish",
        "minervini_range_position": minervini_range_position,
        "market_below_sma200": market_trend == ta.TrendState.DOWNTREND,
        "vix_stress": vix_regime_label in ("pánico", "crisis"),
    }


def collect_samples_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    horizon_days: int,
    benchmark_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
) -> list[FactorSample]:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
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

    last_valid_start = n - horizon_days
    if last_valid_start <= WARMUP_BARS:
        return []

    samples = []
    for i in range(WARMUP_BARS, last_valid_start, horizon_days):
        triggers = compute_triggers_at(
            i,
            close,
            high,
            low,
            volume,
            sma20_s,
            sma50_s,
            sma150_s,
            sma200_s,
            rsi_s,
            adx_s,
            plus_di_s,
            minus_di_s,
            atr_s,
            aligned_benchmark,
            aligned_vix,
        )
        if triggers is None:
            continue
        fwd_return = float(close.iloc[i + horizon_days] / close.iloc[i] - 1)
        samples.append(FactorSample(ticker=ticker, date=close.index[i], fwd_return=fwd_return, triggers=triggers))
    return samples


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
    """Raw stats for one factor - t-test *and* permutation test, same as
    before. The Benjamini-Hochberg adjustment is computed separately, across
    *all* factors at once, in `run_study_for_horizon` (it needs every
    factor's p-value together, not one at a time)."""
    triggered = np.array([s.fwd_return for s in samples if s.triggers[factor]])
    not_triggered = np.array([s.fwd_return for s in samples if not s.triggers[factor]])
    if len(triggered) < 30 or len(not_triggered) < 30:
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


def download_universe_ohlcv(
    regions: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str], pd.Series | None]:
    """Returns (ohlcv_by_ticker, benchmark_ticker_by_ticker, vix_close) - the
    benchmark map and VIX series are shared, single-fetch inputs every
    ticker's factor computation reuses for the market-regime factors."""
    tickers: list[str] = []
    for region in regions:
        tickers.extend(universe_tickers(region))
    tickers = sorted(set(tickers))
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
    return ohlcv_by_ticker, benchmark_ticker_by_ticker, vix_close


def run_study_for_horizon(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    benchmark_ticker_by_ticker: dict[str, str],
    vix_close: pd.Series | None,
    horizon_days: int,
) -> pd.DataFrame:
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

    print(f"\n[horizon={horizon_days}] Total pooled samples: {len(all_samples)}")
    if not all_samples:
        raise SystemExit("No samples collected - check ticker universe / data availability")

    factor_names = list(all_samples[0].triggers.keys())
    raw_results: list[_RawFactorStats] = []
    for factor in factor_names:
        result = analyze_factor(factor, all_samples)
        if result is not None:
            raw_results.append(result)

    # BH needs every factor's p-value at once - computed here, after the loop,
    # not inside analyze_factor. See module docstring.
    bh_adjusted = benjamini_hochberg_adjust([r.perm_p_value for r in raw_results])

    results: list[FactorResult] = []
    for r, perm_p_bh in zip(raw_results, bh_adjusted, strict=True):
        current_points = CURRENT_POINTS.get(r.factor, 0)
        directionally_consistent = (r.mean_diff > 0 and current_points > 0) or (
            r.mean_diff < 0 and current_points < 0
        )
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
            )
        )

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
        }
        for r in results
    ]
    report = pd.DataFrame(rows).sort_values("permutation_p_value_bh")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[21],
        help="Forward-return horizon(s) in trading days - pass several to check horizon sensitivity "
        "(e.g. --horizons 21 63 126 to separate short-term reversal from 3-12mo momentum)",
    )
    parser.add_argument("--regions", nargs="+", default=["us", "europe"], help="Universe regions to include")
    parser.add_argument("--out-prefix", type=str, default="factor_ablation_report", help="Output CSV path prefix")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    ohlcv, benchmark_by_ticker, vix_close = download_universe_ohlcv(args.regions)
    for horizon in args.horizons:
        report = run_study_for_horizon(ohlcv, benchmark_by_ticker, vix_close, horizon)
        print("\n" + "=" * 100)
        print(f"HORIZON = {horizon} trading days")
        print("=" * 100)
        print(report.to_string(index=False))
        out_path = f"{args.out_prefix}_h{horizon}.csv"
        report.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
