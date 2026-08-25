"""Cuarta auditoría independiente, recomendación DEUDA-1: calibrates the
Chandelier Exit multipliers (`trade_manager.CHANDELIER_MULTIPLIER_BY_REGIME` /
`CHANDELIER_MULTIPLIER_PROFIT_LOCK`) against real backtest evidence instead of
leaving them at their first-draft values - the same status
`BUY_THRESHOLD`/`AVOID_THRESHOLD` had before their own dedicated audit (see
`docs/quant_methodology.md` §6).

Deliberately a *sibling* script to `factor_ablation_study.py`, not an
extension of it: that script's own `collect_samples_for_ticker` hardcodes
`vol_regime=None` for every single sample, with its own comment explaining why
- a per-bar GARCH refit across the full ~1,140-ticker universe x 10 years
would be prohibitively expensive at that scale, the same reason Markov/GARCH
are excluded from that study entirely. That means the main ablation study
structurally *cannot* exercise the regime-dependent multipliers this script
exists to test - every sample there falls back to
`CHANDELIER_MULTIPLIER_DEFAULT` regardless of what the regime dict says.

This script pays the GARCH cost deliberately, on a small, representative
sample of tickers (not the full universe) - the same trade-off the original
factor ablation study made at ~217 tickers before this project's universe
grew via the dynamic membership table. In production, the live path (GARCH is
already computed per ticker for "Analizar activo") *does* use the real
regime-dependent multiplier - this script's sample size is a deliberate cost
trade-off for a calibration run, not a claim that regime detection doesn't
matter in the live system.

Reuses `backtest_engine.find_triple_barrier_entries` (the same "what would the
system have proposed here" replay `run_triple_barrier_backtest` itself uses)
and `label_triple_barrier`'s real trailing-Chandelier simulation end to end -
never reimplements the trailing logic. For each candidate multiplier set in a
small grid around the current values, temporarily monkeypatches
`trade_manager`'s own module-level constants (the exact values
`chandelier_multiplier()` reads at call time), re-runs the trailing
simulation across the whole sample, and restores the originals immediately
after - so nothing else importing `trade_manager` in the same process is ever
at risk of seeing a half-swapped value outside the `with` block.

This script only MEASURES - it never edits `trade_manager.py`'s constants.
Any recalibration decision from its output needs the owner's explicit
sign-off, the same rule as every other weight in this system.

Usage:
    python scripts/chandelier_calibration_study.py --horizon-days 21 --sample-size 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Same idiom factor_ablation_study.py already uses - lets this script run
# standalone via `python scripts/chandelier_calibration_study.py` (no
# pythonpath set up for it the way pytest's own pyproject.toml config does
# for the test suite).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.market_data.yfinance_provider import YFinanceProvider  # noqa: E402
from app.services import backtest_engine as be  # noqa: E402
from app.services import technical_analysis as ta  # noqa: E402
from app.services import trade_manager as tm  # noqa: E402
from app.services.market_data_service import MarketDataService  # noqa: E402
from app.services.market_universe import universe_tickers  # noqa: E402
from app.services.volatility_model import fit_garch  # noqa: E402

HISTORY_YEARS = 10
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Small and deliberate - see the module docstring on why the GARCH cost forces
# this instead of the full universe. Stratified by region, not by cap tier:
# this project's own universe module already keeps US/Europe genuinely
# separate (never blended), and volatility-regime behavior is plausibly
# different enough between the two that pooling them into one number would
# hide a real difference rather than average over noise.
DEFAULT_SAMPLE_SIZE_PER_REGION = 30
DEFAULT_REGIONS = ("us", "europe")
DEFAULT_HORIZON_DAYS = 21  # this portfolio's real holding horizon (matches SIGN_CHECK_HORIZON_DAYS elsewhere)

# The grid: the current values, a symmetric +/-0.5 shift of every regime
# multiplier together, and a few profit-lock candidates crossed with each -
# kept intentionally small, since every grid point re-runs the trailing
# simulation across the entire sample.
REGIME_MULTIPLIER_STEP = 0.5
PROFIT_LOCK_CANDIDATES = (1.75, 2.0, 2.25)


@dataclass(frozen=True, slots=True)
class MultiplierCandidate:
    label: str
    by_regime: dict[str, float]
    profit_lock: float


def build_grid() -> list[MultiplierCandidate]:
    """The current live values are always candidate #1 ("actual") - every
    other candidate is measured relative to that baseline, never in
    isolation."""
    base = dict(tm.CHANDELIER_MULTIPLIER_BY_REGIME)
    base_lock = tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK
    candidates = [MultiplierCandidate("actual", base, base_lock)]

    for lock in PROFIT_LOCK_CANDIDATES:
        if lock != base_lock:
            candidates.append(MultiplierCandidate(f"actual_lock{lock}", base, lock))

    for step, tag in ((-REGIME_MULTIPLIER_STEP, "-0.5"), (REGIME_MULTIPLIER_STEP, "+0.5")):
        shifted = {regime: value + step for regime, value in base.items()}
        candidates.append(MultiplierCandidate(f"regimen{tag}", shifted, base_lock))

    return candidates


@contextmanager
def patched_multipliers(candidate: MultiplierCandidate):
    """Swaps `trade_manager`'s module-level constants for the duration of the
    `with` block only, then restores the originals - `chandelier_multiplier()`
    reads these two names at call time, so this is enough to make every
    `label_triple_barrier(trailing=True, ...)` call inside the block use the
    candidate's values without touching that function's own signature."""
    original_by_regime = tm.CHANDELIER_MULTIPLIER_BY_REGIME
    original_lock = tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK
    tm.CHANDELIER_MULTIPLIER_BY_REGIME = candidate.by_regime
    tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK = candidate.profit_lock
    try:
        yield
    finally:
        tm.CHANDELIER_MULTIPLIER_BY_REGIME = original_by_regime
        tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK = original_lock


def sample_tickers(regions: list[str], n_per_region: int, seed: int = 7) -> dict[str, str]:
    """ticker -> region, a random but reproducible (fixed seed) sample from
    the curated universe - not the dynamic point-in-time one, since this
    script's own sample size is already far below the dynamic universe's
    liquidity floor and doesn't need point-in-time membership correctness the
    way the factor ablation study does."""
    rng = np.random.RandomState(seed)
    ticker_region: dict[str, str] = {}
    for region in regions:
        tickers = universe_tickers(region)
        n = min(n_per_region, len(tickers))
        chosen = rng.choice(tickers, size=n, replace=False)
        for t in chosen:
            ticker_region[str(t)] = region
    return ticker_region


def resolve_vol_regime(close: pd.Series) -> str | None:
    """The one genuinely expensive step this script exists to pay for, on a
    deliberately small sample - see the module docstring."""
    returns = close.pct_change()
    garch = fit_garch(returns)
    return garch.regime if garch is not None else None


def trailing_labels_for_ticker(
    df: pd.DataFrame, horizon_days: int, vol_regime: str | None
) -> list[be.TripleBarrierLabel]:
    """Every trailing-strategy label for one ticker at the current (already
    patched, by the caller) Chandelier multipliers - reuses
    `find_triple_barrier_entries` for the entry points, then
    `label_triple_barrier(trailing=True, ...)` per entry, exactly like
    `run_triple_barrier_backtest` does internally, just returning the raw
    labels instead of one pre-aggregated `TradingMetrics` per ticker - this
    script pools labels *across* tickers before aggregating once."""
    close, high, low, open_, volume = df["close"], df["high"], df["low"], df["open"], df["volume"]
    sma20, sma50 = ta.sma(close, 20), ta.sma(close, 50)
    sma150, sma200 = ta.sma(close, 150), ta.sma(close, 200)
    rsi14 = ta.rsi(close)
    adx14 = ta.adx(high, low, close)
    plus_di, minus_di = ta.dmi(high, low, close)
    atr14 = ta.atr(high, low, close)

    entries = be.find_triple_barrier_entries(
        close, sma20, sma50, sma150, sma200, rsi14, adx14, plus_di, minus_di, atr14, horizon_days,
        volume=volume,
    )
    if not entries:
        return []

    labels = []
    for i, stop, target in entries:
        label = be.label_triple_barrier(
            close, high, low, i, stop, target, horizon_days, trailing=True, atr14=atr14, vol_regime=vol_regime,
            open_=open_,
        )
        if label is not None:
            labels.append(label)
    return labels


def run_calibration(
    ticker_region: dict[str, str], horizon_days: int
) -> tuple[dict[str, list[be.TripleBarrierLabel]], dict[str, str | None]]:
    """Downloads OHLCV once and fits GARCH once per ticker (the expensive,
    per-candidate-independent parts), then re-simulates the trailing strategy
    once per grid candidate - the download/GARCH cost is paid exactly once
    regardless of how many candidates are in the grid."""
    provider = YFinanceProvider()
    market_data = MarketDataService(provider)
    end = date.today()
    start = end - timedelta(days=365 * HISTORY_YEARS)

    tickers = sorted(ticker_region)
    print(f"Downloading {HISTORY_YEARS}y of daily OHLCV for {len(tickers)} sampled tickers...")
    ohlcv_by_ticker = market_data.get_bulk_ohlcv(tickers, start, end)
    print(f"Got data for {len(ohlcv_by_ticker)}/{len(tickers)} tickers")

    vol_regime_by_ticker: dict[str, str | None] = {}
    for ticker, df in ohlcv_by_ticker.items():
        vol_regime_by_ticker[ticker] = resolve_vol_regime(df["close"])

    grid = build_grid()
    rows_by_candidate: dict[str, list[be.TripleBarrierLabel]] = {}
    for candidate in grid:
        print(f"[{candidate.label}] simulating trailing exits across {len(ohlcv_by_ticker)} tickers...")
        all_labels: list[be.TripleBarrierLabel] = []
        with patched_multipliers(candidate):
            for ticker, df in ohlcv_by_ticker.items():
                all_labels.extend(
                    trailing_labels_for_ticker(df, horizon_days, vol_regime_by_ticker.get(ticker))
                )
        rows_by_candidate[candidate.label] = all_labels
        print(f"[{candidate.label}] {len(all_labels)} trades simulated")

    return rows_by_candidate, vol_regime_by_ticker


def build_report(rows_by_candidate: dict[str, list[be.TripleBarrierLabel]]) -> pd.DataFrame:
    rows = []
    for label, trade_labels in rows_by_candidate.items():
        metrics = be.compute_trading_metrics(trade_labels)
        rows.append(
            {
                "candidate": label,
                "n_trades": metrics.n_trades,
                "win_rate": metrics.win_rate,
                "expectancy_pct": metrics.expectancy_pct,
                "profit_factor": metrics.profit_factor,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "avg_mae_pct": metrics.avg_mae_pct,
                "avg_mfe_pct": metrics.avg_mfe_pct,
                "net_return_pct": metrics.net_return_pct,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE_PER_REGION,
                         help="tickers per region (not total)")
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--out", default=str(DOCS_DIR / "chandelier_calibration_report.csv"))
    args = parser.parse_args()

    ticker_region = sample_tickers(args.regions, args.sample_size)
    print(f"Sample: {len(ticker_region)} tickers across {args.regions}")

    rows_by_candidate, vol_regime_by_ticker = run_calibration(ticker_region, args.horizon_days)
    report = build_report(rows_by_candidate)
    report.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    print(report.to_string(index=False))

    regime_counts: dict[str, int] = {}
    for regime in vol_regime_by_ticker.values():
        key = regime or "sin_datos"
        regime_counts[key] = regime_counts.get(key, 0) + 1
    print(f"Régimen de volatilidad por ticker en la muestra: {regime_counts}")

    with open(str(DOCS_DIR / "chandelier_calibration_sample.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "region", "vol_regime"])
        for ticker, region in sorted(ticker_region.items()):
            writer.writerow([ticker, region, vol_regime_by_ticker.get(ticker)])
