"""Cuarta auditoría independiente, recomendación DEUDA-1: unit tests for the
pure/synthetic-data-safe pieces of scripts/chandelier_calibration_study.py.
Nothing here touches the network or fits a real GARCH model - `resolve_vol_regime`
is exercised only through `trailing_labels_for_ticker`'s caller contract (a
`vol_regime` string passed straight through), never by actually calling
`fit_garch` in a test.

Imported as `scripts.chandelier_calibration_study` - see
test_factor_ablation_study.py's own docstring for why this import path works
(pytest's `pythonpath = ["."]` + scripts/ as an implicit namespace package)."""

import numpy as np
import pandas as pd
import pytest

import scripts.chandelier_calibration_study as ccs
from app.services import trade_manager as tm


def _df(closes) -> pd.DataFrame:
    n = len(closes)
    index = pd.bdate_range("2015-01-01", periods=n)
    close = pd.Series(closes, index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).bfill(),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series([1_000_000.0] * n, index=index),
        }
    )


# --- build_grid: candidate #1 is always the live baseline ------------------


def test_build_grid_first_candidate_is_the_current_live_values():
    grid = ccs.build_grid()
    assert grid[0].label == "actual"
    assert grid[0].by_regime == tm.CHANDELIER_MULTIPLIER_BY_REGIME
    assert grid[0].profit_lock == tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK


def test_build_grid_includes_a_shifted_regime_candidate_in_both_directions():
    grid = ccs.build_grid()
    labels = [c.label for c in grid]
    assert "regimen-0.5" in labels
    assert "regimen+0.5" in labels
    up = next(c for c in grid if c.label == "regimen+0.5")
    down = next(c for c in grid if c.label == "regimen-0.5")
    for regime in tm.CHANDELIER_MULTIPLIER_BY_REGIME:
        assert up.by_regime[regime] == pytest.approx(tm.CHANDELIER_MULTIPLIER_BY_REGIME[regime] + 0.5)
        assert down.by_regime[regime] == pytest.approx(tm.CHANDELIER_MULTIPLIER_BY_REGIME[regime] - 0.5)


def test_build_grid_never_duplicates_the_baseline_profit_lock():
    # PROFIT_LOCK_CANDIDATES includes the live default (2.0) - a naive
    # cross-product would list "actual" twice under different names.
    grid = ccs.build_grid()
    baseline_lock_rows = [c for c in grid if c.by_regime == tm.CHANDELIER_MULTIPLIER_BY_REGIME
                           and c.profit_lock == tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK]
    assert len(baseline_lock_rows) == 1


# --- patched_multipliers: always restores, even on exception ---------------


def test_patched_multipliers_swaps_values_inside_the_block():
    candidate = ccs.MultiplierCandidate("test", {"baja": 9.0, "normal": 9.0, "elevada": 9.0, "alta": 9.0}, 9.0)
    with ccs.patched_multipliers(candidate):
        assert tm.CHANDELIER_MULTIPLIER_BY_REGIME == candidate.by_regime
        assert tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK == 9.0


def test_patched_multipliers_restores_originals_after_the_block():
    original_by_regime = dict(tm.CHANDELIER_MULTIPLIER_BY_REGIME)
    original_lock = tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK
    candidate = ccs.MultiplierCandidate("test", {"baja": 9.0, "normal": 9.0, "elevada": 9.0, "alta": 9.0}, 9.0)
    with ccs.patched_multipliers(candidate):
        pass
    assert tm.CHANDELIER_MULTIPLIER_BY_REGIME == original_by_regime
    assert tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK == original_lock


def test_patched_multipliers_restores_originals_even_if_the_block_raises():
    original_by_regime = dict(tm.CHANDELIER_MULTIPLIER_BY_REGIME)
    original_lock = tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK
    candidate = ccs.MultiplierCandidate("test", {"baja": 9.0, "normal": 9.0, "elevada": 9.0, "alta": 9.0}, 9.0)
    with pytest.raises(ValueError), ccs.patched_multipliers(candidate):
        raise ValueError("simulated failure mid-candidate")
    assert tm.CHANDELIER_MULTIPLIER_BY_REGIME == original_by_regime
    assert tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK == original_lock


# --- sample_tickers: reproducible, bounded, per-region -----------------------


def test_sample_tickers_is_reproducible_with_the_same_seed():
    first = ccs.sample_tickers(["us"], n_per_region=10, seed=3)
    second = ccs.sample_tickers(["us"], n_per_region=10, seed=3)
    assert first == second


def test_sample_tickers_respects_the_per_region_size():
    result = ccs.sample_tickers(["us", "europe"], n_per_region=5, seed=1)
    us_count = sum(1 for region in result.values() if region == "us")
    europe_count = sum(1 for region in result.values() if region == "europe")
    assert us_count == 5
    assert europe_count == 5


# --- trailing_labels_for_ticker: reuses find_triple_barrier_entries + label_triple_barrier ---


def test_trailing_labels_for_ticker_empty_when_no_comprar_signals_fire():
    n = 1500
    closes = 500 - np.arange(n) * 0.2  # unbroken downtrend - should never replay "comprar"
    df = _df(closes)
    assert ccs.trailing_labels_for_ticker(df, horizon_days=21, vol_regime="normal") == []


def test_trailing_labels_for_ticker_produces_labels_on_an_uptrend():
    n = 1500
    closes = 100 + np.cumsum(np.full(n, 0.15))  # steady, unbroken uptrend
    df = _df(closes)
    labels = ccs.trailing_labels_for_ticker(df, horizon_days=21, vol_regime="normal")
    assert len(labels) > 0
    assert all(isinstance(label, type(labels[0])) for label in labels)


# --- build_report: one row per candidate, from pooled labels ----------------


def test_build_report_has_one_row_per_candidate_with_pooled_trade_count():
    n = 1500
    closes = 100 + np.cumsum(np.full(n, 0.15))
    df = _df(closes)
    labels_a = ccs.trailing_labels_for_ticker(df, horizon_days=21, vol_regime="normal")
    labels_b = ccs.trailing_labels_for_ticker(df, horizon_days=21, vol_regime="alta")
    rows_by_candidate = {"candidate_a": labels_a, "candidate_b": labels_b + labels_a}

    report = ccs.build_report(rows_by_candidate)

    assert set(report["candidate"]) == {"candidate_a", "candidate_b"}
    row_a = report[report["candidate"] == "candidate_a"].iloc[0]
    row_b = report[report["candidate"] == "candidate_b"].iloc[0]
    assert row_a["n_trades"] == len(labels_a)
    assert row_b["n_trades"] == len(labels_b) + len(labels_a)


def test_build_report_empty_candidate_has_zero_trades_not_a_crash():
    report = ccs.build_report({"empty": []})
    assert report.iloc[0]["n_trades"] == 0
    assert pd.isna(report.iloc[0]["win_rate"])
