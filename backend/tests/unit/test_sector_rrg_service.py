"""Cuarta auditoría independiente, Bloque E: unit tests for sector_rrg_service.py
(Relative Rotation Graph reconstruction, explicitly requested, previously
flagged as pending across several rounds)."""

import numpy as np
import pandas as pd
import pytest

from app.services import sector_rrg_service as rrg


def _close(values, n=None) -> pd.Series:
    if n is None:
        n = len(values)
    index = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(values, index=index)


def _df(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"close": close})


# --- compute_rs_ratio_and_momentum: warmup, alignment, quadrant math --------


def test_compute_rs_ratio_and_momentum_none_with_too_little_history():
    n = 100  # well under MIN_OBSERVATIONS
    sector = _close(100 + np.arange(n) * 0.1, n)
    benchmark = _close(100 + np.arange(n) * 0.05, n)
    assert rrg.compute_rs_ratio_and_momentum(sector, benchmark) is None


def test_compute_rs_ratio_and_momentum_outperformer_ends_above_100():
    n = 600
    rng = np.random.RandomState(1)
    # Sector steadily pulls ahead of the benchmark - a genuine, sustained
    # outperformance, not just noise.
    benchmark = _close(100 + np.cumsum(rng.normal(0, 0.3, n)), n)
    sector = _close(benchmark.values * (1 + np.linspace(0, 0.4, n)), n)
    result = rrg.compute_rs_ratio_and_momentum(sector, benchmark)
    assert result is not None
    rs_ratio, rs_momentum = result
    assert rs_ratio.iloc[-1] > 100


def test_compute_rs_ratio_and_momentum_underperformer_ends_below_100():
    n = 600
    rng = np.random.RandomState(2)
    benchmark = _close(100 + np.cumsum(rng.normal(0, 0.3, n)), n)
    sector = _close(benchmark.values * (1 - np.linspace(0, 0.4, n)), n)
    result = rrg.compute_rs_ratio_and_momentum(sector, benchmark)
    assert result is not None
    rs_ratio, _ = result
    assert rs_ratio.iloc[-1] < 100


def test_compute_rs_ratio_and_momentum_returns_aligned_series():
    n = 600
    rng = np.random.RandomState(3)
    benchmark = _close(100 + np.cumsum(rng.normal(0, 0.3, n)), n)
    sector = _close(100 + np.cumsum(rng.normal(0, 0.35, n)), n)
    result = rrg.compute_rs_ratio_and_momentum(sector, benchmark)
    assert result is not None
    rs_ratio, rs_momentum = result
    assert list(rs_ratio.index) == list(rs_momentum.index)
    assert len(rs_ratio) > 0


def test_compute_rs_ratio_and_momentum_none_when_ratio_is_perfectly_flat():
    # A zero-variance rolling window (sector and benchmark move in lockstep
    # the whole time) makes the rolling std exactly 0 - guarded to NaN (and
    # therefore dropped), never left as a raw `inf` from dividing by zero.
    n = 600
    benchmark = _close(100 + np.cumsum(np.full(n, 0.1)), n)
    sector = _close(benchmark.values * 1.0, n)  # identical ratio every single day
    assert rrg.compute_rs_ratio_and_momentum(sector, benchmark) is None


# --- _quadrant: the four-way split, boundary at exactly 100 -----------------


@pytest.mark.parametrize(
    "rs_ratio,rs_momentum,expected",
    [
        (105, 105, rrg.QUADRANT_LEADING),
        (100, 100, rrg.QUADRANT_LEADING),  # >=100 on both axes, the inclusive boundary case
        (105, 95, rrg.QUADRANT_WEAKENING),
        (95, 95, rrg.QUADRANT_LAGGING),
        (95, 105, rrg.QUADRANT_IMPROVING),
    ],
)
def test_quadrant_boundaries(rs_ratio, rs_momentum, expected):
    assert rrg._quadrant(rs_ratio, rs_momentum) == expected


# --- compute_sector_rrg: orchestration, missing data, sorting ---------------


def _trending_pair(seed, drift_sector, drift_benchmark, n=600):
    rng = np.random.RandomState(seed)
    benchmark = _close(100 + np.cumsum(rng.normal(drift_benchmark, 0.3, n)), n)
    sector = _close(100 + np.cumsum(rng.normal(drift_sector, 0.3, n)), n)
    return sector, benchmark


def test_compute_sector_rrg_sorts_by_rs_ratio_descending():
    benchmark = _trending_pair(10, 0.0, 0.05)[1]
    strong = _trending_pair(11, 0.35, 0.0)[0]
    weak = _trending_pair(12, -0.35, 0.0)[0]
    ohlcv_by_ticker = {"SPY": _df(benchmark), "XLK": _df(strong), "XLU": _df(weak)}
    sector_etfs = {"Tecnología": "XLK", "Utilities": "XLU"}

    readings = rrg.compute_sector_rrg(sector_etfs, ohlcv_by_ticker, "SPY")

    assert [r.sector for r in readings] == ["Tecnología", "Utilities"]
    assert readings[0].rs_ratio > readings[1].rs_ratio


def test_compute_sector_rrg_skips_a_sector_with_no_ohlcv_data():
    benchmark = _trending_pair(20, 0.0, 0.05)[1]
    strong = _trending_pair(21, 0.35, 0.0)[0]
    ohlcv_by_ticker = {"SPY": _df(benchmark), "XLK": _df(strong)}
    # "Energía" -> "XLE" has no entry in ohlcv_by_ticker at all.
    sector_etfs = {"Tecnología": "XLK", "Energía": "XLE"}

    readings = rrg.compute_sector_rrg(sector_etfs, ohlcv_by_ticker, "SPY")

    assert [r.sector for r in readings] == ["Tecnología"]


def test_compute_sector_rrg_empty_when_benchmark_missing():
    strong = _trending_pair(30, 0.35, 0.0)[0]
    ohlcv_by_ticker = {"XLK": _df(strong)}
    assert rrg.compute_sector_rrg({"Tecnología": "XLK"}, ohlcv_by_ticker, "SPY") == []


def test_compute_sector_rrg_tail_has_at_most_rrg_tail_length_points_oldest_first():
    benchmark = _trending_pair(40, 0.0, 0.05)[1]
    sector = _trending_pair(41, 0.2, 0.0)[0]
    ohlcv_by_ticker = {"SPY": _df(benchmark), "XLK": _df(sector)}

    readings = rrg.compute_sector_rrg({"Tecnología": "XLK"}, ohlcv_by_ticker, "SPY")

    assert len(readings) == 1
    tail = readings[0].tail
    assert 0 < len(tail) <= rrg.RRG_TAIL_LENGTH
    assert tail == sorted(tail, key=lambda p: p.as_of)  # oldest first
    assert tail[-1].rs_ratio == pytest.approx(readings[0].rs_ratio)
    assert tail[-1].rs_momentum == pytest.approx(readings[0].rs_momentum)


def test_compute_sector_rrg_quadrant_matches_the_readings_own_axes():
    benchmark = _trending_pair(50, 0.0, 0.05)[1]
    leader = _trending_pair(51, 0.4, 0.0)[0]
    ohlcv_by_ticker = {"SPY": _df(benchmark), "XLK": _df(leader)}

    readings = rrg.compute_sector_rrg({"Tecnología": "XLK"}, ohlcv_by_ticker, "SPY")

    assert len(readings) == 1
    reading = readings[0]
    assert reading.quadrant == rrg._quadrant(reading.rs_ratio, reading.rs_momentum)
