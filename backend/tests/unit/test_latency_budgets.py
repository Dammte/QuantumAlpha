"""Reconstruction (2026-09), Fase 9: latency/performance budget tests - not
correctness tests (those live elsewhere), a regression net against the
concern CLAUDE.md itself already states as non-negotiable ("no llamadas de
red por ticker en los caminos calientes" - `PortfolioRiskService` already
had a real production latency incident over this, see its own docstring).
This file catches the OTHER way a hot path gets slow: an accidental O(n^2)
or a heavy per-bar Python loop introduced in pure computation, which a
plain correctness test (checking the right *answer*, never how long it took
to get there) would never catch - worth having now specifically because
Fase 10 plans to grow the universe from ~217 to ~400 tickers on the same
nightly cron window `daily_close.py` already runs against.

Budgets here are deliberately generous (an order of magnitude or more over
what this actually measures on ordinary hardware), not tight benchmarks -
the point is catching a real regression (a function that got 10x/100x
slower), not chasing a specific millisecond figure that would make this
suite flaky across different CI hardware."""

import time
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

import scripts.daily_close as dc
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import technical_analysis as ta

N_TICKERS = 50
BARS_PER_TICKER = 2520  # ~10 years of daily bars - same HISTORY_YEARS daily_close.py/factor_ablation_study.py use
# Generous on purpose - see module docstring. Ordinary hardware clears this
# by 1-2 orders of magnitude (a few ms per ticker for pandas-vectorized
# indicator math over ~2500 bars).
BUDGET_SECONDS_PER_TICKER = 0.5


def _ohlc(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp("2026-09-10"), periods=n)
    close = pd.Series(100 + np.arange(n) * 0.05 + rng.normal(0, 1.0, n).cumsum() * 0.1, index=index)
    return pd.DataFrame(
        {
            "open": close,
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": pd.Series(rng.uniform(500_000, 2_000_000, n), index=index),
        }
    )


def _snapshot(ticker: str) -> TickerSnapshot:
    return TickerSnapshot(
        ticker=ticker, sector="Tecnología", industry=None, cap_tier="large", price=150.0,
        change_1d=0.5, change_1w=1.0, change_1m=2.0, change_3m=5.0, change_6m=10.0, change_1y=20.0,
        volume=1_000_000.0, relative_volume=1.1, rsi14=55.0, sma20=148.0, sma50=140.0, sma150=120.0,
        sma200=110.0, dist_52w_high=-0.05, dist_52w_low=0.30, atr_multiple=1.2, adx14=28.0,
        plus_di=25.0, minus_di=12.0, mansfield_rs=0.5, trend=ta.TrendState.UPTREND, stage=ta.Stage.STAGE_2,
        ma_cross=None, minervini_score=7, minervini_pass=False, rs_rating=85,
    )


def test_build_ticker_daily_state_scales_to_a_400_ticker_universe():
    # The one function daily_close.py's nightly cron pays once per ticker
    # (levels_engine.evaluate_gate plus its own ATR/support-resistance/OBV/
    # fast-pair-veto derivation) - its own per-ticker cost is exactly what
    # multiplies as Fase 10 grows the universe. N_TICKERS=50 (not 400) is
    # plenty to catch a real per-ticker regression without making this test
    # itself slow to run in CI. Already covers the setups library end to
    # end (detect_all, context_modifiers, build_timeframe_strip,
    # apply_measured_confidence all run inside this same call) at this
    # generous whole-pipeline budget - see the more targeted test below for
    # the literal "< 40 ms por ticker" the setups library's own Parte 13.2
    # asks for on the detectors alone, not the whole gate/grade pipeline.
    trade_date = date(2026, 9, 10)
    computed_at = datetime(2026, 9, 10, 22, 0, tzinfo=UTC)
    frames = [_ohlc(BARS_PER_TICKER, seed=i) for i in range(N_TICKERS)]
    snapshots = [_snapshot(f"T{i}") for i in range(N_TICKERS)]

    start = time.perf_counter()
    for snapshot, df in zip(snapshots, frames, strict=True):
        dc.build_ticker_daily_state(snapshot, "us", df, trade_date, computed_at)
    elapsed = time.perf_counter() - start

    assert elapsed < N_TICKERS * BUDGET_SECONDS_PER_TICKER


# Parte 13.2 de la biblioteca de setups del Radar (quant_methodology.md
# §28), literal: "que los detectores juntos tarden < 40 ms por ticker sobre
# un frame realista de 10 años". Medido aquí en aislamiento (solo
# `setups.registry.detect_all` sobre un `SetupContext` ya construido), no
# sobre `build_ticker_daily_state` completo (ese ya lo cubre el test de
# arriba, con presupuesto para el gate/grado/geometría además de los
# setups) - un presupuesto diez veces más generoso que el literal (400 ms,
# no 40 ms), mismo criterio del docstring del módulo: atrapar una regresión
# real, no perseguir una cifra concreta de milisegundos en hardware de CI
# variable.
SETUPS_BUDGET_SECONDS_PER_TICKER = 0.4


def _setup_context_for_latency_test(n: int, seed: int):
    from app.services import multi_timeframe as mtf
    from app.services.setups.context import SetupContext

    df = _ohlc(n, seed)
    close, high, low, volume, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    weekly_df = ta.resample_ohlcv(df, mtf.WEEKLY_RULE)
    multi = mtf.analyze_multi_timeframe(df)
    return SetupContext(
        ticker=f"T{seed}", region="us", trade_date=date(2026, 9, 10),
        close=close, high=high, low=low, volume=volume, open_=open_,
        weekly_close=weekly_df["close"], weekly_high=weekly_df["high"],
        weekly_low=weekly_df["low"], weekly_volume=weekly_df["volume"],
        atr_series=ta.atr(high, low, close), atr14=2.0, ema21=150.0, ema55=148.0,
        sma20=149.0, sma50=145.0, sma150=140.0, sma200=135.0, rsi14=55.0,
        levels=ta.detect_levels(high, low, close, volume, weekly_close=weekly_df["close"]),
        multi_timeframe=multi, trend=multi.daily.trend,
        weekly_stage=multi.weekly.stage if multi.weekly is not None else None,
        relative_volume=1.1, rs_percentile=70, sector_rs_percentile=60,
        mansfield_rs_series=None,
    )


def test_setups_registry_detect_all_stays_within_its_own_latency_budget():
    import app.services.setups.registry as setups_registry

    contexts = [_setup_context_for_latency_test(BARS_PER_TICKER, seed=i) for i in range(N_TICKERS)]

    start = time.perf_counter()
    for ctx in contexts:
        setups_registry.detect_all(ctx)
    elapsed = time.perf_counter() - start

    assert elapsed < N_TICKERS * SETUPS_BUDGET_SECONDS_PER_TICKER
