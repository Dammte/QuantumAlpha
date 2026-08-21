"""Answers, with numbers instead of impressions, the question Fase 0 exists
for: did the system's past verdicts and position signals actually work out.

Before this module existed, `RecommendationSnapshotORM` rows were written on
every real "Analizar activo" call and then only ever read back one ticker at
a time (`GET /market/tickers/{ticker}/history`) - nobody aggregated them into
a hit rate, a mean forward return, or a list of the worst misses. This is
that aggregation, plus (since Fase 0) the position-level signal history
`RecommendationSnapshotORM` never captured at all (see
`PositionSignalSnapshotORM`'s docstring for why that data literally didn't
exist before now, and can only be measured going forward).

Pure aggregation functions (`compute_verdict_outcomes`, `compute_signal_outcomes`,
`find_false_negatives`, `forward_return`) take plain domain objects and a
dict of price series - no DB, no network - so they're unit-tested with
hand-built synthetic snapshots. `build_signal_performance_report` is the one
orchestration function that actually fetches price history (a single batched
call across every distinct ticker that appears in the snapshots, never one
call per ticker).
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from app.domain.models.position_signal_snapshot import PositionSignalSnapshot
from app.domain.models.recommendation_snapshot import RecommendationSnapshot
from app.services.market_data_service import MarketDataService

FORWARD_HORIZONS = (5, 10, 21, 63)  # trading sessions
FALSE_NEGATIVE_HORIZON_DAYS = 10
FALSE_NEGATIVE_DROP_THRESHOLD = -0.05  # -5%
FALSE_NEGATIVE_SIGNAL = "hold"


@dataclass(frozen=True, slots=True)
class OutcomeStats:
    label: str  # the verdict or signal string this row summarizes
    horizon_days: int
    n: int
    hit_rate: float | None  # fraction of observations with a positive forward return - see module docstring
    mean_return: float | None
    median_return: float | None


@dataclass(frozen=True, slots=True)
class FalseNegative:
    """A `hold` immediately followed by a real drawdown - the concrete,
    by-name list of exactly the false negatives that motivated Fase 0:
    "¿cuántas veces el sistema dijo hold y el activo cayó?"."""

    portfolio_id: int
    ticker: str
    snapshot_at: datetime
    price_at_signal: float
    price_after: float
    return_pct: float
    horizon_days: int


@dataclass(frozen=True, slots=True)
class SignalPerformanceReport:
    verdict_outcomes: list[OutcomeStats]
    signal_outcomes: list[OutcomeStats]
    false_negatives: list[FalseNegative]
    as_of: datetime


def forward_return(close: pd.Series, snapshot_date: date, horizon_days: int) -> float | None:
    """% change from the close on/just after `snapshot_date` to the close
    `horizon_days` *trading* sessions later - `None` if `close` doesn't cover
    `snapshot_date` at all, or doesn't yet reach `horizon_days` sessions past
    it (a recent snapshot whose forward window hasn't happened yet)."""
    if close.empty:
        return None
    dates = close.index
    on_or_after = dates[dates.date >= snapshot_date] if hasattr(dates, "date") else None
    if on_or_after is None or len(on_or_after) == 0:
        return None
    start_pos = dates.get_loc(on_or_after[0])
    if isinstance(start_pos, slice):  # duplicate index labels - take the first occurrence
        start_pos = start_pos.start
    target_pos = start_pos + horizon_days
    if target_pos >= len(dates):
        return None
    start_price = float(close.iloc[start_pos])
    if start_price == 0:
        return None
    end_price = float(close.iloc[target_pos])
    return end_price / start_price - 1


# Labels that are themselves a call to *avoid or exit* a ticker - "hit" for
# these means the market went the other way (a negative subsequent return
# vindicates the call), the mirror of every other label. Segunda auditoría,
# Bloque 1: the original "always fraction-positive" definition was picked
# deliberately, as one single, uniform reading meant to stay neutral rather
# than flip criteria per category - but a bearish label with a 65% hit_rate
# under that definition means the stock rose 65% of the time anyway, the
# *opposite* of vindication, and nothing in the UI distinguished the two
# readings. `mean_return`/`median_return` stay raw, unsigned either way -
# only which side of zero counts as "hit" changes.
BEARISH_VERDICT_LABELS = frozenset({"evitar"})
BEARISH_SIGNAL_LABELS = frozenset({"exit_warning"})


def _stats_for(
    returns: list[float], bearish: bool = False
) -> tuple[int, float | None, float | None, float | None]:
    n = len(returns)
    if n == 0:
        return 0, None, None, None
    hit_rate = sum(1 for r in returns if (r < 0 if bearish else r > 0)) / n
    return n, hit_rate, float(np.mean(returns)), float(np.median(returns))


def _deduplicate_latest_per_ticker_and_day(snapshots: list) -> list:
    """One snapshot per (ticker, calendar day) - the most recent of that
    day's - before anything gets aggregated. Without this, every dashboard
    reload or manual "Actualizar ahora" that lands on a cache miss appends
    another observation for the exact same ticker/day, so `n` measures how
    often the page got reloaded, not how many genuinely distinct calls the
    system made."""
    latest_by_key: dict[tuple[str, object], object] = {}
    for snap in snapshots:
        key = (snap.ticker, snap.created_at.date())
        existing = latest_by_key.get(key)
        if existing is None or snap.created_at > existing.created_at:
            latest_by_key[key] = snap
    return list(latest_by_key.values())


def compute_verdict_outcomes(
    snapshots: list[RecommendationSnapshot], price_by_ticker: dict[str, pd.Series]
) -> list[OutcomeStats]:
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for snap in _deduplicate_latest_per_ticker_and_day(snapshots):
        close = price_by_ticker.get(snap.ticker)
        if close is None:
            continue
        snap_date = snap.created_at.date()
        for horizon in FORWARD_HORIZONS:
            ret = forward_return(close, snap_date, horizon)
            if ret is not None:
                by_key[(snap.verdict, horizon)].append(ret)

    outcomes = []
    for (verdict, horizon), returns in sorted(by_key.items()):
        n, hit_rate, mean_r, median_r = _stats_for(returns, bearish=verdict in BEARISH_VERDICT_LABELS)
        outcomes.append(
            OutcomeStats(label=verdict, horizon_days=horizon, n=n, hit_rate=hit_rate, mean_return=mean_r,
                         median_return=median_r)
        )
    return outcomes


def compute_signal_outcomes(
    snapshots: list[PositionSignalSnapshot], price_by_ticker: dict[str, pd.Series]
) -> list[OutcomeStats]:
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for snap in _deduplicate_latest_per_ticker_and_day(snapshots):
        close = price_by_ticker.get(snap.ticker)
        if close is None:
            continue
        snap_date = snap.created_at.date()
        for horizon in FORWARD_HORIZONS:
            ret = forward_return(close, snap_date, horizon)
            if ret is not None:
                by_key[(snap.signal, horizon)].append(ret)

    outcomes = []
    for (signal, horizon), returns in sorted(by_key.items()):
        n, hit_rate, mean_r, median_r = _stats_for(returns, bearish=signal in BEARISH_SIGNAL_LABELS)
        outcomes.append(
            OutcomeStats(label=signal, horizon_days=horizon, n=n, hit_rate=hit_rate, mean_return=mean_r,
                         median_return=median_r)
        )
    return outcomes


def find_false_negatives(
    snapshots: list[PositionSignalSnapshot], price_by_ticker: dict[str, pd.Series]
) -> list[FalseNegative]:
    """Every `hold` immediately followed by a drop of more than 5% within 10
    sessions - listed by ticker and date, not just counted, so each one can
    actually be looked at."""
    results = []
    for snap in _deduplicate_latest_per_ticker_and_day(snapshots):
        if snap.signal != FALSE_NEGATIVE_SIGNAL:
            continue
        close = price_by_ticker.get(snap.ticker)
        if close is None:
            continue
        ret = forward_return(close, snap.created_at.date(), FALSE_NEGATIVE_HORIZON_DAYS)
        if ret is None or ret > FALSE_NEGATIVE_DROP_THRESHOLD:
            continue
        price_after = snap.price * (1 + ret)
        results.append(
            FalseNegative(
                portfolio_id=snap.portfolio_id,
                ticker=snap.ticker,
                snapshot_at=snap.created_at,
                price_at_signal=snap.price,
                price_after=price_after,
                return_pct=ret,
                horizon_days=FALSE_NEGATIVE_HORIZON_DAYS,
            )
        )
    return results


def build_signal_performance_report(
    recommendation_snapshots: list[RecommendationSnapshot],
    position_snapshots: list[PositionSignalSnapshot],
    market_data: MarketDataService,
) -> SignalPerformanceReport:
    """The one orchestration function here: fetches price history for every
    distinct ticker across both snapshot sets in a single batched call
    (never one network round-trip per ticker), then delegates to the pure
    aggregation functions above."""
    all_dates = [s.created_at for s in recommendation_snapshots] + [s.created_at for s in position_snapshots]
    tickers = sorted({s.ticker for s in recommendation_snapshots} | {s.ticker for s in position_snapshots})
    if not tickers:
        return SignalPerformanceReport([], [], [], datetime.now(UTC))

    start = min(d.date() for d in all_dates)
    end = date.today()
    ohlcv = market_data.get_bulk_ohlcv(tickers, start, end)
    price_by_ticker = {ticker: frame["close"] for ticker, frame in ohlcv.items()}

    return SignalPerformanceReport(
        verdict_outcomes=compute_verdict_outcomes(recommendation_snapshots, price_by_ticker),
        signal_outcomes=compute_signal_outcomes(position_snapshots, price_by_ticker),
        false_negatives=find_false_negatives(position_snapshots, price_by_ticker),
        as_of=datetime.now(UTC),
    )
