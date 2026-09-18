"""Fase 8 (docs/quant_methodology.md §25): "¿está funcionando el sistema?"
(Parte 0, pregunta 4) measured against the new gate/trigger vocabulary
instead of the retired checklist's verdict/signal labels -
`signal_performance_service.py` keeps measuring those (still meaningful for
snapshots recorded before the Fase 4 cutover, and it says so in its own
Fase 8 forward pointer) - this module is the new primary read, built on
`TriggerEvent` (Fase 2's own append-only "what changed" log) instead of a
per-request "Analizar activo" audit trail.

Three event types worth measuring separately, all hypotheses about whether
the gate/setup-library design has any real predictive value at all - not
assumed true:

- `gate_passed`: did price rise in the sessions after the gate flipped from
  failing to passing?
- `entry_triggered`: did price rise in the sessions after the entry trigger
  price was actually crossed (a later, more specific moment than the gate
  merely turning on)?
- `setup_triggered` (biblioteca de setups, §28.1's internal Fase 10 - "taken
  derivado contra transacciones reales, nunca persistido, misma decisión que
  este módulo ya tomó"): did price rise after a specific named setup
  (`daily_close.ticker_trigger_events`) reached `SetupStage.TRIGGERED`? A
  distinct, more granular question than `entry_triggered` - the gate's own
  generic trigger vs. a specific pattern (VCP, breakout...) confirming -
  answered here for free by reusing the exact same `TriggerEvent`/
  `compute_trigger_outcomes` machinery, no new aggregation needed.

`exit_urgency_changed` (position-level) is deliberately excluded - it's
about an already-open position's own exit engine reacting, not a fresh
entry signal, and blending exit-side and entry-side base rates into one hit
rate would answer neither question honestly (the same reasoning
`BEARISH_SIGNAL_LABELS` encodes in `signal_performance_service.py`).

Pure aggregation (`compute_trigger_outcomes`) takes plain `TriggerEvent`
objects and a dict of price series - no DB, no network - so it's
unit-tested with hand-built synthetic events, same shape as
`signal_performance_service.py`. `build_trigger_performance_report` is the
one orchestration function that fetches price history, in a single batched
call across every distinct ticker.

**Parte 13 - "qué pasó con lo que no compraste"**: `TriggerEvent` has no
persisted `taken` flag the way the brief's own `trigger_history` schema
does - whether an `entry_triggered`/`setup_triggered` event was actually
acted on is derived here instead, from the portfolios' own real
`Transaction` history (a BUY of that ticker within `TAKEN_WINDOW_DAYS` of
the trigger), rather than adding a column nothing else would ever write to.
`buy_dates_by_ticker` is optional everywhere below (`None` skips the
comparison entirely, same as before this existed) precisely so this stays
meaningful only for `TAKEN_ELIGIBLE_EVENT_TYPES` - a mere `gate_passed`
isn't something the propietario acts on directly, so "taken" isn't a
coherent question to ask of it."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from app.domain.models.trigger_event import TriggerEvent
from app.services import signal_performance_service as sps
from app.services.market_data_service import MarketDataService

FORWARD_HORIZONS = (5, 10, 21, 63)  # trading sessions - same as signal_performance_service.py

MEASURED_EVENT_TYPES = ("gate_passed", "entry_triggered", "setup_triggered")

# De los tipos medidos, en cuáles tiene sentido preguntar "¿se tomó de
# verdad?" (Parte 13) - un `gate_passed` es un estado, no una acción
# concreta sobre la que el propietario decide comprar o no.
TAKEN_ELIGIBLE_EVENT_TYPES = ("entry_triggered", "setup_triggered")

# Calendar days, not trading sessions - a generous window past the primary
# 5-session horizon (covers weekends/holidays) for "did a BUY follow this
# trigger", not a precise session count the way FORWARD_HORIZONS is.
TAKEN_WINDOW_DAYS = 10


@dataclass(frozen=True, slots=True)
class TriggerOutcomeStats:
    event_type: str
    horizon_days: int
    n: int
    hit_rate: float | None  # fraction of observations with a positive forward return
    mean_return: float | None
    median_return: float | None
    # Parte 13: None for the combined row (every measured event, same as
    # before this field existed) - True/False only for `entry_triggered`
    # rows split by whether a real BUY followed within TAKEN_WINDOW_DAYS,
    # and only when the caller actually supplied transaction data to split
    # against (see `compute_trigger_outcomes`'s own docstring).
    taken: bool | None = None


def _was_taken(
    ticker: str, trigger_date: date, buy_dates_by_ticker: dict[str, list[date]], window_days: int
) -> bool:
    buy_dates = buy_dates_by_ticker.get(ticker)
    if not buy_dates:
        return False
    window_end = trigger_date + timedelta(days=window_days)
    return any(trigger_date <= buy_date <= window_end for buy_date in buy_dates)


@dataclass(frozen=True, slots=True)
class TriggerPerformanceReport:
    outcomes: list[TriggerOutcomeStats]
    as_of: datetime


def compute_trigger_outcomes(
    events: list[TriggerEvent],
    price_by_ticker: dict[str, pd.Series],
    buy_dates_by_ticker: dict[str, list[date]] | None = None,
) -> list[TriggerOutcomeStats]:
    """`buy_dates_by_ticker` (Parte 13, optional - `None` reproduces the
    exact pre-existing behavior, one combined row per event type/horizon)
    additionally splits every row whose event type is in
    `TAKEN_ELIGIBLE_EVENT_TYPES` into a `taken=True` and a `taken=False`
    row, each measured against only its own subset of events - "did the
    triggers you actually acted on outperform the ones you skipped" is a
    different, real question from the combined hit rate above it, not a
    refinement of the same number."""
    by_key: dict[tuple[str, int, bool | None], list[float]] = defaultdict(list)
    for event in events:
        if event.entity_type != "ticker" or event.event_type not in MEASURED_EVENT_TYPES:
            continue
        close = price_by_ticker.get(event.entity_key)
        if close is None:
            continue
        event_date = event.occurred_at.date()
        taken = None
        if buy_dates_by_ticker is not None and event.event_type in TAKEN_ELIGIBLE_EVENT_TYPES:
            taken = _was_taken(event.entity_key, event_date, buy_dates_by_ticker, TAKEN_WINDOW_DAYS)
        for horizon in FORWARD_HORIZONS:
            ret = sps.forward_return(close, event_date, horizon)
            if ret is None:
                continue
            by_key[(event.event_type, horizon, None)].append(ret)
            if taken is not None:
                by_key[(event.event_type, horizon, taken)].append(ret)

    def _sort_key(item: tuple[tuple[str, int, bool | None], list[float]]) -> tuple:
        (event_type, horizon, taken), _ = item
        return (event_type, horizon, taken is not None, taken or False)

    outcomes = []
    for (event_type, horizon, taken), returns in sorted(by_key.items(), key=_sort_key):
        n = len(returns)
        hit_rate = sum(1 for r in returns if r > 0) / n
        outcomes.append(
            TriggerOutcomeStats(
                event_type=event_type,
                horizon_days=horizon,
                n=n,
                hit_rate=hit_rate,
                mean_return=float(np.mean(returns)),
                median_return=float(np.median(returns)),
                taken=taken,
            )
        )
    return outcomes


def build_trigger_performance_report(
    events: list[TriggerEvent],
    market_data: MarketDataService,
    buy_dates_by_ticker: dict[str, list[date]] | None = None,
) -> TriggerPerformanceReport:
    """Fetches price history for every distinct ticker behind a measured
    event in a single batched call (never one network round-trip per
    ticker), then delegates to `compute_trigger_outcomes` above.
    `buy_dates_by_ticker` (Parte 13) is passed straight through - see that
    function's own docstring; the caller (the API endpoint, which already
    has portfolio access) builds it from every portfolio's real BUY
    transactions, not something this orchestration function fetches itself
    (this module has no portfolio dependency otherwise, and shouldn't gain
    one just for this)."""
    measured = [e for e in events if e.entity_type == "ticker" and e.event_type in MEASURED_EVENT_TYPES]
    tickers = sorted({e.entity_key for e in measured})
    if not tickers:
        return TriggerPerformanceReport([], datetime.now(UTC))

    start = min(e.occurred_at for e in measured).date()
    end = datetime.now(UTC).date()
    ohlcv = market_data.get_bulk_ohlcv(tickers, start, end)
    price_by_ticker = {ticker: frame["close"] for ticker, frame in ohlcv.items()}

    return TriggerPerformanceReport(
        outcomes=compute_trigger_outcomes(measured, price_by_ticker, buy_dates_by_ticker),
        as_of=datetime.now(UTC),
    )
