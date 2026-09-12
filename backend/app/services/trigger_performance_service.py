"""Fase 8 (docs/quant_methodology.md §25): "¿está funcionando el sistema?"
(Parte 0, pregunta 4) measured against the new gate/trigger vocabulary
instead of the retired checklist's verdict/signal labels -
`signal_performance_service.py` keeps measuring those (still meaningful for
snapshots recorded before the Fase 4 cutover, and it says so in its own
Fase 8 forward pointer) - this module is the new primary read, built on
`TriggerEvent` (Fase 2's own append-only "what changed" log) instead of a
per-request "Analizar activo" audit trail.

Two event types worth measuring separately, both hypotheses about whether
the gate design has any real predictive value at all - not assumed true:

- `gate_passed`: did price rise in the sessions after the gate flipped from
  failing to passing?
- `entry_triggered`: did price rise in the sessions after the entry trigger
  price was actually crossed (a later, more specific moment than the gate
  merely turning on)?

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
call across every distinct ticker."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from app.domain.models.trigger_event import TriggerEvent
from app.services import signal_performance_service as sps
from app.services.market_data_service import MarketDataService

FORWARD_HORIZONS = (5, 10, 21, 63)  # trading sessions - same as signal_performance_service.py

MEASURED_EVENT_TYPES = ("gate_passed", "entry_triggered")


@dataclass(frozen=True, slots=True)
class TriggerOutcomeStats:
    event_type: str
    horizon_days: int
    n: int
    hit_rate: float | None  # fraction of observations with a positive forward return
    mean_return: float | None
    median_return: float | None


@dataclass(frozen=True, slots=True)
class TriggerPerformanceReport:
    outcomes: list[TriggerOutcomeStats]
    as_of: datetime


def compute_trigger_outcomes(
    events: list[TriggerEvent], price_by_ticker: dict[str, pd.Series]
) -> list[TriggerOutcomeStats]:
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for event in events:
        if event.entity_type != "ticker" or event.event_type not in MEASURED_EVENT_TYPES:
            continue
        close = price_by_ticker.get(event.entity_key)
        if close is None:
            continue
        event_date = event.occurred_at.date()
        for horizon in FORWARD_HORIZONS:
            ret = sps.forward_return(close, event_date, horizon)
            if ret is not None:
                by_key[(event.event_type, horizon)].append(ret)

    outcomes = []
    for (event_type, horizon), returns in sorted(by_key.items()):
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
            )
        )
    return outcomes


def build_trigger_performance_report(
    events: list[TriggerEvent], market_data: MarketDataService
) -> TriggerPerformanceReport:
    """Fetches price history for every distinct ticker behind a measured
    event in a single batched call (never one network round-trip per
    ticker), then delegates to `compute_trigger_outcomes` above."""
    measured = [e for e in events if e.entity_type == "ticker" and e.event_type in MEASURED_EVENT_TYPES]
    tickers = sorted({e.entity_key for e in measured})
    if not tickers:
        return TriggerPerformanceReport([], datetime.now(UTC))

    start = min(e.occurred_at for e in measured).date()
    end = datetime.now(UTC).date()
    ohlcv = market_data.get_bulk_ohlcv(tickers, start, end)
    price_by_ticker = {ticker: frame["close"] for ticker, frame in ohlcv.items()}

    return TriggerPerformanceReport(
        outcomes=compute_trigger_outcomes(measured, price_by_ticker),
        as_of=datetime.now(UTC),
    )
