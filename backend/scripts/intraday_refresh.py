"""Reconstruction (2026-09), Fase 2, Job B: cheap, frequent intraday
companion to `daily_close.py` (Job A) - re-checks a live quote against
*yesterday's* already-computed entry trigger (`TickerDailyState.
entry_trigger_price`, from last night's close) without recomputing the full
levels/triggers gate mid-session. See `trade_geometry.compute_entry_trigger`'s
own docstring for why `already_triggered` was designed around exactly this
"stale levels, fresh price" use case from the start.

Only re-checks tickers that actually have an active entry trigger from the
last close - most of the universe doesn't, on any given day, so this stays a
small, fast quote-only fetch (`MarketDataService.get_latest_quotes`, no OHLCV
download) even though `daily_close.py`'s own per-ticker computation is much
heavier. Run every 15-30 minutes during market hours (a Render Cron Job -
see `render.yaml`), never replacing Job A's own nightly full recompute.

Only the "breakout" trigger type can meaningfully flip intraday: a
"pullback_bounce" trigger is already `already_triggered=True` the moment
`daily_close.py` computes it (see `trade_geometry.compute_entry_trigger`'s
own priority rule), so there's nothing left to re-check for it here -
carried forward unchanged.

Usage (same `python -m scripts.X` convention as daily_close.py - see that
script's own docstring for why):
    python -m scripts.intraday_refresh
    python -m scripts.intraday_refresh --region us
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_intraday_state import TickerIntradayState
from app.domain.models.trigger_event import TriggerEvent
from app.infrastructure.db.repositories.job_run_repository import JobRunRepository
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.infrastructure.db.repositories.ticker_intraday_state_repository import TickerIntradayStateRepository
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

REGIONS = ("us", "europe")


def compute_intraday_state(state: TickerDailyState, price: float, updated_at: datetime) -> TickerIntradayState:
    """Pure function: `state` is last night's `TickerDailyState` for this
    ticker (guaranteed by the caller to have a non-`None`
    `entry_trigger_price`), `price` is today's live quote."""
    if state.entry_trigger_type == "breakout":
        already_triggered = price >= state.entry_trigger_price
    else:
        # "pullback_bounce" - see module docstring; anything else (shouldn't
        # happen given the caller's own filter) has nothing to re-check.
        already_triggered = state.entry_already_triggered
    return TickerIntradayState(
        ticker=state.ticker,
        region=state.region,
        updated_at=updated_at,
        price=price,
        entry_already_triggered=already_triggered,
    )


def previous_already_triggered(
    daily_state: TickerDailyState, existing_intraday: TickerIntradayState | None, today: date
) -> bool | None:
    """What "already triggered" meant *before* this refresh - the baseline a
    fresh intraday trigger is diffed against. An intraday row from earlier
    *today* (a previous refresh run this same day) is the freshest baseline;
    one from a prior day (or no row at all yet) is stale/absent, so last
    night's own daily-close read is the right baseline instead."""
    if existing_intraday is not None and existing_intraday.updated_at.date() == today:
        return existing_intraday.entry_already_triggered
    return daily_state.entry_already_triggered


def intraday_trigger_event(
    ticker: str, previous_triggered: bool | None, new_triggered: bool | None, price: float, now: datetime
) -> TriggerEvent | None:
    """`None` unless this is a genuine False -> True transition - the one
    direction worth an alert (a trigger firing). A True -> False "reversal"
    (price falling back below the breakout level) isn't logged here: Job A's
    own nightly recompute is the authoritative record of where things
    actually stood at each day's close."""
    if previous_triggered or not new_triggered:
        return None
    return TriggerEvent(
        id=None,
        entity_type="ticker",
        entity_key=ticker,
        event_type="entry_triggered",
        previous_value="false",
        new_value="true",
        occurred_at=now,
        details={"price": price, "source": "intraday"},
    )


@dataclass(frozen=True, slots=True)
class IntradayRefreshResult:
    job_status: str
    rows_processed: int
    new_triggers: int


def run_intraday_refresh(
    db: Session, market_data: MarketDataService, regions: tuple[str, ...] = REGIONS
) -> IntradayRefreshResult:
    job_repo = JobRunRepository(db)
    daily_repo = TickerDailyStateRepository(db)
    intraday_repo = TickerIntradayStateRepository(db)
    trigger_repo = TriggerEventRepository(db)

    job = job_repo.start("intraday_refresh")
    now = datetime.now(UTC)
    today = date.today()
    rows_processed = 0
    new_triggers = 0

    try:
        for region in regions:
            watchable = [s for s in daily_repo.latest_by_region(region) if s.entry_trigger_price is not None]
            if not watchable:
                continue
            quotes = market_data.get_latest_quotes([s.ticker for s in watchable])
            for state in watchable:
                quote = quotes.get(state.ticker)
                if quote is None:
                    continue
                existing_intraday = intraday_repo.get(state.ticker)
                previous_triggered = previous_already_triggered(state, existing_intraday, today)
                new_state = compute_intraday_state(state, quote.price, now)
                intraday_repo.upsert(new_state)
                rows_processed += 1
                event = intraday_trigger_event(
                    state.ticker, previous_triggered, new_state.entry_already_triggered, quote.price, now
                )
                if event is not None:
                    trigger_repo.record(event)
                    new_triggers += 1

        finished = job_repo.finish(job.id, status="success", rows_processed=rows_processed, error_message=None)
        return IntradayRefreshResult(finished.status, rows_processed, new_triggers)
    except Exception as exc:
        logger.exception("intraday_refresh failed")
        job_repo.finish(job.id, status="failed", rows_processed=rows_processed, error_message=str(exc)[:2000])
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region", choices=REGIONS, action="append", dest="regions", help="Limit to this region (repeatable)"
    )
    args = parser.parse_args()
    regions = tuple(args.regions) if args.regions else REGIONS

    market_data = MarketDataService(YFinanceProvider())
    db = SessionLocal()
    try:
        result = run_intraday_refresh(db, market_data, regions=regions)
        print(
            f"[intraday_refresh] {result.job_status} - {result.rows_processed} tickers revisados, "
            f"{result.new_triggers} entradas nuevas disparadas"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
