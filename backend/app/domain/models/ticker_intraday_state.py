from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TickerIntradayState:
    """Cheap, frequently-overwritten companion to `TickerDailyState` - re-checks
    a live intraday quote against *yesterday's* already-computed entry
    trigger (`trade_geometry.EntryTrigger.already_triggered`'s whole reason
    for existing - see that module's docstring) without recomputing the full
    daily state mid-session. One row per (region, ticker), always the latest
    read - overwritten on every `intraday_refresh.py` run, never a history
    (that's what `TriggerEvent`/`trigger_history` is for)."""

    ticker: str
    region: str
    updated_at: datetime
    price: float
    entry_already_triggered: bool | None  # None when there was no entry trigger to re-check against
