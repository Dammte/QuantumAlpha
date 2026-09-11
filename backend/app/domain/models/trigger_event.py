from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TriggerEvent:
    """One row per meaningful state change `daily_close.py` (or
    `intraday_refresh.py`) detects between yesterday's precomputed state and
    today's - the append-only audit trail Fase 8 measures hit rates against
    ("is the system working", Parte 0's fourth question). Deliberately not
    just "today's snapshot" (that's `TickerDailyState`/`PositionDailyState`):
    a *change* (the gate flipped from failing to passing, an entry trigger
    fired, an exit urgency escalated) is the actionable, countable unit a
    measurement study needs - re-deriving "what changed" from two full daily
    snapshots every time a study runs would be slower and more error-prone
    than recording it once, the moment it's first detected."""

    id: int | None
    entity_type: str  # "ticker" | "position"
    entity_key: str  # a ticker, or "{portfolio_id}:{ticker}" for a position
    event_type: str  # "gate_passed" | "gate_failed" | "entry_triggered" | "exit_urgency_escalated" | ...
    previous_value: str | None
    new_value: str | None
    occurred_at: datetime
    details: dict  # JSON-safe extra context (price at the time, trigger price, etc.)
