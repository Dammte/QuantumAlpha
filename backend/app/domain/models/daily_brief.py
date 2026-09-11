from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class DailyBrief:
    """One row per (portfolio_id, brief_date) - the "Hoy" dashboard's whole
    reason for existing: a short, precomputed summary of what changed since
    the last time the propietario looked, so opening the app answers "what do
    I need to look at today" without hunting through the Radar/Activo views
    by hand. Built once by `daily_close.py` from that same run's own
    `TriggerEvent` rows, never recomputed on every page load.

    Scoped per portfolio because `positions_needing_action` genuinely is
    (only that portfolio's own open positions) - `new_entry_triggers`/
    `new_gate_passes` are universe-wide counts (not portfolio-specific) that
    end up duplicated across every portfolio's row on a given day rather than
    living in their own global table; simpler than splitting one dashboard's
    data across two different scopes for a personal, single-user app with at
    most a handful of portfolios."""

    id: int | None
    portfolio_id: int
    brief_date: date
    computed_at: datetime
    positions_needing_action: int  # position rows with urgency != HOLD
    new_entry_triggers: int  # ticker rows newly already_triggered today
    new_gate_passes: int  # ticker rows whose gate flipped false -> true today
    headline: str  # short, human-readable one-liner - see daily_close.py
