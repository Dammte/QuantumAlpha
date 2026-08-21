from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class UniverseMember:
    """One ticker's membership in one region's investable universe, as of one
    monthly snapshot - the D14 fix (Segunda auditoría, Bloque 3): the
    curated, hardcoded universe in `market_universe.py` has no survivorship-
    bias protection at all (a ticker that got delisted or dropped from an
    index simply isn't in today's dict, so any backtest run against it never
    sees the failures). Persisting a dated snapshot, one per region per
    refresh, means a later point-in-time query can ask "who was actually in
    this universe as of month X" instead of implicitly asking "who's in it
    today" for every historical date at once.
    """

    ticker: str
    region: str  # "us" | "europe"
    sector: str | None
    as_of_date: date  # the calendar month this snapshot represents
    source: str  # "live" (Wikipedia + liquidity filter) | "curated_fallback"
