"""Durable, DB-backed read-through/write-through cache for expensive market
computations - a companion to (not a replacement for) each service's own
in-process cache (see `market_screener_service.py`, `portfolio_risk_service.py`,
`premium_watchlist_service.py`).

Why both layers exist: the in-process cache is free to read and write, and is
all that's needed while the process stays up. But this is a `plan: starter`
Render web service - it doesn't spin down between requests, but it *does*
restart on every deploy, and a personal project under active development
deploys often. Each restart empties every in-process cache, which would
otherwise force the very next request after every single deploy to eat the
full multi-second-to-tens-of-seconds compute cost synchronously - on Render's
shared, throttled CPU that means blocking the *entire* single-process server
for however long that takes, for whichever user happens to make that request
(see `portfolio_risk_service.py`'s docstring on why that used to be worse, not
better, when parallelized). Backing the cache with Postgres means a restart
only loses freshness, never the data itself: the next request after a restart
can serve the last known-good result immediately (however many hours old)
instead of blocking, matching the "always show saved data, refresh
occasionally or on request" behavior this cache layer promises everywhere
else.

Deliberately takes an explicit `Session` (the same one FastAPI's `DbSession`
dependency hands every endpoint) rather than opening its own - so this always
goes through the same connection pool and, critically, the same dependency
override the test suite already uses to point every DB access at an isolated
in-memory SQLite database instead of the real one.

Every failure here is swallowed rather than raised: a durable-cache read or
write is a pure optimization on top of a system that already works without
it (each service's own in-process cache, and a live recompute as the final
fallback) - a transient DB hiccup should never be the reason a request that
could still succeed on its own fails instead.

**Every caught failure also rolls back the session before returning.** This
isn't optional cleanup: on Postgres, one failed statement leaves the whole
transaction "aborted" until an explicit ROLLBACK - every later statement on
that *same* session (e.g. the endpoint's own repository queries, which share
this exact session via FastAPI's per-request DbSession dependency) raises
`InFailedSqlTransaction` too, even though nothing is actually wrong with
them. A caught-and-logged exception here that skips the rollback doesn't fail
softly at all - it silently turns into a 500 on whatever the endpoint does
next. This is exactly what happened in production the one time this table
didn't exist yet (before the migration had been run): every other
`computation_cache` query being swallowed correctly masked *that* failure,
but the *next* query in the same request (fetching the portfolio's own
transactions) blew up unhandled - a caught exception two calls upstream
disguised as an unrelated one downstream. Covered by
`test_portfolios_api.py::test_portfolio_risk_survives_a_missing_computation_cache_table`.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.infrastructure.db.repositories.computation_cache_repository import ComputationCacheRepository

logger = logging.getLogger(__name__)


def _safe_get(db: Session, cache_key: str) -> tuple[Any, datetime] | None:
    try:
        return ComputationCacheRepository(db).get(cache_key)
    except Exception:
        logger.exception("Durable cache read failed for %s", cache_key)
        db.rollback()  # leaves `db` usable for whatever the caller does next - see module docstring
        return None


def load_fresh(db: Session, cache_key: str, max_age: timedelta) -> Any | None:
    """The cached payload if present and younger than `max_age`, else None -
    None either means "nothing cached yet" or "cached, but stale enough that
    the caller should recompute", which is exactly the one bit of information
    a caller needs (it already knows how to compute a fresh value)."""
    cached = _safe_get(db, cache_key)
    if cached is None:
        return None
    payload, computed_at = cached
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=UTC)
    if datetime.now(UTC) - computed_at >= max_age:
        return None
    return payload


def load_any(db: Session, cache_key: str) -> tuple[Any, datetime] | None:
    """The cached payload regardless of age, plus when it was computed - for a
    caller that wants to show stale-but-real data (with an "actualizado hace
    X" disclosure) rather than force a live recompute no matter how old the
    last one is."""
    cached = _safe_get(db, cache_key)
    if cached is None:
        return None
    payload, computed_at = cached
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=UTC)
    return payload, computed_at


def save(db: Session, cache_key: str, payload: Any, computed_at: datetime | None = None) -> datetime:
    """Best-effort write-through. Returns the `computed_at` stamped on this
    write regardless of whether the write itself actually succeeded, so a
    caller can use it for the response's "actualizado hace X" even when
    persistence quietly failed - the value just won't survive a restart. Pass
    `computed_at` explicitly when the response being cached needs to carry the
    exact same timestamp (e.g. inside its own JSON body) rather than a
    microsecond-later one generated here."""
    computed_at = computed_at or datetime.now(UTC)
    try:
        ComputationCacheRepository(db).set(cache_key, payload, computed_at)
    except Exception:
        logger.exception("Durable cache write failed for %s", cache_key)
        db.rollback()  # see module docstring - must leave `db` usable for the rest of the request
    return computed_at
