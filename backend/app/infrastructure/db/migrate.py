"""Runs pending Alembic migrations automatically at application startup.

Exists so a schema change never again requires a human to open Supabase's SQL
editor and run something by hand - the exact failure mode that left the
durable cache (and, before that, cash-movement tracking) effectively unable
to do its job in production for days after each migration shipped, because
running it was a manual step nobody had gotten to yet. Alembic's `upgrade
head` is idempotent - running it against a database that's already fully
migrated is a no-op - so this is safe to call unconditionally on every single
boot, not just the first one after a schema change actually ships.

Shares the app's own already-configured `engine` (see session.py, notably its
`prepare_threshold=None` for compatibility with Supabase's Supavisor
transaction-mode pooler) rather than letting Alembic build a second, more
naively-configured one from the bare connection URL - see env.py's
`run_migrations_online` for the other half of this.

Deliberately never allowed to crash the app: any failure here (a transient
connection issue, a permissions problem, anything) degrades back to exactly
today's status quo - the app still starts and serves traffic against
whatever schema is already live, with the same graceful degradation
`durable_cache.py` already has for a genuinely out-of-date schema. A failed
migration is a reason to look at the logs, never a reason the whole service
should refuse to start.

Skipped entirely under pytest (`PYTEST_CURRENT_TEST` is set automatically by
pytest for the duration of every test - a standard, zero-config way to detect
this without touching every test's fixtures). The integration test suite
builds `TestClient(app)` as a context manager, which fires this same startup
event on every single test - without this guard, each one would try to open
a real database connection using the raw `engine` (which, unlike `get_db`,
tests never override via `app.dependency_overrides`), turning a fast,
fully-isolated SQLite-backed test suite into hundreds of real, possibly
hanging network calls.
"""

import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from app.core.config import get_settings
from app.infrastructure.db.session import engine

logger = logging.getLogger(__name__)

ALEMBIC_DIR = Path(__file__).resolve().parents[3] / "alembic"


def run_migrations_on_startup() -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    try:
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", str(ALEMBIC_DIR))
        alembic_cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
        with engine.connect() as connection:
            alembic_cfg.attributes["connection"] = connection
            command.upgrade(alembic_cfg, "head")
        logger.info("Database schema is up to date.")
    except Exception:
        logger.exception(
            "Automatic migration on startup failed - continuing with whatever schema is currently live. "
            "May need to be applied by hand (see alembic/versions/)."
        )
