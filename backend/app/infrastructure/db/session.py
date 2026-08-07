from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# prepare_threshold=None disables psycopg's server-side prepared statements.
# Required when DATABASE_URL points at a transaction-mode pooler (e.g. Supabase's
# Supavisor on port 6543, or any PgBouncer in transaction mode): prepared statements
# are tied to one physical server connection, but a transaction pooler hands out a
# different one per transaction, so a statement prepared on request N can vanish by
# request N+5 ("prepared statement ... does not exist"). Harmless against a direct
# connection too, so left unconditional rather than branching on the URL.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args={"prepare_threshold": None},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
