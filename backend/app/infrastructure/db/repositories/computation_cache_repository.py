from datetime import datetime

from sqlalchemy.orm import Session

from app.infrastructure.db.models import ComputationCacheORM


class ComputationCacheRepository:
    """Plain key -> (payload, computed_at) store backing `app/services/durable_cache.py`
    - see `ComputationCacheORM`'s docstring for why this table exists."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, cache_key: str) -> tuple[dict, datetime] | None:
        row = self.db.get(ComputationCacheORM, cache_key)
        return (row.payload, row.computed_at) if row is not None else None

    def set(self, cache_key: str, payload: dict, computed_at: datetime) -> None:
        # Plain get-then-add/update rather than a dialect-specific "ON CONFLICT" -
        # this table is written at most a few times an hour per key, so the extra
        # round trip is irrelevant, and staying dialect-agnostic means this works
        # unchanged against both the SQLite engine the test suite uses and the
        # real Postgres database.
        row = self.db.get(ComputationCacheORM, cache_key)
        if row is None:
            self.db.add(ComputationCacheORM(cache_key=cache_key, computed_at=computed_at, payload=payload))
        else:
            row.computed_at = computed_at
            row.payload = payload
        self.db.commit()
