from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models.recommendation_snapshot import RecommendationSnapshot
from app.infrastructure.db.models import RecommendationSnapshotORM


class RecommendationSnapshotRepository:
    """Append-only audit trail for "Analizar activo" verdicts - see
    `RecommendationSnapshotORM`'s docstring for why this exists. Writes are
    best-effort from the caller's perspective (the API endpoint that calls
    `save` catches and logs any failure rather than letting a logging problem
    break the actual analysis response - see `endpoints/ticker_analysis.py`)."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def save(
        self,
        ticker: str,
        verdict: str,
        score: int,
        price: float,
        currency: str,
        horizon: str,
        engine_version: str,
        factors: list[dict],
    ) -> RecommendationSnapshotORM:
        snapshot = RecommendationSnapshotORM(
            ticker=ticker,
            verdict=verdict,
            score=score,
            price=price,
            currency=currency,
            horizon=horizon,
            engine_version=engine_version,
            factors=factors,
        )
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

    def list_for_ticker(self, ticker: str, limit: int = 50) -> list[RecommendationSnapshotORM]:
        stmt = (
            select(RecommendationSnapshotORM)
            .where(RecommendationSnapshotORM.ticker == ticker.upper())
            .order_by(RecommendationSnapshotORM.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def list_all(self, since: datetime | None = None) -> list[RecommendationSnapshot]:
        """Every snapshot across every ticker, as domain objects - the raw
        material `signal_performance_service.py` aggregates into per-verdict
        hit rates and mean returns. Deliberately a separate method from
        `list_for_ticker` (which stays ORM-returning, unchanged, for the
        existing per-ticker history endpoint) rather than widening that
        method's contract for a second, unrelated use case."""
        stmt = select(RecommendationSnapshotORM).order_by(RecommendationSnapshotORM.created_at)
        if since is not None:
            stmt = stmt.where(RecommendationSnapshotORM.created_at >= since)
        return [
            RecommendationSnapshot(
                ticker=orm.ticker,
                created_at=orm.created_at,
                verdict=orm.verdict,
                score=orm.score,
                price=float(orm.price),
            )
            for orm in self.db.scalars(stmt).all()
        ]
