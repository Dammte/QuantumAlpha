from sqlalchemy import select
from sqlalchemy.orm import Session

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
