from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.interfaces.position_signal_snapshot_repository import PositionSignalSnapshotRepositoryPort
from app.domain.models.position_signal_snapshot import PositionSignalSnapshot
from app.infrastructure.db.models import PositionSignalSnapshotORM


def _to_domain(orm: PositionSignalSnapshotORM) -> PositionSignalSnapshot:
    return PositionSignalSnapshot(
        portfolio_id=orm.portfolio_id,
        ticker=orm.ticker,
        created_at=orm.created_at,
        signal=orm.signal,
        exit_urgency=orm.exit_urgency,
        score=orm.score,
        price=float(orm.price),
        r_multiple=float(orm.r_multiple) if orm.r_multiple is not None else None,
        engine_version=orm.engine_version,
    )


class PositionSignalSnapshotRepository(PositionSignalSnapshotRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def save(
        self,
        portfolio_id: int,
        ticker: str,
        signal: str,
        exit_urgency: str | None,
        score: int,
        price: float,
        r_multiple: float | None,
        engine_version: str,
    ) -> PositionSignalSnapshot:
        orm = PositionSignalSnapshotORM(
            portfolio_id=portfolio_id,
            ticker=ticker,
            signal=signal,
            exit_urgency=exit_urgency,
            score=score,
            price=price,
            r_multiple=r_multiple,
            engine_version=engine_version,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def list_all(self, since: datetime | None = None) -> list[PositionSignalSnapshot]:
        stmt = select(PositionSignalSnapshotORM).order_by(PositionSignalSnapshotORM.created_at)
        if since is not None:
            stmt = stmt.where(PositionSignalSnapshotORM.created_at >= since)
        return [_to_domain(orm) for orm in self.db.scalars(stmt).all()]
