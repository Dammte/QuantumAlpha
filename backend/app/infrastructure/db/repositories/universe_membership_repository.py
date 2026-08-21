from datetime import date

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.domain.interfaces.universe_membership_repository import UniverseMembershipRepositoryPort
from app.domain.models.universe_membership import UniverseMember
from app.infrastructure.db.models import UniverseMembershipORM


def _to_domain(orm: UniverseMembershipORM) -> UniverseMember:
    return UniverseMember(
        ticker=orm.ticker, region=orm.region, sector=orm.sector, as_of_date=orm.as_of_date, source=orm.source
    )


class UniverseMembershipRepository(UniverseMembershipRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def save_snapshot(self, region: str, as_of_date: date, source: str, members: list[UniverseMember]) -> None:
        # Idempotent per (region, as_of_date) - see the port's own docstring.
        self.db.execute(
            delete(UniverseMembershipORM).where(
                UniverseMembershipORM.region == region, UniverseMembershipORM.as_of_date == as_of_date
            )
        )
        self.db.add_all(
            UniverseMembershipORM(
                region=region, ticker=m.ticker, sector=m.sector, as_of_date=as_of_date, source=source
            )
            for m in members
        )
        self.db.commit()

    def latest_as_of_date(self, region: str) -> date | None:
        stmt = select(func.max(UniverseMembershipORM.as_of_date)).where(UniverseMembershipORM.region == region)
        return self.db.scalar(stmt)

    def members_as_of(self, region: str, as_of_date: date | None = None) -> list[UniverseMember]:
        target_date = as_of_date if as_of_date is not None else self.latest_as_of_date(region)
        if target_date is None:
            return []
        # The latest snapshot at or before target_date - not necessarily an
        # exact match, so a caller asking for a date between two monthly
        # refreshes still gets the most recent one that actually applied then.
        snapshot_date = self.db.scalar(
            select(func.max(UniverseMembershipORM.as_of_date)).where(
                UniverseMembershipORM.region == region, UniverseMembershipORM.as_of_date <= target_date
            )
        )
        if snapshot_date is None:
            return []
        stmt = select(UniverseMembershipORM).where(
            UniverseMembershipORM.region == region, UniverseMembershipORM.as_of_date == snapshot_date
        )
        return [_to_domain(orm) for orm in self.db.scalars(stmt).all()]

    def all_as_of_dates(self, region: str) -> list[date]:
        stmt = (
            select(UniverseMembershipORM.as_of_date)
            .where(UniverseMembershipORM.region == region)
            .distinct()
            .order_by(UniverseMembershipORM.as_of_date)
        )
        return list(self.db.scalars(stmt).all())
