from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domain.interfaces.daily_brief_repository import DailyBriefRepositoryPort
from app.domain.models.daily_brief import DailyBrief
from app.infrastructure.db.models import DailyBriefORM


def _to_domain(orm: DailyBriefORM) -> DailyBrief:
    return DailyBrief(
        id=orm.id,
        portfolio_id=orm.portfolio_id,
        brief_date=orm.brief_date,
        computed_at=orm.computed_at,
        positions_needing_action=orm.positions_needing_action,
        new_entry_triggers=orm.new_entry_triggers,
        new_gate_passes=orm.new_gate_passes,
        headline=orm.headline,
    )


class DailyBriefRepository(DailyBriefRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, brief: DailyBrief) -> DailyBrief:
        self.db.execute(
            delete(DailyBriefORM).where(
                DailyBriefORM.portfolio_id == brief.portfolio_id,
                DailyBriefORM.brief_date == brief.brief_date,
            )
        )
        orm = DailyBriefORM(
            portfolio_id=brief.portfolio_id,
            brief_date=brief.brief_date,
            computed_at=brief.computed_at,
            positions_needing_action=brief.positions_needing_action,
            new_entry_triggers=brief.new_entry_triggers,
            new_gate_passes=brief.new_gate_passes,
            headline=brief.headline,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def latest_for_portfolio(self, portfolio_id: int) -> DailyBrief | None:
        stmt = (
            select(DailyBriefORM)
            .where(DailyBriefORM.portfolio_id == portfolio_id)
            .order_by(DailyBriefORM.brief_date.desc())
            .limit(1)
        )
        orm = self.db.scalars(stmt).first()
        return _to_domain(orm) if orm is not None else None
