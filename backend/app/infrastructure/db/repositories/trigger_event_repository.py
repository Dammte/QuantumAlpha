from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.interfaces.trigger_event_repository import TriggerEventRepositoryPort
from app.domain.models.trigger_event import TriggerEvent
from app.infrastructure.db.models import TriggerEventORM


def _to_domain(orm: TriggerEventORM) -> TriggerEvent:
    return TriggerEvent(
        id=orm.id,
        entity_type=orm.entity_type,
        entity_key=orm.entity_key,
        event_type=orm.event_type,
        previous_value=orm.previous_value,
        new_value=orm.new_value,
        occurred_at=orm.occurred_at,
        details=orm.details,
    )


class TriggerEventRepository(TriggerEventRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def record(self, event: TriggerEvent) -> TriggerEvent:
        orm = TriggerEventORM(
            entity_type=event.entity_type,
            entity_key=event.entity_key,
            event_type=event.event_type,
            previous_value=event.previous_value,
            new_value=event.new_value,
            occurred_at=event.occurred_at,
            details=event.details,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def list_since(self, since: datetime, entity_type: str | None = None) -> list[TriggerEvent]:
        stmt = select(TriggerEventORM).where(TriggerEventORM.occurred_at >= since)
        if entity_type is not None:
            stmt = stmt.where(TriggerEventORM.entity_type == entity_type)
        stmt = stmt.order_by(TriggerEventORM.occurred_at)
        return [_to_domain(o) for o in self.db.scalars(stmt).all()]
