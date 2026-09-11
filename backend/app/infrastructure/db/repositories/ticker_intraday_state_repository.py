from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.interfaces.ticker_intraday_state_repository import TickerIntradayStateRepositoryPort
from app.domain.models.ticker_intraday_state import TickerIntradayState
from app.infrastructure.db.models import TickerIntradayStateORM


def _to_domain(orm: TickerIntradayStateORM) -> TickerIntradayState:
    return TickerIntradayState(
        ticker=orm.ticker,
        region=orm.region,
        updated_at=orm.updated_at,
        price=float(orm.price),
        entry_already_triggered=orm.entry_already_triggered,
    )


class TickerIntradayStateRepository(TickerIntradayStateRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, state: TickerIntradayState) -> TickerIntradayState:
        orm = self.db.get(TickerIntradayStateORM, state.ticker)
        if orm is None:
            orm = TickerIntradayStateORM(ticker=state.ticker)
            self.db.add(orm)
        orm.region = state.region
        orm.updated_at = state.updated_at
        orm.price = state.price
        orm.entry_already_triggered = state.entry_already_triggered
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def get(self, ticker: str) -> TickerIntradayState | None:
        orm = self.db.get(TickerIntradayStateORM, ticker)
        return _to_domain(orm) if orm is not None else None

    def list_by_region(self, region: str) -> list[TickerIntradayState]:
        stmt = select(TickerIntradayStateORM).where(TickerIntradayStateORM.region == region)
        return [_to_domain(o) for o in self.db.scalars(stmt).all()]
