from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domain.interfaces.position_daily_state_repository import PositionDailyStateRepositoryPort
from app.domain.models.position_daily_state import PositionDailyState
from app.infrastructure.db.models import PositionDailyStateORM


def _to_domain(orm: PositionDailyStateORM) -> PositionDailyState:
    return PositionDailyState(
        id=orm.id,
        portfolio_id=orm.portfolio_id,
        ticker=orm.ticker,
        trade_date=orm.trade_date,
        computed_at=orm.computed_at,
        urgency=orm.urgency,
        reasons=orm.reasons,
        price=float(orm.price),
        r_multiple=float(orm.r_multiple) if orm.r_multiple is not None else None,
        current_stop=float(orm.current_stop) if orm.current_stop is not None else None,
        engine_version=orm.engine_version,
    )


class PositionDailyStateRepository(PositionDailyStateRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, state: PositionDailyState) -> PositionDailyState:
        self.db.execute(
            delete(PositionDailyStateORM).where(
                PositionDailyStateORM.portfolio_id == state.portfolio_id,
                PositionDailyStateORM.ticker == state.ticker,
                PositionDailyStateORM.trade_date == state.trade_date,
            )
        )
        orm = PositionDailyStateORM(
            portfolio_id=state.portfolio_id,
            ticker=state.ticker,
            trade_date=state.trade_date,
            computed_at=state.computed_at,
            urgency=state.urgency,
            reasons=state.reasons,
            price=state.price,
            r_multiple=state.r_multiple,
            current_stop=state.current_stop,
            engine_version=state.engine_version,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def latest_for_portfolio(self, portfolio_id: int) -> list[PositionDailyState]:
        stmt = select(PositionDailyStateORM).where(PositionDailyStateORM.portfolio_id == portfolio_id)
        best_by_ticker: dict[str, PositionDailyStateORM] = {}
        for orm in self.db.scalars(stmt).all():
            current = best_by_ticker.get(orm.ticker)
            if current is None or orm.trade_date > current.trade_date:
                best_by_ticker[orm.ticker] = orm
        return [_to_domain(o) for o in best_by_ticker.values()]

    def for_portfolio_and_date(self, portfolio_id: int, trade_date: date) -> list[PositionDailyState]:
        stmt = select(PositionDailyStateORM).where(
            PositionDailyStateORM.portfolio_id == portfolio_id,
            PositionDailyStateORM.trade_date == trade_date,
        )
        return [_to_domain(o) for o in self.db.scalars(stmt).all()]
