from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domain.interfaces.setup_ticker_history_repository import SetupTickerHistoryRepositoryPort
from app.domain.models.setup_ticker_history import SetupTickerHistory
from app.infrastructure.db.models import SetupTickerHistoryORM


def _to_domain(orm: SetupTickerHistoryORM) -> SetupTickerHistory:
    return SetupTickerHistory(
        id=orm.id,
        ticker=orm.ticker,
        region=orm.region,
        setup_name=orm.setup_name,
        family=orm.family,
        n_observations=orm.n_observations,
        n_triggered=orm.n_triggered,
        n_target_hit=orm.n_target_hit,
        first_ready_date=orm.first_ready_date,
        last_ready_date=orm.last_ready_date,
        computed_at=orm.computed_at,
    )


class SetupTickerHistoryRepository(SetupTickerHistoryRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def replace_all(self, rows: list[SetupTickerHistory]) -> None:
        self.db.execute(delete(SetupTickerHistoryORM))
        for row in rows:
            self.db.add(
                SetupTickerHistoryORM(
                    ticker=row.ticker,
                    region=row.region,
                    setup_name=row.setup_name,
                    family=row.family,
                    n_observations=row.n_observations,
                    n_triggered=row.n_triggered,
                    n_target_hit=row.n_target_hit,
                    first_ready_date=row.first_ready_date,
                    last_ready_date=row.last_ready_date,
                    computed_at=row.computed_at,
                )
            )
        self.db.commit()

    def all(self) -> list[SetupTickerHistory]:
        stmt = select(SetupTickerHistoryORM)
        return [_to_domain(orm) for orm in self.db.scalars(stmt).all()]
