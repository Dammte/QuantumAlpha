from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domain.interfaces.setup_performance_repository import SetupPerformanceRepositoryPort
from app.domain.models.setup_performance import SetupPerformance
from app.infrastructure.db.models import SetupPerformanceORM


def _to_domain(orm: SetupPerformanceORM) -> SetupPerformance:
    return SetupPerformance(
        id=orm.id,
        setup_name=orm.setup_name,
        family=orm.family,
        grade=orm.grade,
        market_regime=orm.market_regime,
        n_observations=orm.n_observations,
        trigger_rate=float(orm.trigger_rate) if orm.trigger_rate is not None else None,
        win_rate=float(orm.win_rate) if orm.win_rate is not None else None,
        expectancy_r=float(orm.expectancy_r) if orm.expectancy_r is not None else None,
        median_bars_held=float(orm.median_bars_held) if orm.median_bars_held is not None else None,
        mae_p80_pct=float(orm.mae_p80_pct) if orm.mae_p80_pct is not None else None,
        failure_rate_3d=float(orm.failure_rate_3d) if orm.failure_rate_3d is not None else None,
        confidence=orm.confidence,
        computed_at=orm.computed_at,
    )


class SetupPerformanceRepository(SetupPerformanceRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def replace_all(self, rows: list[SetupPerformance]) -> None:
        self.db.execute(delete(SetupPerformanceORM))
        for row in rows:
            self.db.add(
                SetupPerformanceORM(
                    setup_name=row.setup_name,
                    family=row.family,
                    grade=row.grade,
                    market_regime=row.market_regime,
                    n_observations=row.n_observations,
                    trigger_rate=row.trigger_rate,
                    win_rate=row.win_rate,
                    expectancy_r=row.expectancy_r,
                    median_bars_held=row.median_bars_held,
                    mae_p80_pct=row.mae_p80_pct,
                    failure_rate_3d=row.failure_rate_3d,
                    confidence=row.confidence,
                    computed_at=row.computed_at,
                )
            )
        self.db.commit()

    def all(self) -> list[SetupPerformance]:
        stmt = select(SetupPerformanceORM)
        return [_to_domain(orm) for orm in self.db.scalars(stmt).all()]
