from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.interfaces.trade_plan_repository import TradePlanRepositoryPort
from app.domain.models.trade_plan import TradePlan
from app.infrastructure.db.models import TradePlanORM


def _to_domain(orm: TradePlanORM) -> TradePlan:
    return TradePlan(
        id=orm.id,
        portfolio_id=orm.portfolio_id,
        ticker=orm.ticker,
        entry_price=float(orm.entry_price),
        entry_date=orm.entry_date,
        initial_stop=float(orm.initial_stop) if orm.initial_stop is not None else None,
        initial_target=float(orm.initial_target) if orm.initial_target is not None else None,
        current_stop=float(orm.current_stop) if orm.current_stop is not None else None,
        initial_stop_basis=orm.initial_stop_basis,
        initial_stop_level_kind=orm.initial_stop_level_kind,
        current_stop_basis=orm.current_stop_basis,
        highest_close_since_entry=float(orm.highest_close_since_entry),
        initial_quantity=float(orm.initial_quantity),
        thesis=orm.thesis,
        engine_version=orm.engine_version,
        updated_at=orm.updated_at,
        closed_at=orm.closed_at,
    )


class TradePlanRepository(TradePlanRepositoryPort):
    """See `TradePlanORM`'s docstring for why there's no uniqueness
    constraint on (portfolio_id, ticker): a ticker can be bought, fully sold
    and bought again, and each such lot is its own row. `get_open` is always
    how a caller finds "the plan for what's held right now"."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_open(self, portfolio_id: int, ticker: str) -> TradePlan | None:
        stmt = (
            select(TradePlanORM)
            .where(
                TradePlanORM.portfolio_id == portfolio_id,
                TradePlanORM.ticker == ticker,
                TradePlanORM.closed_at.is_(None),
            )
            .order_by(TradePlanORM.entry_date.desc())
        )
        orm = self.db.scalars(stmt).first()
        return _to_domain(orm) if orm is not None else None

    def create(
        self,
        portfolio_id: int,
        ticker: str,
        entry_price: float,
        entry_date: date,
        initial_stop: float | None,
        initial_target: float | None,
        initial_quantity: float,
        thesis: str,
        engine_version: str,
        initial_stop_basis: str | None = None,
        initial_stop_level_kind: str | None = None,
    ) -> TradePlan:
        # Auditoria del Radar, bloque H2 (validación en la capa de
        # persistencia): un stop inicial en o por encima del precio de
        # entrada nunca es una geometría real, sea cual sea el origen de la
        # llamada - `compute_entry_geometry` ya lo impide en el cálculo
        # (`raw_risk_per_share <= 0` rechaza la operación), esto es la misma
        # garantía en el último punto antes de tocar la base de datos,
        # defensa en profundidad. Nunca se rellena con un valor que parezca
        # real (Regla 0 del propietario) - se guarda honestamente sin stop,
        # y el objetivo/anclaje que dependían de él tampoco se persisten.
        if initial_stop is not None and initial_stop >= entry_price:
            initial_stop = None
            initial_target = None
            initial_stop_basis = None
            initial_stop_level_kind = None
        orm = TradePlanORM(
            portfolio_id=portfolio_id,
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            initial_stop=initial_stop,
            initial_target=initial_target,
            current_stop=initial_stop,  # trailing starts equal to the initial stop
            initial_stop_basis=initial_stop_basis,
            initial_stop_level_kind=initial_stop_level_kind,
            current_stop_basis=initial_stop_basis,  # el trailing empieza en el mismo anclaje que el inicial
            highest_close_since_entry=entry_price,
            initial_quantity=initial_quantity,
            thesis=thesis,
            engine_version=engine_version,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def update_trailing(
        self,
        plan_id: int,
        current_stop: float,
        highest_close_since_entry: float,
        current_stop_basis: str | None = None,
    ) -> None:
        orm = self.db.get(TradePlanORM, plan_id)
        if orm is None:
            return
        orm.current_stop = current_stop
        orm.highest_close_since_entry = highest_close_since_entry
        # Auditoria del Radar, bloque H2: `None` significa "sin cambio de
        # anclaje" (el llamador no tenía uno nuevo que ofrecer), nunca
        # "borra el anclaje existente" - solo se sobrescribe cuando se pasa
        # un texto real.
        if current_stop_basis is not None:
            orm.current_stop_basis = current_stop_basis
        orm.updated_at = datetime.now(UTC)
        self.db.commit()

    def close(self, portfolio_id: int, ticker: str) -> None:
        """Marks the currently-open plan (if any) as closed - called when a
        SELL brings the held quantity for this ticker back to 0, so a future
        re-entry starts a fresh plan/lot rather than reusing a stale one."""
        plan = self.get_open(portfolio_id, ticker)
        if plan is None or plan.id is None:
            return
        orm = self.db.get(TradePlanORM, plan.id)
        if orm is not None:
            orm.closed_at = datetime.now(UTC)
            self.db.commit()
