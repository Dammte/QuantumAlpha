from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domain.interfaces.ticker_daily_state_repository import TickerDailyStateRepositoryPort
from app.domain.models.ticker_daily_state import TickerDailyState
from app.infrastructure.db.models import TickerDailyStateORM


def _to_domain(orm: TickerDailyStateORM) -> TickerDailyState:
    return TickerDailyState(
        id=orm.id,
        region=orm.region,
        ticker=orm.ticker,
        trade_date=orm.trade_date,
        computed_at=orm.computed_at,
        price=float(orm.price),
        currency=orm.currency,
        trend=orm.trend,
        stage=orm.stage,
        rs_rating=orm.rs_rating,
        adx14=float(orm.adx14) if orm.adx14 is not None else None,
        atr_multiple=float(orm.atr_multiple) if orm.atr_multiple is not None else None,
        rsi14=float(orm.rsi14) if orm.rsi14 is not None else None,
        gate_passes=orm.gate_passes,
        gate_conditions=orm.gate_conditions,
        gate_version=orm.gate_version,
        entry_trigger_type=orm.entry_trigger_type,
        entry_trigger_price=float(orm.entry_trigger_price) if orm.entry_trigger_price is not None else None,
        entry_already_triggered=orm.entry_already_triggered,
        stop_loss=float(orm.stop_loss) if orm.stop_loss is not None else None,
        take_profit=float(orm.take_profit) if orm.take_profit is not None else None,
        take_profit_method=orm.take_profit_method,
        risk_reward=float(orm.risk_reward) if orm.risk_reward is not None else None,
        entry_geometry=orm.entry_geometry,
        grade=orm.grade,
        setups=orm.setups,
        timeframe_strip=orm.timeframe_strip,
        sector=orm.sector,
        sector_rs_percentile=orm.sector_rs_percentile,
    )


class TickerDailyStateRepository(TickerDailyStateRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, state: TickerDailyState) -> TickerDailyState:
        self.db.execute(
            delete(TickerDailyStateORM).where(
                TickerDailyStateORM.region == state.region,
                TickerDailyStateORM.ticker == state.ticker,
                TickerDailyStateORM.trade_date == state.trade_date,
            )
        )
        orm = TickerDailyStateORM(
            region=state.region,
            ticker=state.ticker,
            trade_date=state.trade_date,
            computed_at=state.computed_at,
            price=state.price,
            currency=state.currency,
            trend=state.trend,
            stage=state.stage,
            rs_rating=state.rs_rating,
            adx14=state.adx14,
            atr_multiple=state.atr_multiple,
            rsi14=state.rsi14,
            gate_passes=state.gate_passes,
            gate_conditions=state.gate_conditions,
            gate_version=state.gate_version,
            entry_trigger_type=state.entry_trigger_type,
            entry_trigger_price=state.entry_trigger_price,
            entry_already_triggered=state.entry_already_triggered,
            stop_loss=state.stop_loss,
            take_profit=state.take_profit,
            take_profit_method=state.take_profit_method,
            risk_reward=state.risk_reward,
            entry_geometry=state.entry_geometry,
            grade=state.grade,
            setups=state.setups,
            timeframe_strip=state.timeframe_strip,
            sector=state.sector,
            sector_rs_percentile=state.sector_rs_percentile,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def get(self, region: str, ticker: str, trade_date: date) -> TickerDailyState | None:
        stmt = select(TickerDailyStateORM).where(
            TickerDailyStateORM.region == region,
            TickerDailyStateORM.ticker == ticker,
            TickerDailyStateORM.trade_date == trade_date,
        )
        orm = self.db.scalars(stmt).first()
        return _to_domain(orm) if orm is not None else None

    def latest_for_ticker(self, ticker: str) -> TickerDailyState | None:
        stmt = (
            select(TickerDailyStateORM)
            .where(TickerDailyStateORM.ticker == ticker)
            .order_by(TickerDailyStateORM.trade_date.desc())
            .limit(1)
        )
        orm = self.db.scalars(stmt).first()
        return _to_domain(orm) if orm is not None else None

    def latest_by_region(self, region: str) -> list[TickerDailyState]:
        # One row per ticker: the one with the max trade_date for that ticker
        # within this region. Grouped in Python rather than a DISTINCT ON/
        # window-function query - simpler and dialect-portable (Postgres in
        # production, SQLite in tests/integration), and cheap enough at this
        # universe's size (a few hundred to ~1000 rows per region at most).
        stmt = select(TickerDailyStateORM).where(TickerDailyStateORM.region == region)
        best_by_ticker: dict[str, TickerDailyStateORM] = {}
        for orm in self.db.scalars(stmt).all():
            current = best_by_ticker.get(orm.ticker)
            if current is None or orm.trade_date > current.trade_date:
                best_by_ticker[orm.ticker] = orm
        return [_to_domain(o) for o in best_by_ticker.values()]

    def for_region_and_date(self, region: str, trade_date: date) -> list[TickerDailyState]:
        stmt = select(TickerDailyStateORM).where(
            TickerDailyStateORM.region == region,
            TickerDailyStateORM.trade_date == trade_date,
        )
        return [_to_domain(o) for o in self.db.scalars(stmt).all()]
