"""Tercera auditoría, Bloque A-7: the real, DB-facing
`PositionSignalSnapshotRepository` had no test at all before this file - the
duplication bug (every fresh evaluation inserting another same-day row) lived
entirely at this layer, and `signal_performance_service.py`'s own dedupe
logic could only ever compensate for it after the fact, never prevent it.
Uses the same `engine`/`db_session` SQLite fixtures `tests/integration/conftest.py`
already provides for the API tests - no FastAPI app needed here, just a real
Session against a real (in-memory) schema.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.infrastructure.db.models import PortfolioORM
from app.infrastructure.db.repositories.position_signal_snapshot_repository import (
    PositionSignalSnapshotRepository,
)


def _make_portfolio(db_session: Session) -> int:
    portfolio = PortfolioORM(name="Test", base_currency="USD")
    db_session.add(portfolio)
    db_session.commit()
    db_session.refresh(portfolio)
    return portfolio.id


def test_save_replaces_a_same_day_snapshot_instead_of_accumulating(db_session: Session):
    # The exact production bug: a manual "Actualizar ahora" (bypasses
    # PortfolioRiskService's cache on purpose) or simply enough cache-miss
    # reloads in one day used to insert another row every time, so `n` in
    # signal_performance_service.py measured reload frequency, not distinct
    # evaluations.
    portfolio_id = _make_portfolio(db_session)
    repo = PositionSignalSnapshotRepository(db_session)

    repo.save(
        portfolio_id=portfolio_id, ticker="AAPL", signal="hold", exit_urgency=None, score=3, price=100.0,
        r_multiple=0.5, engine_version="v1",
    )
    repo.save(
        portfolio_id=portfolio_id, ticker="AAPL", signal="watch", exit_urgency="tighten_stop", score=1,
        price=98.0, r_multiple=0.2, engine_version="v1",
    )

    all_snapshots = repo.list_all()
    assert len(all_snapshots) == 1  # not 2 - the second call replaced the first, same (portfolio, ticker, day)
    assert all_snapshots[0].signal == "watch"  # the latest one, not the first
    assert all_snapshots[0].price == 98.0


def test_save_keeps_snapshots_from_different_portfolios_distinct(db_session: Session):
    portfolio_a = _make_portfolio(db_session)
    portfolio_b = PortfolioORM(name="Test 2", base_currency="USD")
    db_session.add(portfolio_b)
    db_session.commit()
    db_session.refresh(portfolio_b)

    repo = PositionSignalSnapshotRepository(db_session)
    repo.save(
        portfolio_id=portfolio_a, ticker="AAPL", signal="hold", exit_urgency=None, score=3, price=100.0,
        r_multiple=0.5, engine_version="v1",
    )
    repo.save(
        portfolio_id=portfolio_b.id, ticker="AAPL", signal="hold", exit_urgency=None, score=3, price=100.0,
        r_multiple=0.5, engine_version="v1",
    )

    assert len(repo.list_all()) == 2  # two distinct portfolios holding the same ticker the same day


def test_save_keeps_a_snapshot_from_a_genuinely_different_day(db_session: Session):
    portfolio_id = _make_portfolio(db_session)
    repo = PositionSignalSnapshotRepository(db_session)

    repo.save(
        portfolio_id=portfolio_id, ticker="AAPL", signal="hold", exit_urgency=None, score=3, price=100.0,
        r_multiple=0.5, engine_version="v1",
    )
    # Directly age the row back a day, then save a fresh one "today" - both
    # must survive, since they're genuinely distinct daily observations.
    from app.infrastructure.db.models import PositionSignalSnapshotORM

    row = db_session.query(PositionSignalSnapshotORM).one()
    row.created_at = datetime.now(UTC) - timedelta(days=1)
    db_session.commit()

    repo.save(
        portfolio_id=portfolio_id, ticker="AAPL", signal="watch", exit_urgency=None, score=1, price=98.0,
        r_multiple=0.2, engine_version="v1",
    )

    assert len(repo.list_all()) == 2
