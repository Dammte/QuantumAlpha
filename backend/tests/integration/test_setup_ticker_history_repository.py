"""Parte 11.2 (biblioteca de setups del Radar, quant_methodology.md §28):
la tabla `setup_ticker_history` es una FOTO COMPLETA reemplazada en cada
corrida de `scripts/setup_replay_study.py`, no un histórico acumulado - ver
`SetupTickerHistory`'s propio docstring. Mismo patrón que
`test_setup_performance_repository.py`: una Session real contra el esquema
SQLite de pruebas, sin la app FastAPI de por medio."""

from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.domain.models.setup_ticker_history import SetupTickerHistory
from app.infrastructure.db.repositories.setup_ticker_history_repository import SetupTickerHistoryRepository


def _row(ticker: str = "NVDA", setup_name: str = "vcp_3_contracciones", **overrides) -> SetupTickerHistory:
    defaults = dict(
        id=None, ticker=ticker, region="us", setup_name=setup_name, family="vcp",
        n_observations=4, n_triggered=3, n_target_hit=2,
        first_ready_date=date(2020, 1, 1), last_ready_date=date(2024, 6, 1),
        computed_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return SetupTickerHistory(**defaults)


def test_replace_all_persists_every_field(db_session: Session):
    repo = SetupTickerHistoryRepository(db_session)

    repo.replace_all([_row()])

    rows = repo.all()
    assert len(rows) == 1
    row = rows[0]
    assert row.id is not None
    assert row.ticker == "NVDA"
    assert row.region == "us"
    assert row.setup_name == "vcp_3_contracciones"
    assert row.family == "vcp"
    assert row.n_observations == 4
    assert row.n_triggered == 3
    assert row.n_target_hit == 2
    assert row.first_ready_date == date(2020, 1, 1)
    assert row.last_ready_date == date(2024, 6, 1)


def test_replace_all_replaces_the_whole_table_not_accumulates(db_session: Session):
    repo = SetupTickerHistoryRepository(db_session)

    repo.replace_all([_row(ticker="NVDA"), _row(ticker="AAPL")])
    assert len(repo.all()) == 2

    repo.replace_all([_row(ticker="MSFT")])

    rows = repo.all()
    assert len(rows) == 1
    assert rows[0].ticker == "MSFT"


def test_replace_all_keeps_distinct_setups_for_the_same_ticker(db_session: Session):
    repo = SetupTickerHistoryRepository(db_session)

    repo.replace_all(
        [
            _row(ticker="NVDA", setup_name="vcp_3_contracciones", n_observations=4),
            _row(ticker="NVDA", setup_name="ruptura_darvas", family="breakout", n_observations=2),
        ]
    )

    rows = repo.all()
    assert len(rows) == 2
    by_name = {row.setup_name: row.n_observations for row in rows}
    assert by_name == {"vcp_3_contracciones": 4, "ruptura_darvas": 2}


def test_all_returns_empty_list_before_any_study_has_run(db_session: Session):
    repo = SetupTickerHistoryRepository(db_session)
    assert repo.all() == []


def test_replace_all_round_trips_a_setup_that_never_triggered(db_session: Session):
    repo = SetupTickerHistoryRepository(db_session)
    repo.replace_all([_row(n_observations=2, n_triggered=0, n_target_hit=0)])

    row = repo.all()[0]
    assert row.n_observations == 2
    assert row.n_triggered == 0
    assert row.n_target_hit == 0
