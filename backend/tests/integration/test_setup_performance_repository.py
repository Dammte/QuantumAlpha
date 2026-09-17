"""Parte 10.2 (biblioteca de setups del Radar, quant_methodology.md §28):
la tabla `setup_performance` es una FOTO COMPLETA reemplazada en cada
corrida de `scripts/setup_replay_study.py`, no un histórico acumulado - ver
`SetupPerformance`'s propio docstring para el porqué (Postgres trata cada
NULL de `grade`/`market_regime` como distinto en una restricción UNIQUE,
así que `replace_all` borra e inserta en vez de confiar en una). Mismo
patrón que `test_position_signal_snapshot_repository.py`: una Session real
contra el esquema SQLite de pruebas, sin la app FastAPI de por medio."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.domain.models.setup_performance import SetupPerformance
from app.infrastructure.db.repositories.setup_performance_repository import SetupPerformanceRepository


def _row(setup_name: str = "vcp_3_contracciones", grade: str | None = None, **overrides) -> SetupPerformance:
    defaults = dict(
        id=None, setup_name=setup_name, family="vcp", grade=grade, market_regime=None,
        n_observations=35, trigger_rate=0.6, win_rate=0.55, expectancy_r=0.42,
        median_bars_held=6.0, mae_p80_pct=-0.03, failure_rate_3d=0.1, confidence="measured",
        computed_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return SetupPerformance(**defaults)


def test_replace_all_persists_every_field_and_round_trips_none_grade(db_session: Session):
    repo = SetupPerformanceRepository(db_session)

    repo.replace_all([_row(grade=None)])

    rows = repo.all()
    assert len(rows) == 1
    row = rows[0]
    assert row.id is not None
    assert row.setup_name == "vcp_3_contracciones"
    assert row.family == "vcp"
    assert row.grade is None
    assert row.market_regime is None
    assert row.n_observations == 35
    assert row.trigger_rate == 0.6
    assert row.win_rate == 0.55
    assert row.expectancy_r == 0.42
    assert row.median_bars_held == 6.0
    assert row.mae_p80_pct == -0.03
    assert row.failure_rate_3d == 0.1
    assert row.confidence == "measured"


def test_replace_all_replaces_the_whole_table_not_accumulates(db_session: Session):
    repo = SetupPerformanceRepository(db_session)

    repo.replace_all([_row(setup_name="vcp_3_contracciones"), _row(setup_name="caja_de_darvas")])
    assert len(repo.all()) == 2

    # Una segunda corrida del estudio con un solo setup medido - la
    # primera foto (dos filas) debe desaparecer entera, no quedar mezclada.
    repo.replace_all([_row(setup_name="ruptura_de_nivel")])

    rows = repo.all()
    assert len(rows) == 1
    assert rows[0].setup_name == "ruptura_de_nivel"


def test_replace_all_keeps_segmented_rows_for_the_same_setup_name_distinct(db_session: Session):
    # grade=None (la fila sin segmentar) y grade="A" para el MISMO nombre -
    # exactamente el caso que una restricción UNIQUE con NULL manejaría mal.
    repo = SetupPerformanceRepository(db_session)

    repo.replace_all(
        [
            _row(setup_name="vcp_3_contracciones", grade=None, n_observations=35),
            _row(setup_name="vcp_3_contracciones", grade="A", n_observations=20),
            _row(setup_name="vcp_3_contracciones", grade="B", n_observations=15),
        ]
    )

    rows = repo.all()
    assert len(rows) == 3
    by_grade = {row.grade: row.n_observations for row in rows}
    assert by_grade == {None: 35, "A": 20, "B": 15}


def test_all_returns_empty_list_before_any_study_has_run(db_session: Session):
    repo = SetupPerformanceRepository(db_session)
    assert repo.all() == []


def test_replace_all_round_trips_none_metrics_for_a_setup_that_never_triggered(db_session: Session):
    repo = SetupPerformanceRepository(db_session)
    repo.replace_all(
        [
            _row(
                setup_name="caja_de_darvas", n_observations=5, trigger_rate=0.0, win_rate=None,
                expectancy_r=None, median_bars_held=None, mae_p80_pct=None, failure_rate_3d=None,
                confidence="thin",
            )
        ]
    )

    row = repo.all()[0]
    assert row.win_rate is None
    assert row.expectancy_r is None
    assert row.median_bars_held is None
    assert row.mae_p80_pct is None
    assert row.failure_rate_3d is None
    assert row.confidence == "thin"
