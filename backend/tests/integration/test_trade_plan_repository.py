"""Auditoria del Radar, bloque H2: validación en la capa de persistencia de
que un stop inicial nunca se guarda en o por encima del precio de entrada -
`compute_entry_geometry` ya lo impide en el cálculo, pero
`TradePlanRepository.create` es el último punto antes de tocar la base de
datos, y debe negarse a persistir esa combinación sin importar de dónde venga
la llamada (defensa en profundidad, no solo confianza en el llamador). Usa las
mismas fixtures `engine`/`db_session` SQLite de `tests/integration/conftest.py`
que los demás tests de repositorio - una Session real contra un esquema real
(en memoria), sin la app de FastAPI.
"""

from datetime import date

from sqlalchemy.orm import Session

from app.infrastructure.db.models import PortfolioORM
from app.infrastructure.db.repositories.trade_plan_repository import TradePlanRepository


def _make_portfolio(db_session: Session) -> int:
    portfolio = PortfolioORM(name="Test", base_currency="USD")
    db_session.add(portfolio)
    db_session.commit()
    db_session.refresh(portfolio)
    return portfolio.id


def test_create_never_persists_a_stop_at_or_above_entry_price(db_session: Session):
    portfolio_id = _make_portfolio(db_session)
    repo = TradePlanRepository(db_session)

    plan = repo.create(
        portfolio_id=portfolio_id, ticker="AAPL", entry_price=100.0, entry_date=date(2026, 1, 1),
        initial_stop=100.0,  # en el precio de entrada - geometría imposible
        initial_target=120.0, initial_quantity=10.0, thesis="", engine_version="v1",
        initial_stop_basis="bajo el soporte en 100.00", initial_stop_level_kind="pivot_support",
    )

    assert plan.initial_stop is None
    assert plan.current_stop is None
    assert plan.initial_target is None
    assert plan.initial_stop_basis is None
    assert plan.initial_stop_level_kind is None


def test_create_never_persists_a_stop_above_entry_price(db_session: Session):
    portfolio_id = _make_portfolio(db_session)
    repo = TradePlanRepository(db_session)

    plan = repo.create(
        portfolio_id=portfolio_id, ticker="AAPL", entry_price=100.0, entry_date=date(2026, 1, 1),
        initial_stop=105.0,  # por encima del precio de entrada
        initial_target=120.0, initial_quantity=10.0, thesis="", engine_version="v1",
    )

    assert plan.initial_stop is None
    assert plan.current_stop is None


def test_create_persists_a_real_stop_below_entry_price_untouched(db_session: Session):
    portfolio_id = _make_portfolio(db_session)
    repo = TradePlanRepository(db_session)

    plan = repo.create(
        portfolio_id=portfolio_id, ticker="AAPL", entry_price=100.0, entry_date=date(2026, 1, 1),
        initial_stop=95.0, initial_target=120.0, initial_quantity=10.0, thesis="", engine_version="v1",
        initial_stop_basis="bajo el soporte en 95.00", initial_stop_level_kind="pivot_support",
    )

    assert plan.initial_stop == 95.0
    assert plan.current_stop == 95.0
    assert plan.initial_target == 120.0
    assert plan.initial_stop_basis == "bajo el soporte en 95.00"
    assert plan.initial_stop_level_kind == "pivot_support"
    assert plan.current_stop_basis == "bajo el soporte en 95.00"


def test_update_trailing_leaves_the_basis_unchanged_when_none_is_given(db_session: Session):
    # `None` significa "sin anclaje nuevo que ofrecer" (Chandelier no
    # gobierna esta evaluación) - nunca "borra el anclaje existente".
    portfolio_id = _make_portfolio(db_session)
    repo = TradePlanRepository(db_session)
    plan = repo.create(
        portfolio_id=portfolio_id, ticker="AAPL", entry_price=100.0, entry_date=date(2026, 1, 1),
        initial_stop=95.0, initial_target=120.0, initial_quantity=10.0, thesis="", engine_version="v1",
        initial_stop_basis="bajo el soporte en 95.00", initial_stop_level_kind="pivot_support",
    )

    repo.update_trailing(plan.id, current_stop=96.0, highest_close_since_entry=101.0)
    reloaded = repo.get_open(portfolio_id, "AAPL")

    assert reloaded.current_stop == 96.0
    assert reloaded.current_stop_basis == "bajo el soporte en 95.00"


def test_update_trailing_overwrites_the_basis_when_the_chandelier_takes_over(db_session: Session):
    portfolio_id = _make_portfolio(db_session)
    repo = TradePlanRepository(db_session)
    plan = repo.create(
        portfolio_id=portfolio_id, ticker="AAPL", entry_price=100.0, entry_date=date(2026, 1, 1),
        initial_stop=95.0, initial_target=120.0, initial_quantity=10.0, thesis="", engine_version="v1",
        initial_stop_basis="bajo el soporte en 95.00", initial_stop_level_kind="pivot_support",
    )

    repo.update_trailing(
        plan.id, current_stop=110.0, highest_close_since_entry=120.0,
        current_stop_basis="Chandelier 2.00x ATR desde el máximo de 10 sesiones",
    )
    reloaded = repo.get_open(portfolio_id, "AAPL")

    assert reloaded.current_stop == 110.0
    assert reloaded.current_stop_basis == "Chandelier 2.00x ATR desde el máximo de 10 sesiones"
    # El ancla inicial nunca se toca - solo la del trailing en curso.
    assert reloaded.initial_stop_basis == "bajo el soporte en 95.00"
