"""Reconstruction (2026-09), Fase 5: GET /portfolios/{id}/today - a pure read
over `daily_close.py`'s own precomputed `position_daily_states`/
`daily_briefs`, additive alongside the existing live `/risk` endpoint. See
docs/quant_methodology.md §25 and the endpoint's own docstring."""

from datetime import UTC, date, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.domain.models.daily_brief import DailyBrief
from app.domain.models.position_daily_state import PositionDailyState
from app.infrastructure.db.repositories.daily_brief_repository import DailyBriefRepository
from app.infrastructure.db.repositories.position_daily_state_repository import (
    PositionDailyStateRepository,
)


def _create_portfolio(client: TestClient, name: str = "Main") -> int:
    response = client.post("/api/v1/portfolios", json={"name": name, "base_currency": "USD"})
    return response.json()["id"]


def _seed_position_state(db: Session, portfolio_id: int, **overrides) -> PositionDailyState:
    defaults = dict(
        id=None,
        portfolio_id=portfolio_id,
        ticker="AAPL",
        trade_date=date.today(),
        computed_at=datetime.now(UTC),
        urgency="hold",
        reasons=["Tendencia intacta, sin acción requerida"],
        price=120.0,
        r_multiple=1.5,
        current_stop=110.0,
        engine_version="2026-09-levels-v1",
    )
    defaults.update(overrides)
    return PositionDailyStateRepository(db).upsert(PositionDailyState(**defaults))


def _seed_brief(db: Session, portfolio_id: int, **overrides) -> DailyBrief:
    defaults = dict(
        id=None,
        portfolio_id=portfolio_id,
        brief_date=date.today(),
        computed_at=datetime.now(UTC),
        positions_needing_action=1,
        new_entry_triggers=2,
        new_gate_passes=1,
        headline="1 posición necesita atención hoy",
    )
    defaults.update(overrides)
    return DailyBriefRepository(db).upsert(DailyBrief(**defaults))


def test_today_returns_404_for_unknown_portfolio(client: TestClient) -> None:
    response = client.get("/api/v1/portfolios/999/today")
    assert response.status_code == 404


def test_today_returns_empty_brief_and_positions_before_daily_close_has_run(
    client: TestClient,
) -> None:
    portfolio_id = _create_portfolio(client)

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/today")

    assert response.status_code == 200
    body = response.json()
    assert body["brief"] is None
    assert body["positions"] == []


def test_today_returns_the_precomputed_brief_and_positions(
    client: TestClient, db_session: Session
) -> None:
    portfolio_id = _create_portfolio(client)
    _seed_position_state(db_session, portfolio_id, urgency="exit_now", reasons=["Stop de Chandelier roto"])
    _seed_brief(db_session, portfolio_id, headline="1 posición necesita atención hoy")

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/today").json()

    assert body["brief"]["headline"] == "1 posición necesita atención hoy"
    assert body["brief"]["positions_needing_action"] == 1
    assert len(body["positions"]) == 1
    position = body["positions"][0]
    assert position["ticker"] == "AAPL"
    assert position["urgency"] == "exit_now"
    assert position["reasons"] == ["Stop de Chandelier roto"]


def test_today_is_scoped_by_portfolio(client: TestClient, db_session: Session) -> None:
    portfolio_a = _create_portfolio(client, name="Portfolio A")
    portfolio_b = _create_portfolio(client, name="Portfolio B")
    _seed_position_state(db_session, portfolio_a, ticker="AAPL")
    _seed_position_state(db_session, portfolio_b, ticker="MSFT")

    body_a = client.get(f"/api/v1/portfolios/{portfolio_a}/today").json()
    body_b = client.get(f"/api/v1/portfolios/{portfolio_b}/today").json()

    assert {p["ticker"] for p in body_a["positions"]} == {"AAPL"}
    assert {p["ticker"] for p in body_b["positions"]} == {"MSFT"}


def test_today_only_shows_the_latest_row_per_ticker(client: TestClient, db_session: Session) -> None:
    portfolio_id = _create_portfolio(client)
    _seed_position_state(db_session, portfolio_id, trade_date=date(2020, 1, 1), urgency="exit_now", price=100.0)
    _seed_position_state(db_session, portfolio_id, trade_date=date.today(), urgency="hold", price=150.0)

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/today").json()

    matches = [p for p in body["positions"] if p["ticker"] == "AAPL"]
    assert len(matches) == 1
    assert matches[0]["urgency"] == "hold"
    assert matches[0]["price"] == 150.0
