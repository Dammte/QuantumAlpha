"""Reconstruction (2026-09), Fase 5: GET /market/radar - a pure read over
`daily_close.py`'s own precomputed `ticker_daily_states`, never a live
universe scan. See docs/quant_methodology.md §25 and the endpoint's own
docstring."""

from datetime import UTC, date, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.domain.models.ticker_daily_state import TickerDailyState
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository


def _seed_state(db: Session, **overrides) -> TickerDailyState:
    defaults = dict(
        id=None,
        region="us",
        ticker="AAPL",
        trade_date=date.today(),
        computed_at=datetime.now(UTC),
        price=120.0,
        currency="USD",
        trend="uptrend",
        stage="stage2",
        rs_rating=85,
        adx14=28.0,
        atr_multiple=1.2,
        rsi14=55.0,
        gate_passes=True,
        gate_conditions=[
            {"label": "Tendencia alcista o Fase 2 de Weinstein", "passed": True},
            {"label": "Sin extensión parabólica (ATR múltiplo <= 4)", "passed": True},
        ],
        gate_version="2026-09-levels-v1",
        entry_trigger_type="breakout",
        entry_trigger_price=125.0,
        entry_already_triggered=False,
        stop_loss=110.0,
        take_profit=140.0,
        take_profit_method="objetivo 2:1 sobre el riesgo",
        risk_reward=2.0,
    )
    defaults.update(overrides)
    return TickerDailyStateRepository(db).upsert(TickerDailyState(**defaults))


def test_radar_returns_a_passing_gate_ticker(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session)

    response = client.get("/api/v1/market/radar?region=us")

    assert response.status_code == 200
    body = response.json()
    assert body["computed_at"] is not None
    tickers = {item["ticker"] for item in body["items"]}
    assert "AAPL" in tickers
    aapl = next(item for item in body["items"] if item["ticker"] == "AAPL")
    assert aapl["gate_passes"] is True
    assert len(aapl["gate_conditions"]) == 2
    assert aapl["entry_trigger"]["trigger_type"] == "breakout"
    assert aapl["stop_and_target"]["stop_loss"] == 110.0


def test_radar_includes_a_failing_gate_ticker_with_an_active_trigger(
    client: TestClient, db_session: Session
) -> None:
    # "About to trigger" (Parte 0, pregunta 2) is about the trigger, not
    # necessarily a fully-passing gate - a ticker approaching a breakout
    # level still belongs on the radar even if e.g. it's currently
    # overextended.
    _seed_state(
        db_session, ticker="MSFT", gate_passes=False, entry_trigger_type="breakout", entry_trigger_price=410.0
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    tickers = {item["ticker"] for item in body["items"]}
    assert "MSFT" in tickers
    msft = next(item for item in body["items"] if item["ticker"] == "MSFT")
    assert msft["gate_passes"] is False


def test_radar_excludes_a_ticker_with_no_trigger_and_a_failing_gate(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(
        db_session, ticker="XYZ", gate_passes=False, entry_trigger_type=None, entry_trigger_price=None,
        stop_loss=None, take_profit=None, take_profit_method=None, risk_reward=None,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    tickers = {item["ticker"] for item in body["items"]}
    assert "XYZ" not in tickers


def test_radar_computed_at_is_none_when_nothing_has_run_yet(client: TestClient) -> None:
    body = client.get("/api/v1/market/radar?region=us").json()
    assert body["items"] == []
    assert body["computed_at"] is None


def test_radar_is_scoped_by_region(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="AAPL", region="us")
    _seed_state(db_session, ticker="EURO.DE", region="europe")

    us_body = client.get("/api/v1/market/radar?region=us").json()
    europe_body = client.get("/api/v1/market/radar?region=europe").json()

    assert {item["ticker"] for item in us_body["items"]} == {"AAPL"}
    assert {item["ticker"] for item in europe_body["items"]} == {"EURO.DE"}


def test_radar_only_shows_the_latest_row_per_ticker(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="AAPL", trade_date=date(2020, 1, 1), gate_passes=False, entry_trigger_type=None)
    _seed_state(db_session, ticker="AAPL", trade_date=date.today(), gate_passes=True, price=150.0)

    body = client.get("/api/v1/market/radar?region=us").json()

    matches = [item for item in body["items"] if item["ticker"] == "AAPL"]
    assert len(matches) == 1
    assert matches[0]["price"] == 150.0
    assert matches[0]["gate_passes"] is True
