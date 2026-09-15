"""Reconstruction (2026-09), Fase 5: GET /market/radar - a pure read over
`daily_close.py`'s own precomputed `ticker_daily_states`, never a live
universe scan. See docs/quant_methodology.md §25 and the endpoint's own
docstring."""

from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.trading_params import RISK_PER_TRADE_PCT
from app.domain.models.ticker_daily_state import TickerDailyState
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.services.portfolio_construction_service import MAX_SECTOR_CONCENTRATION_PCT

_VIABLE_GEOMETRY = {
    "entry_price": 50.0,
    "stop_price": 45.0,
    "stop_basis": "bajo el soporte en 45.00",
    "entry_type": "pullback_support",
    "risk_pct": 0.10,
    "risk_atr": 2.5,
    "risk_ceiling_pct": 0.07,
    "target_price": 56.0,
    "target_basis": "objetivo 2:1 sobre el riesgo",
    "reward_pct": 0.12,
    "risk_reward_gross": 2.0,
    "risk_reward_net": 1.9,
    "shares_for_risk_budget": None,
    "position_value": None,
    "pct_of_portfolio": None,
    "viable": True,
    "rejection_reason": None,
}


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


def test_radar_row_with_no_entry_geometry_is_none_pre_migration_row(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(db_session)
    aapl = client.get("/api/v1/market/radar?region=us").json()["items"][0]
    assert aapl["entry_geometry"] is None


def test_radar_exposes_the_unsized_entry_geometry_without_a_portfolio(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    geometry = nvda["entry_geometry"]
    assert geometry is not None
    assert geometry["viable"] is True
    assert geometry["entry_type"] == "pullback_support"
    assert geometry["stop_price"] == 45.0
    # Parte 7: never sized without a specific portfolio's capital in view.
    assert geometry["shares_for_risk_budget"] is None
    assert geometry["position_value"] is None


def test_radar_sizes_the_entry_geometry_against_a_portfolios_capital(
    client: TestClient, db_session: Session
) -> None:
    # `cash_balance` only ever reflects deposits/sale proceeds (see
    # `Portfolio.cash_balance`'s own docstring) - a buy alone never reduces
    # it, so `total_portfolio_value` here is simply the market value of what
    # was bought: 100 shares at the fake provider's fixed 150.0 AAPL quote.
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 100, "price": 100},
    )
    capital_total = 100 * 150.0
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    geometry = nvda["entry_geometry"]
    # _VIABLE_GEOMETRY's 10% risk_pct (entry 50, stop 45) keeps the fixed-risk
    # position comfortably under MAX_POSITION_PCT (0.01/0.10 = 10% < 15%), so
    # this exercises the plain `capital * RISK_PER_TRADE_PCT / risk_per_share`
    # formula, not the position-value cap.
    expected_shares = (capital_total * RISK_PER_TRADE_PCT) / (50.0 - 45.0)
    assert geometry["shares_for_risk_budget"] == pytest.approx(expected_shares)
    assert geometry["position_value"] == pytest.approx(expected_shares * 50.0)
    assert geometry["pct_of_portfolio"] == pytest.approx(geometry["position_value"] / capital_total)
    # Stop/target/entry_type themselves are untouched by sizing.
    assert geometry["stop_price"] == 45.0
    assert geometry["entry_type"] == "pullback_support"


def test_radar_with_portfolio_id_leaves_a_non_viable_geometry_unsized(
    client: TestClient, db_session: Session
) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    rejected = dict(_VIABLE_GEOMETRY, viable=False, rejection_reason="riesgo demasiado alto")
    _seed_state(db_session, ticker="NVDA", entry_geometry=rejected)

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    assert nvda["entry_geometry"]["viable"] is False
    assert nvda["entry_geometry"]["shares_for_risk_budget"] is None


def test_radar_with_unknown_portfolio_id_returns_404(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)
    response = client.get("/api/v1/market/radar?region=us&portfolio_id=999")
    assert response.status_code == 404


def test_radar_narrows_a_sized_candidate_by_the_portfolios_sector_concentration(
    client: TestClient, db_session: Session
) -> None:
    # `size_position` alone (the previous test) is blind to every *other*
    # open position - `portfolio_construction_service.apply_portfolio_limits`
    # is the "one layer up" narrowing its own docstring points to. MSFT is
    # "Tecnología", the same curated sector as the NVDA candidate below; TSLA
    # (a different sector) only dilutes total capital, at the fake provider's
    # shared 150.0 quote for both.
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "MSFT", "transaction_type": "buy", "quantity": 25, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "TSLA", "transaction_type": "buy", "quantity": 75, "price": 100},
    )
    capital_total = 150.0 * (25 + 75)  # = 15,000; MSFT is exactly 25% of it.
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    geometry = nvda["entry_geometry"]
    # size_position alone would size this at capital*RISK_PER_TRADE_PCT/5 = 30
    # shares - but Tecnología is already at 25% of capital, leaving only a 5%
    # sector headroom (30% cap), narrower than the risk-based 30 shares.
    sector_headroom_pct = MAX_SECTOR_CONCENTRATION_PCT - 0.25
    expected_shares = (sector_headroom_pct * capital_total) / 50.0
    assert expected_shares < (capital_total * RISK_PER_TRADE_PCT) / 5.0  # confirms the sector cap is what binds
    assert geometry["shares_for_risk_budget"] == pytest.approx(expected_shares)
    assert geometry["position_value"] == pytest.approx(expected_shares * 50.0)
    assert geometry["viable"] is True
