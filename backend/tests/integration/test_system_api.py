from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core import trading_params as tp
from app.domain.models.trigger_event import TriggerEvent
from app.infrastructure.db.repositories.trigger_event_repository import TriggerEventRepository


def test_signal_performance_empty_when_no_snapshots_exist_yet(client: TestClient) -> None:
    response = client.get("/api/v1/system/signal-performance")
    assert response.status_code == 200
    body = response.json()
    assert body["verdict_outcomes"] == []
    assert body["signal_outcomes"] == []
    assert body["false_negatives"] == []
    assert body["trigger_outcomes"] == []
    assert "as_of" in body


def test_signal_performance_reflects_a_position_signal_snapshot(client: TestClient) -> None:
    """A /risk evaluation writes a PositionSignalSnapshot (Fase 0) - confirms
    the whole pipeline (write on a fresh evaluation, read back through the
    aggregation service) round-trips through the real DB without erroring,
    even though the forward-return windows for a same-day snapshot can't
    have resolved into any outcome rows yet."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    risk_response = client.get(f"/api/v1/portfolios/{portfolio_id}/risk")
    assert risk_response.status_code == 200

    response = client.get("/api/v1/system/signal-performance")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["verdict_outcomes"], list)
    assert isinstance(body["signal_outcomes"], list)
    assert isinstance(body["false_negatives"], list)
    assert isinstance(body["trigger_outcomes"], list)


def test_signal_performance_reflects_a_trigger_event(client: TestClient, db_session: Session) -> None:
    """Fase 8: the whole pipeline (a `daily_close.py`-style TriggerEvent
    written to the real DB, read back through trigger_performance_service's
    aggregation) round-trips without erroring, using the fake provider's
    deterministic AAPL price history to actually resolve a forward return."""
    TriggerEventRepository(db_session).record(
        TriggerEvent(
            id=None,
            entity_type="ticker",
            entity_key="AAPL",
            event_type="gate_passed",
            previous_value="False",
            new_value="True",
            occurred_at=datetime(2024, 1, 2, tzinfo=UTC),
            details={"price": 100.0},
        )
    )

    response = client.get("/api/v1/system/signal-performance")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["trigger_outcomes"], list)


def test_signal_performance_splits_entry_triggered_by_taken(client: TestClient, db_session: Session) -> None:
    """Parte 13: an entry_triggered TriggerEvent followed by a real BUY of
    the same ticker within TAKEN_WINDOW_DAYS shows up as its own taken=True
    row, end to end through the real portfolio/transaction repositories -
    not just `trigger_performance_service`'s own unit tests."""
    TriggerEventRepository(db_session).record(
        TriggerEvent(
            id=None,
            entity_type="ticker",
            entity_key="AAPL",
            event_type="entry_triggered",
            previous_value=None,
            new_value="125.40",
            occurred_at=datetime(2024, 1, 2, tzinfo=UTC),
            details={"price": 125.40},
        )
    )
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main", "base_currency": "USD"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={
            "ticker": "AAPL",
            "transaction_type": "buy",
            "quantity": 10,
            "price": 125.40,
            "executed_at": "2024-01-03T00:00:00Z",
        },
    )

    body = client.get("/api/v1/system/signal-performance").json()

    entry_triggered_rows = [o for o in body["trigger_outcomes"] if o["event_type"] == "entry_triggered"]
    assert any(o["taken"] is True for o in entry_triggered_rows)


def test_trading_params_mirrors_the_core_module_exactly(client: TestClient) -> None:
    """Parte 19: a pure read, no DB/network involved - every field matches
    `app.core.trading_params`'s own constant value, so the UI never has a
    second, driftable copy of these numbers to keep in sync by hand."""
    response = client.get("/api/v1/system/params")

    assert response.status_code == 200
    body = response.json()
    assert body["risk_per_trade_pct"] == tp.RISK_PER_TRADE_PCT
    assert body["max_position_pct"] == tp.MAX_POSITION_PCT
    assert body["max_aggregate_risk_pct"] == tp.MAX_AGGREGATE_RISK_PCT
    assert body["max_open_positions"] == tp.MAX_OPEN_POSITIONS
    assert body["min_position_usd"] == tp.MIN_POSITION_USD
    assert body["min_position_for_scaling"] == tp.MIN_POSITION_FOR_SCALING
    assert body["transaction_cost_pct"] == tp.TRANSACTION_COST_PCT
    assert body["stop_atr_ceiling"] == tp.STOP_ATR_CEILING
    assert body["chandelier_window"] == tp.CHANDELIER_WINDOW
    assert body["chandelier_mult_by_vol"] == tp.CHANDELIER_MULT_BY_VOL
    assert body["chandelier_profit_lock_r"] == tp.CHANDELIER_PROFIT_LOCK_R
    assert body["high_correlation_threshold"] == tp.HIGH_CORRELATION_THRESHOLD
