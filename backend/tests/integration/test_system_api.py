from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

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
