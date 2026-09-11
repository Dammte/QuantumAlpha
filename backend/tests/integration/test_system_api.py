from fastapi.testclient import TestClient


def test_signal_performance_empty_when_no_snapshots_exist_yet(client: TestClient) -> None:
    response = client.get("/api/v1/system/signal-performance")
    assert response.status_code == 200
    body = response.json()
    assert body["verdict_outcomes"] == []
    assert body["signal_outcomes"] == []
    assert body["false_negatives"] == []
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
