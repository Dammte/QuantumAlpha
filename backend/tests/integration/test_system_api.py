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


def test_factor_ablation_report_reads_the_real_saved_csv(client: TestClient) -> None:
    """Segunda auditoría, Bloque 4: docs/factor_ablation_report_v2_h21.csv is
    a real file checked into the repo (scripts/factor_ablation_study.py's
    own last saved run) - this reads it back through the real filesystem,
    not a fixture."""
    response = client.get("/api/v1/system/factor-ablation", params={"horizon_days": 21})
    assert response.status_code == 200
    body = response.json()
    assert body["horizon_days"] == 21
    assert len(body["results"]) > 0
    for row in body["results"]:
        assert isinstance(row["directionally_consistent"], bool)
        assert isinstance(row["current_points"], int)


def test_factor_ablation_report_defaults_to_horizon_21(client: TestClient) -> None:
    response = client.get("/api/v1/system/factor-ablation")
    assert response.status_code == 200
    assert response.json()["horizon_days"] == 21


def test_factor_ablation_report_empty_for_a_horizon_never_measured(client: TestClient) -> None:
    response = client.get("/api/v1/system/factor-ablation", params={"horizon_days": 999})
    assert response.status_code == 200
    assert response.json()["results"] == []
