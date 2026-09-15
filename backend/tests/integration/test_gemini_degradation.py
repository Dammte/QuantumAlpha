"""Parte 12/17 acceptance criterion, made explicit and auditable as its own
test rather than left implicit in "every integration test happens to run
without a key": boots the app with no `GEMINI_API_KEY` (the actual test
default - `.env.example` ships it commented out, `conftest.py` never sets
one) and confirms a representative spread of endpoints - including the two
that actually run `GeminiNarrator.explain_gate` under the hood
(`compute_core_signals`, shared by "Analizar activo" and portfolio risk) -
all respond normally. Parte 12's own hard rule: Gemini's absence must never
break anything, and there must be a test that verifies exactly that."""

from fastapi.testclient import TestClient

from app.core.config import get_settings


def test_gemini_api_key_is_unset_in_the_test_environment() -> None:
    # The premise every other test in this file depends on - if this ever
    # stops being true (a real key configured in the test environment), the
    # rest of this file would silently stop testing the "no key" path at all.
    assert get_settings().gemini_api_key is None


def _create_portfolio_with_a_position(client: TestClient) -> int:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main", "base_currency": "USD"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    return portfolio_id


def test_ticker_analysis_responds_without_a_gemini_key(client: TestClient) -> None:
    # The one endpoint that actually calls GeminiNarrator.explain_gate.
    response = client.get("/api/v1/market/tickers/AAPL/analysis")
    assert response.status_code == 200
    assert response.json()["llm_narrative"] is None


def test_portfolio_risk_responds_without_a_gemini_key(client: TestClient) -> None:
    # Reuses compute_core_signals (same Gemini call path as "Analizar
    # activo") for every held position - must not fail just because a
    # portfolio, not a bare ticker search, is what triggers it this time.
    portfolio_id = _create_portfolio_with_a_position(client)
    response = client.get(f"/api/v1/portfolios/{portfolio_id}/risk")
    assert response.status_code == 200
    assert len(response.json()["positions"]) == 1


def test_portfolio_today_responds_without_a_gemini_key(client: TestClient) -> None:
    portfolio_id = _create_portfolio_with_a_position(client)
    response = client.get(f"/api/v1/portfolios/{portfolio_id}/today")
    assert response.status_code == 200


def test_radar_responds_without_a_gemini_key(client: TestClient) -> None:
    response = client.get("/api/v1/market/radar")
    assert response.status_code == 200


def test_market_screener_responds_without_a_gemini_key(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener")
    assert response.status_code == 200


def test_system_params_responds_without_a_gemini_key(client: TestClient) -> None:
    response = client.get("/api/v1/system/params")
    assert response.status_code == 200


def test_system_signal_performance_responds_without_a_gemini_key(client: TestClient) -> None:
    response = client.get("/api/v1/system/signal-performance")
    assert response.status_code == 200
