import pytest
from fastapi.testclient import TestClient


def test_ticker_analysis_returns_full_payload(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/AAPL/analysis")
    assert response.status_code == 200
    body = response.json()

    assert body["ticker"] == "AAPL"
    assert body["price"] > 0
    assert body["trend"] in {"uptrend", "downtrend", "sideways"}
    assert len(body["seasonality"]) == 12
    assert len(body["price_history"]) > 0
    first_point = body["price_history"][0]
    assert {"close", "sma20", "bb_upper", "gann_1x1"} <= first_point.keys()

    recommendation = body["recommendation"]
    assert recommendation["verdict"] in {"comprar", "esperar", "evitar"}
    assert len(recommendation["factors"]) > 0

    assert body["fundamentals"]["name"] == "AAPL Inc."
    assert len(body["news"]) > 0

    # 10 years of fake daily bars comfortably clears every module's minimum
    # history requirement, so these should come back populated (not None) -
    # this is what actually exercises the GARCH optimizer, the Markov chain
    # estimation and the Monte Carlo simulator end-to-end through the API.
    garch = body["garch"]
    assert garch is not None
    assert garch["regime"] in {"baja", "normal", "elevada", "alta"}
    assert 0.0 <= garch["vol_percentile"] <= 1.0

    markov = body["markov"]
    assert markov is not None
    assert 0 <= markov["current_state"] < len(markov["state_labels"])
    assert len(markov["transition_matrix"]) == len(markov["state_labels"])

    monte_carlo = body["monte_carlo"]
    assert monte_carlo is not None
    assert monte_carlo["method"] in {"garch_filtered", "block_bootstrap"}
    assert len(monte_carlo["percentiles"]) > 0
    for p in monte_carlo["percentiles"]:
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]

    if recommendation["verdict"] == "comprar":
        assert monte_carlo["probability_stop_before_target"] is not None
        assert monte_carlo["probability_target_before_stop"] is not None


def test_ticker_analysis_position_sizing_present_only_for_buy_verdicts(client: TestClient) -> None:
    for ticker in ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]:
        body = client.get(f"/api/v1/market/tickers/{ticker}/analysis").json()
        if body["recommendation"]["verdict"] != "comprar":
            continue
        sizing = body["position_sizing"]
        if sizing is None:
            continue  # can legitimately be absent if the barrier simulation never resolves
        assert 0.0 <= sizing["recommended_position_pct"] <= 0.25
        assert sizing["reward_risk_ratio"] == pytest.approx(body["recommendation"]["risk_reward"])


def test_ticker_analysis_horizon_query_param_changes_monte_carlo_window(client: TestClient) -> None:
    default_body = client.get("/api/v1/market/tickers/AAPL/analysis").json()
    short_body = client.get("/api/v1/market/tickers/AAPL/analysis?horizon=1m").json()
    long_body = client.get("/api/v1/market/tickers/AAPL/analysis?horizon=6m").json()

    default_max_day = max(p["day"] for p in default_body["monte_carlo"]["percentiles"])
    short_max_day = max(p["day"] for p in short_body["monte_carlo"]["percentiles"])
    long_max_day = max(p["day"] for p in long_body["monte_carlo"]["percentiles"])

    assert default_max_day == 63
    assert short_max_day == 21
    assert long_max_day == 126


def test_ticker_analysis_invalid_horizon_returns_422(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/AAPL/analysis?horizon=bogus")
    assert response.status_code == 422


def test_ticker_analysis_unknown_ticker_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/UNKNOWN/analysis")
    assert response.status_code == 404


def test_ticker_analysis_buy_verdict_includes_stop_loss(client: TestClient) -> None:
    # The fake provider's random walk is ticker-seeded, so different tickers land in
    # different technical states - just assert the invariant: whenever the verdict
    # is "comprar", a stop-loss must be present (never a naked buy signal).
    for ticker in ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]:
        body = client.get(f"/api/v1/market/tickers/{ticker}/analysis").json()
        if body["recommendation"]["verdict"] == "comprar":
            assert body["recommendation"]["stop_loss"] is not None
            assert body["recommendation"]["stop_loss"] < body["price"]
