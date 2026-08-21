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


def test_ticker_analysis_includes_multi_timeframe_and_triple_barrier_backtest(client: TestClient) -> None:
    """Segunda auditoría, Bloque 2: before this, "Analizar activo" never
    referenced analyze_multi_timeframe/closed_bars, and run_triple_barrier_backtest
    had zero callers anywhere. 10 years of fake daily bars comfortably clears
    both modules' minimum history, so this exercises the real end-to-end wiring,
    not just the unit-level pure functions."""
    response = client.get("/api/v1/market/tickers/AAPL/analysis")
    assert response.status_code == 200
    body = response.json()

    mtf = body["multi_timeframe"]
    assert mtf is not None
    assert mtf["daily"] is not None
    assert mtf["alignment"] in {"bullish_aligned", "bearish_aligned", "conflicted", "transitioning"}

    tbb = body["triple_barrier_backtest"]
    assert tbb is not None
    assert tbb["horizon_days"] == 21
    assert tbb["n_signals_evaluated"] >= 0
    for bucket in ("strategy_fixed", "strategy_trailing", "buy_and_hold", "random_entries"):
        assert "n_trades" in tbb[bucket]


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


def test_ticker_analysis_includes_entry_timing(client: TestClient) -> None:
    body = client.get("/api/v1/market/tickers/AAPL/analysis").json()
    assert "entry_timing" in body
    timing = body["entry_timing"]
    # Only None when atr_multiple itself couldn't be computed - 10 years of fake
    # daily bars is comfortably enough, so this should be populated.
    assert timing is not None
    assert timing["status"] in {"optimal", "valid", "late", "extended"}
    assert timing["atr_multiple"] >= 0


def test_ticker_analysis_includes_statistical_structure_and_regime_context(client: TestClient) -> None:
    body = client.get("/api/v1/market/tickers/AAPL/analysis").json()
    # These may legitimately be null (e.g. insufficient history for Hurst/ADF,
    # or no benchmark data) - the point is the keys exist and are well-formed
    # when present, not that they're always non-null against fake data.
    assert "statistical_structure" in body
    assert "market_trend" in body
    assert "vix_regime" in body
    assert "is_intraday_snapshot" in body
    assert isinstance(body["is_intraday_snapshot"], bool)
    structure = body["statistical_structure"]
    if structure is not None:
        assert structure["regime"] in {"tendencial", "reversion", "aleatorio", "desconocido"}


def test_ticker_analysis_persists_a_recommendation_snapshot(client: TestClient) -> None:
    client.get("/api/v1/market/tickers/AAPL/analysis")
    response = client.get("/api/v1/market/tickers/AAPL/history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) >= 1
    latest = history[0]
    assert latest["ticker"] == "AAPL"
    assert latest["verdict"] in {"comprar", "esperar", "evitar"}
    assert latest["horizon"] == "3m"
    assert latest["engine_version"]
    assert len(latest["factors"]) > 0
    assert {"label", "points", "triggered"} <= latest["factors"][0].keys()


def test_ticker_analysis_history_is_most_recent_first_and_ticker_scoped(client: TestClient) -> None:
    client.get("/api/v1/market/tickers/MSFT/analysis?horizon=1m")
    client.get("/api/v1/market/tickers/MSFT/analysis?horizon=6m")

    history = client.get("/api/v1/market/tickers/MSFT/history").json()
    assert len(history) >= 2
    assert all(h["ticker"] == "MSFT" for h in history)
    # Most recent call (horizon=6m) must come first.
    assert history[0]["horizon"] == "6m"
    timestamps = [h["created_at"] for h in history]
    assert timestamps == sorted(timestamps, reverse=True)


def test_ticker_analysis_history_empty_for_a_ticker_never_analyzed(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/NEVERSEEN/history")
    assert response.status_code == 200
    assert response.json() == []
