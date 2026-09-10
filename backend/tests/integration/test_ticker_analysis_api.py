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


def test_ticker_analysis_includes_regime_context(client: TestClient) -> None:
    body = client.get("/api/v1/market/tickers/AAPL/analysis").json()
    # These may legitimately be null (e.g. no benchmark data) - the point is
    # the keys exist and are well-formed when present, not that they're
    # always non-null against fake data.
    assert "market_trend" in body
    assert "vix_regime" in body
    assert "is_intraday_snapshot" in body
    assert isinstance(body["is_intraday_snapshot"], bool)


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
