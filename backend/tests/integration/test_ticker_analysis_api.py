from fastapi.testclient import TestClient


def test_ticker_analysis_returns_full_payload(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/AAPL/analysis")
    assert response.status_code == 200
    body = response.json()

    assert body["ticker"] == "AAPL"
    assert body["price"] > 0
    assert body["trend"] in {"uptrend", "downtrend", "sideways"}
    assert len(body["price_history"]) > 0
    first_point = body["price_history"][0]
    assert {"close", "sma20", "bb_upper"} <= first_point.keys()

    gate = body["gate"]
    assert isinstance(gate["passes"], bool)
    assert len(gate["conditions"]) == 6
    assert {"label", "passed"} <= gate["conditions"][0].keys()

    # Parte 7: AAPL's fake OHLCV history comfortably clears the EMA55 warm-up,
    # so evaluate_gate always gets real ema21/ema55 here - entry_geometry is
    # never sized (no capital in scope at this endpoint, see its own
    # docstring), but the rest of the fields must be there.
    geometry = gate["entry_geometry"]
    assert geometry is not None
    assert geometry["shares_for_risk_budget"] is None
    assert geometry["position_value"] is None
    assert isinstance(geometry["viable"], bool)

    assert body["fundamentals"]["name"] == "AAPL Inc."
    assert len(body["news"]) > 0

    # Fase 7: no GEMINI_API_KEY configured in tests (GeminiNarrator's
    # "cableado, no activado" default) - never null-crashes the endpoint,
    # just comes back empty.
    assert body["llm_narrative"] is None


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


def test_ticker_analysis_passing_gate_includes_stop_loss(client: TestClient) -> None:
    # The fake provider's random walk is ticker-seeded, so different tickers land in
    # different technical states - just assert the invariant: whenever the gate
    # passes, a stop-loss must be present (never a naked buy signal).
    for ticker in ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]:
        body = client.get(f"/api/v1/market/tickers/{ticker}/analysis").json()
        if body["gate"]["passes"]:
            stop_and_target = body["gate"]["stop_and_target"]
            assert stop_and_target["stop_loss"] is not None
            assert stop_and_target["stop_loss"] < body["price"]


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
    # 2026-09 (reconstruction, Fase 4): the live read behind this snapshot is
    # now GateResult, remapped onto this table's unchanged verdict/score/
    # factors columns (see the endpoint's own docstring) - "evitar" simply
    # never occurs anymore (the gate has no third, worse-than-failing state),
    # so the enum tightens to the two values that actually get written.
    client.get("/api/v1/market/tickers/AAPL/analysis")
    response = client.get("/api/v1/market/tickers/AAPL/history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) >= 1
    latest = history[0]
    assert latest["ticker"] == "AAPL"
    assert latest["verdict"] in {"comprar", "esperar"}
    assert latest["horizon"] == "3m"
    assert latest["engine_version"]
    assert len(latest["factors"]) == 6
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
