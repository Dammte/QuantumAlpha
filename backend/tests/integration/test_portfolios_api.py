from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.infrastructure.db.models import ComputationCacheORM


def test_create_and_get_portfolio(client: TestClient) -> None:
    response = client.post("/api/v1/portfolios", json={"name": "Main", "base_currency": "USD"})
    assert response.status_code == 201
    portfolio_id = response.json()["id"]

    response = client.get(f"/api/v1/portfolios/{portfolio_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Main"


def test_get_unknown_portfolio_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/portfolios/999")
    assert response.status_code == 404


def test_add_transaction_and_summary(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]

    response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    assert response.status_code == 201

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/summary")
    assert response.status_code == 200
    body = response.json()
    assert len(body["positions"]) == 1
    position = body["positions"][0]
    assert position["ticker"] == "AAPL"
    assert position["quantity"] == 10
    assert position["current_price"] == 150.0
    assert position["unrealized_pnl"] == (150.0 - 100) * 10


def test_realized_pnl_from_a_full_sell(client: TestClient) -> None:
    """Buy 10 @ 100, sell all 10 @ 150: the $500 gain must show up as realized
    P&L even though the position is now fully closed (0 shares, not listed
    among `positions` at all) - this is exactly the "does my sell get counted"
    check the gain must survive after a position no longer exists."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "sell", "quantity": 10, "price": 150},
    )

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["positions"] == []
    assert body["total_realized_pnl"] == pytest.approx(500.0)
    assert body["total_unrealized_pnl"] == 0.0
    assert body["total_pnl"] == pytest.approx(500.0)


def test_realized_pnl_from_a_partial_sell_keeps_average_cost(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "sell", "quantity": 4, "price": 150},
    )

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["total_realized_pnl"] == pytest.approx((150 - 100) * 4)  # 200

    position = body["positions"][0]
    assert position["quantity"] == pytest.approx(6)
    assert position["average_cost"] == pytest.approx(100)  # unchanged by the sell
    assert position["unrealized_pnl"] == pytest.approx((150.0 - 100) * 6)  # fake quote price is 150

    # Total accumulated gain = realized (banked) + unrealized (still held)
    assert body["total_pnl"] == pytest.approx(body["total_realized_pnl"] + body["total_unrealized_pnl"])


def test_realized_pnl_accumulates_across_multiple_sells(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 20, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "sell", "quantity": 5, "price": 120},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "sell", "quantity": 5, "price": 90},
    )

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    expected = (120 - 100) * 5 + (90 - 100) * 5  # +100 then -50 = +50
    assert body["total_realized_pnl"] == pytest.approx(expected)


def test_selling_more_than_held_is_rejected(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 5, "price": 100},
    )

    response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "sell", "quantity": 10, "price": 150},
    )
    assert response.status_code == 400

    # The rejected sell must not have been recorded (realized P&L unaffected)
    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["total_realized_pnl"] == 0.0
    assert body["positions"][0]["quantity"] == 5


def test_position_with_unavailable_quote_degrades_gracefully(client: TestClient) -> None:
    """A ticker the provider can't quote right now (delisted, mistyped, a
    momentary hiccup) must show up as "unavailable" for that one position, not
    take down the whole portfolio summary with a 500."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "UNKNOWN", "transaction_type": "buy", "quantity": 10, "price": 100},
    )

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/summary")
    assert response.status_code == 200
    body = response.json()
    position = body["positions"][0]
    assert position["current_price"] is None
    assert position["market_value"] is None
    assert position["unrealized_pnl"] is None
    assert body["total_market_value"] == 0.0  # the unpriced position contributes nothing, not a crash


def test_daily_change_reported_on_summary(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    # Fake provider: price=150, previous_close=148
    position = body["positions"][0]
    assert position["day_change"] == pytest.approx((150.0 - 148.0) * 10)
    assert position["day_change_pct"] == pytest.approx((150.0 - 148.0) / 148.0)
    assert body["total_day_change"] == pytest.approx((150.0 - 148.0) * 10)


def test_multi_currency_positions_convert_to_base_currency_for_totals(client: TestClient) -> None:
    """A EUR-priced position must still show its own price in EUR (native,
    per-row) but contribute a properly FX-converted amount to the portfolio's
    USD totals - never dollars and euros summed as if they were the same
    unit. Fake provider: EURO.DE quotes at €100 (prev close €98), EURUSD
    fake rate is 1.1."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main", "base_currency": "USD"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "EURO.DE", "transaction_type": "buy", "quantity": 10, "price": 90},
    )

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    euro_position = next(p for p in body["positions"] if p["ticker"] == "EURO.DE")
    assert euro_position["currency"] == "EUR"
    assert euro_position["current_price"] == 100.0  # native EUR, not converted
    assert euro_position["market_value"] == 1000.0  # 10 x €100, still in EUR
    assert euro_position["market_value_base"] == pytest.approx(1000.0 * 1.1)  # converted for weights/totals

    aapl_market_value_usd = 10 * 150.0
    euro_market_value_usd = 10 * 100.0 * 1.1  # converted at the fake EURUSD rate
    assert body["total_market_value"] == pytest.approx(aapl_market_value_usd + euro_market_value_usd)


def test_portfolio_history(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={
            "ticker": "AAPL",
            "transaction_type": "buy",
            "quantity": 10,
            "price": 100,
            "executed_at": "2024-01-03T00:00:00",
        },
    )

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/history?start=2024-01-01&end=2024-01-05")
    assert response.status_code == 200
    body = response.json()
    assert body["base_currency"] == "USD"
    # The leading zero-value days (before the buy settles) are trimmed off the front.
    assert [p["date"] for p in body["points"]] == ["2024-01-03", "2024-01-04", "2024-01-05"]
    assert body["points"][0]["value"] == pytest.approx(10 * 102)
    assert body["benchmark_points"] is None


def test_portfolio_history_unknown_portfolio_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/portfolios/999/history")
    assert response.status_code == 404


def test_delete_transaction(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    tx = client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    ).json()

    response = client.delete(f"/api/v1/portfolios/{portfolio_id}/transactions/{tx['id']}")
    assert response.status_code == 204

    remaining = client.get(f"/api/v1/portfolios/{portfolio_id}/transactions").json()
    assert remaining == []

    response = client.delete(f"/api/v1/portfolios/{portfolio_id}/transactions/{tx['id']}")
    assert response.status_code == 404


def test_portfolio_risk_for_unknown_portfolio_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/portfolios/999/risk")
    assert response.status_code == 404


def test_portfolio_risk_empty_portfolio_returns_no_positions(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Empty"}).json()["id"]
    response = client.get(f"/api/v1/portfolios/{portfolio_id}/risk")
    assert response.status_code == 200
    body = response.json()
    assert body["positions"] == []
    assert body["swap_suggestions"] == []


def test_portfolio_risk_assesses_every_held_ticker(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "MSFT", "transaction_type": "buy", "quantity": 5, "price": 200},
    )

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/risk")
    assert response.status_code == 200
    body = response.json()
    positions = body["positions"]
    assert {p["ticker"] for p in positions} == {"AAPL", "MSFT"}
    for p in positions:
        assert p["signal"] in {"exit_warning", "add_candidate", "watch", "hold"}
        assert isinstance(p["score"], int)
        assert len(p["reasons"]) > 0
        # Full quant suite - the same one "Analizar activo" runs - backs every holding now
        assert {"garch", "markov", "monte_carlo", "backtest", "position_sizing"} <= p["signals"].keys()
        assert p["signals"]["recommendation"]["verdict"] in {"comprar", "esperar", "evitar"}

    held_tickers = {p["ticker"] for p in positions}
    for suggestion in body["swap_suggestions"]:
        assert suggestion["held_ticker"] in held_tickers
        assert suggestion["candidate_ticker"] not in held_tickers
        assert suggestion["candidate_score"] - suggestion["held_score"] >= 4.0


def test_portfolio_risk_includes_exit_engine_and_trade_plan_fields(client: TestClient) -> None:
    """The Paso C wiring (see exit_engine.py/trade_plan_service.py): a held
    position's risk read now also carries an independent exit_urgency,
    backed by a persisted/reconstructed trade plan, and a multi-timeframe
    alignment read - all additive to the legacy signal/score/reasons
    contract asserted above (test_portfolio_risk_assesses_every_held_ticker),
    never replacing it."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/risk")
    assert response.status_code == 200
    position = response.json()["positions"][0]

    assert position["exit_urgency"] in {"exit_now", "reduce", "tighten_stop", "watch", "hold"}
    assert isinstance(position["exit_reasons"], list)
    assert position["r_multiple"] is None or isinstance(position["r_multiple"], float)

    # No trade plan existed before this buy - assess_position_risk reconstructs
    # one lazily on first read (trade_plan_service.ensure_trade_plan), anchored
    # to this exact BUY transaction's own price.
    plan = position["trade_plan"]
    assert plan is not None
    assert plan["entry_price"] == pytest.approx(100.0)
    assert plan["thesis"]  # reconstructed - see trade_plan_service.RECONSTRUCTED_THESIS

    multi_timeframe = position["multi_timeframe"]
    assert multi_timeframe is not None
    assert multi_timeframe["daily"] is not None
    assert multi_timeframe["alignment"] in {"bullish_aligned", "bearish_aligned", "conflicted", "transitioning"}

    # Fase 7 (position detail card - "sesiones mantenidas"): a same-session
    # buy has held for 0 closed bars since entry, never None or negative once
    # a trade plan exists.
    assert position["bars_held"] is not None
    assert position["bars_held"] >= 0


def test_portfolio_risk_trailing_stop_persists_and_never_moves_down(client: TestClient) -> None:
    """Paso E (trade_manager.py): the Chandelier trailing stop persisted in
    trade_plans must round-trip through the real DB and never move down
    across evaluations, even when force-refreshed twice back to back (the
    fake provider's OHLCV is deterministic, so the stop should at worst stay
    exactly equal - the real invariant this locks in is that persistence
    and re-read never silently produce a *lower* value)."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )

    first = client.get(f"/api/v1/portfolios/{portfolio_id}/risk").json()
    first_plan = first["positions"][0]["trade_plan"]

    second = client.get(f"/api/v1/portfolios/{portfolio_id}/risk", params={"refresh": True}).json()
    second_plan = second["positions"][0]["trade_plan"]

    if first_plan["current_stop"] is not None and second_plan["current_stop"] is not None:
        assert second_plan["current_stop"] >= first_plan["current_stop"]

    scaled_exit = second["positions"][0]["scaled_exit"]
    assert scaled_exit is not None
    assert scaled_exit["action"] in {"none", "sell_at_1r", "sell_at_2r"}


def test_portfolio_risk_response_carries_computed_at(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )

    first = client.get(f"/api/v1/portfolios/{portfolio_id}/risk").json()
    assert "computed_at" in first

    # A second request without `refresh=true` must serve the same durably-cached
    # read (same computed_at), not silently recompute - see durable_cache.py.
    second = client.get(f"/api/v1/portfolios/{portfolio_id}/risk").json()
    assert second["computed_at"] == first["computed_at"]

    refreshed = client.get(f"/api/v1/portfolios/{portfolio_id}/risk", params={"refresh": True}).json()
    assert refreshed["computed_at"] >= first["computed_at"]


def test_selling_moves_proceeds_into_cash_balance(client: TestClient) -> None:
    """The literal bug this feature fixes: selling a position must not make its
    proceeds disappear from the portfolio's total - they become liquidity.

    No deposit is logged before the buy (the realistic case for basically
    every existing portfolio, since deposits didn't exist as a concept before
    this feature) - `cash_balance` floors at 0 rather than going to -1000,
    since this app has no concept of margin/debt and a real buy always has to
    be funded by cash that actually exists. See `build_portfolio`'s docstring
    for why flooring at every step (not just the end) matters: it's what
    makes the $1500 in proceeds show up in full below, not just $500."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["cash_balance"] == 0.0  # floored - no un-logged debt implied by the buy
    assert body["total_portfolio_value"] == pytest.approx(body["total_market_value"])

    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "sell", "quantity": 10, "price": 150},
    )
    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["positions"] == []
    assert body["total_market_value"] == 0.0  # nothing held any more...
    assert body["cash_balance"] == pytest.approx(1500.0)  # ...but the full $1500 sale proceeds became cash
    assert body["total_portfolio_value"] == pytest.approx(1500.0)


def test_cash_balance_never_goes_negative_without_a_logged_deposit(client: TestClient) -> None:
    """General property behind the fix above: however much is spent buying
    positions with no deposit ever logged to cover it, cash_balance floors at
    0 instead of accruing an implied debt this app has no concept of."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "MSFT", "transaction_type": "buy", "quantity": 5, "price": 200},
    )
    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["cash_balance"] == 0.0
    assert body["total_portfolio_value"] == pytest.approx(body["total_market_value"])

    # A withdrawal with nothing behind it floors the same way, for the same reason.
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"transaction_type": "withdrawal", "quantity": 500},
    )
    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["cash_balance"] == 0.0


def test_deposit_and_withdrawal_move_cash_balance(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]

    response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"transaction_type": "deposit", "quantity": 1000},
    )
    assert response.status_code == 201
    tx = response.json()
    assert tx["ticker"] is None
    assert tx["price"] == 1.0

    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["cash_balance"] == pytest.approx(1000.0)
    assert body["total_portfolio_value"] == pytest.approx(1000.0)

    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"transaction_type": "withdrawal", "quantity": 400},
    )
    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["cash_balance"] == pytest.approx(600.0)

    # Buying still spends from that cash balance, exactly like real capital.
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 5, "price": 100},
    )
    body = client.get(f"/api/v1/portfolios/{portfolio_id}/summary").json()
    assert body["cash_balance"] == pytest.approx(100.0)  # 600 - 500 spent
    assert body["total_portfolio_value"] == pytest.approx(body["total_market_value"] + 100.0)


def test_deposit_with_ticker_is_rejected(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "deposit", "quantity": 1000},
    )
    assert response.status_code == 422


def test_buy_without_price_is_rejected(client: TestClient) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10},
    )
    assert response.status_code == 422


def test_portfolio_risk_survives_a_missing_computation_cache_table(client: TestClient, engine: Engine) -> None:
    """Reproduces the exact production incident this test is named after: the
    durable cache's own table doesn't exist yet (a real state - the migration
    that creates it must be run by hand against Supabase, so there's a real
    window where the table is genuinely missing). A caught-but-not-rolled-back
    failure here leaves the session unusable for every query the endpoint
    makes afterward (fetching the portfolio's transactions), turning an
    optional-cache miss into an unrelated 500 - see durable_cache.py's
    module docstring. The recommendation column going blank in production was
    this exact bug."""
    ComputationCacheORM.__table__.drop(bind=engine)
    try:
        portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
        client.post(
            f"/api/v1/portfolios/{portfolio_id}/transactions",
            json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
        )

        response = client.get(f"/api/v1/portfolios/{portfolio_id}/risk")
        assert response.status_code == 200
        positions = response.json()["positions"]
        assert {p["ticker"] for p in positions} == {"AAPL"}

        # And a completely unrelated subsequent request on the same shared
        # session must still work too - the whole point of rolling back.
        summary = client.get(f"/api/v1/portfolios/{portfolio_id}/summary")
        assert summary.status_code == 200
    finally:
        ComputationCacheORM.__table__.create(bind=engine)


def test_portfolio_risk_recomputes_when_cached_payload_shape_is_stale(
    client: TestClient, db_session: Session
) -> None:
    """The real incident this locks in: `imminent_cross` was added to
    CoreSignalsResponse after some portfolio-risk rows were already cached,
    and every read of those rows 500ed on Pydantic validation until they
    naturally expired (up to PORTFOLIO_RISK_DURABLE_TTL later). A cached
    payload that's fresh by age but the wrong shape must be treated as a
    miss and recomputed, not crash the endpoint."""
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 10, "price": 100},
    )

    # Seeded directly, as if written by an older version of the app whose
    # PortfolioRiskResponse didn't have `swap_suggestions`/`computed_at` yet.
    db_session.add(
        ComputationCacheORM(
            cache_key=f"portfolio_risk:{portfolio_id}",
            computed_at=datetime.now(UTC),  # fresh by age - this must fail on *shape*, not staleness
            payload={"positions": []},
        )
    )
    db_session.commit()

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/risk")
    assert response.status_code == 200
    body = response.json()
    assert {p["ticker"] for p in body["positions"]} == {"AAPL"}  # recomputed for real, not the stale empty list


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
