from datetime import UTC, date, datetime

from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.infrastructure.db.models import ComputationCacheORM, UniverseMembershipORM
from app.services.market_universe import INDUSTRIES, universe_tickers


def test_get_universe(client: TestClient) -> None:
    response = client.get("/api/v1/market/universe")
    assert response.status_code == 200
    body = response.json()
    assert "Tecnología" in body["sectors"]
    assert "Semiconductores" in body["sectors"]["Tecnología"]
    industry_names = {i["name"] for i in body["industries"]}
    assert industry_names == {i.name for i in INDUSTRIES}
    semis = next(i for i in body["industries"] if i["name"] == "Semiconductores")
    assert "NVDA" in semis["tickers"]
    assert semis["etf"] == "SOXX"


def test_get_universe_europe_region(client: TestClient) -> None:
    response = client.get("/api/v1/market/universe", params={"region": "europe"})
    assert response.status_code == 200
    body = response.json()
    assert "Tecnología" in body["sectors"]
    # Europe's own industry breakdown, not the US one
    assert "Software y semiconductores" in body["sectors"]["Tecnología"]
    tech = next(i for i in body["industries"] if i["name"] == "Software y semiconductores")
    assert "SAP.DE" in tech["tickers"]
    assert tech["etf"] == "EXV3.DE"


def test_screener_returns_a_snapshot_per_universe_ticker(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(universe_tickers())
    first = body[0]
    assert {
        "ticker", "sector", "industry", "cap_tier", "currency", "price", "change_1d", "rsi14",
        "trend", "stage", "adx14", "rs_rating", "minervini_score", "minervini_pass",
    } <= first.keys()


def test_screener_europe_region_returns_a_snapshot_per_european_ticker(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"region": "europe"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(universe_tickers("europe"))
    tickers = {row["ticker"] for row in body}
    assert "SAP.DE" in tickers
    sap = next(row for row in body if row["ticker"] == "SAP.DE")
    assert sap["currency"] == "EUR"
    lse_names = {row["ticker"] for row in body if row["ticker"].endswith(".L")}
    assert lse_names, "expected at least some LSE tickers in the European universe"
    for ticker in lse_names:
        row = next(row for row in body if row["ticker"] == ticker)
        assert row["currency"] == "GBp"


def test_screener_rejects_invalid_region(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"region": "asia"})
    assert response.status_code == 422


def test_screener_filters_by_sector(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"sector": "Tecnología"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    assert all(row["sector"] == "Tecnología" for row in body)


def test_screener_filters_by_industry(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"industry": "Semiconductores"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    assert all(row["industry"] == "Semiconductores" for row in body)


def test_screener_filters_by_cap_tier(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"cap_tier": "mega"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    assert all(row["cap_tier"] == "mega" for row in body)


def test_screener_rs_rating_is_a_1_to_99_percentile(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener")
    ratings = [row["rs_rating"] for row in response.json() if row["rs_rating"] is not None]
    assert ratings, "expected at least some tickers to have an RS rating"
    assert all(1 <= r <= 99 for r in ratings)


def test_screener_sorts_ascending(client: TestClient) -> None:
    response = client.get(
        "/api/v1/market/screener", params={"sort_by": "change_1d", "sort_dir": "asc"}
    )
    values = [row["change_1d"] for row in response.json() if row["change_1d"] is not None]
    assert values == sorted(values)

def test_screener_sort_by_rs_rating(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"sort_by": "rs_rating", "sort_dir": "desc"})
    values = [row["rs_rating"] for row in response.json() if row["rs_rating"] is not None]
    assert values == sorted(values, reverse=True)


def test_screener_rejects_invalid_sort_field(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"sort_by": "not_a_field"})
    assert response.status_code == 400


def test_screener_price_filter(client: TestClient) -> None:
    response = client.get("/api/v1/market/screener", params={"min_price": 1_000_000})
    assert response.status_code == 200
    assert response.json() == []


def test_movers_has_all_groups_within_size_limit(client: TestClient) -> None:
    response = client.get("/api/v1/market/movers")
    assert response.status_code == 200
    body = response.json()
    expected_groups = {
        "gainers", "losers", "near_52w_high", "near_52w_low", "high_volume",
        "oversold", "overbought", "golden_cross", "death_cross", "rs_leaders", "strong_trend",
    }
    assert expected_groups <= body.keys()
    for group in expected_groups:
        assert len(body[group]) <= 10


def test_trend_breadth_totals_match_universe_size(client: TestClient) -> None:
    response = client.get("/api/v1/market/trend")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == len(universe_tickers())
    assert 0.0 <= body["pct_above_sma50"] <= 1.0
    assert 0.0 <= body["pct_above_sma200"] <= 1.0
    assert body["count_stage2"] >= 0
    assert body["count_minervini_pass"] >= 0


def test_trend_detail_returns_ticker_level_lists(client: TestClient) -> None:
    response = client.get("/api/v1/market/trend/detail")
    assert response.status_code == 200
    body = response.json()
    expected_groups = {
        "uptrend", "downtrend", "golden_cross", "death_cross", "overbought",
        "oversold", "stage2", "stage4", "minervini_pass", "strong_trend",
    }
    assert expected_groups <= body.keys()
    for group in expected_groups:
        assert isinstance(body[group], list)
        for row in body[group]:
            assert "ticker" in row and "price" in row


def test_trend_breadth_reports_imminent_cross_counts(client: TestClient) -> None:
    # Corto/mediano plazo (ago 2026): a projected SMA21/SMA50 crossover, not
    # yet confirmed - separate counts from the confirmed golden_crosses/death_crosses above.
    response = client.get("/api/v1/market/trend")
    assert response.status_code == 200
    body = response.json()
    assert body["count_imminent_golden"] >= 0
    assert body["count_imminent_death"] >= 0


def test_trend_detail_imminent_cross_group_has_a_direction_and_sessions_estimate(client: TestClient) -> None:
    response = client.get("/api/v1/market/trend/detail")
    assert response.status_code == 200
    body = response.json()
    assert "imminent_cross" in body
    assert isinstance(body["imminent_cross"], list)
    for row in body["imminent_cross"]:
        assert row["imminent_cross_short_term"]["direction"] in ("golden", "death")
        assert row["imminent_cross_short_term"]["bars_until"] > 0


def test_levels_proximity_matches_are_within_threshold(client: TestClient) -> None:
    response = client.get("/api/v1/market/levels/proximity", params={"threshold": 0.05})
    assert response.status_code == 200
    for match in response.json():
        assert abs(match["level"]["distance_pct"]) <= 0.05


def test_support_resistance_for_known_ticker(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/AAPL/levels")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["price"] > 0
    for level in body["levels"]:
        assert level["kind"] in {"support", "resistance"}


def test_support_resistance_unknown_ticker_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/UNKNOWN/levels")
    assert response.status_code == 404


def test_relationship_map_for_us_ticker_has_both_layers(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/AAPL/relationships")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["region"] == "us"
    assert isinstance(body["statistical"], list)
    assert isinstance(body["sector_peers"], list)
    for relation in body["statistical"]:
        assert relation["ticker"] != "AAPL"
    for peer in body["sector_peers"]:
        assert peer["ticker"] != "AAPL"
    assert "disclosed" not in body
    assert "disclosed_available" not in body


def test_relationship_map_europe_region(client: TestClient) -> None:
    response = client.get("/api/v1/market/tickers/SAP.DE/relationships", params={"region": "europe"})
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "SAP.DE"
    assert body["region"] == "europe"
    assert any(peer["ticker"] != "SAP.DE" for peer in body["sector_peers"])


def test_relationship_map_auto_resolves_region_for_a_ticker_searched_without_one(client: TestClient) -> None:
    # No `region` query param at all - "Analizar activo" style search. Must
    # infer "europe" from the ticker itself (market_universe.region_of), not
    # silently default to "us" and look SAP.DE up in the wrong universe.
    response = client.get("/api/v1/market/tickers/SAP.DE/relationships")
    assert response.status_code == 200
    assert response.json()["region"] == "europe"


def test_relationship_map_unknown_ticker_returns_empty_layers_not_an_error(client: TestClient) -> None:
    # UNKNOWN isn't in the universe or in FakeMarketDataProvider.get_ticker_info -
    # the endpoint should degrade to empty layers, never a 500 or a fabricated result.
    response = client.get("/api/v1/market/tickers/UNKNOWN/relationships")
    assert response.status_code == 200
    body = response.json()
    assert body["statistical"] == []
    assert body["sector_peers"] == []


def test_market_context_has_indices_vix_and_regime(client: TestClient) -> None:
    response = client.get("/api/v1/market/context")
    assert response.status_code == 200
    body = response.json()
    assert len(body["indices"]) == 6
    assert {"level", "sma50", "regime", "term_structure"} <= body["vix"].keys()
    assert body["regime"]["verdict"] in {"favorable", "precaucion", "evitar"}
    assert body["regime"]["headline"]
    assert len(body["regime"]["reasons"]) > 0
    assert isinstance(body["news"], list)
    assert "fear_greed" not in body
    assert "liquidity" not in body
    assert "macro" not in body


def test_market_endpoints_survive_a_missing_computation_cache_table(client: TestClient, engine: Engine) -> None:
    """Same production incident as
    test_portfolios_api.py::test_portfolio_risk_survives_a_missing_computation_cache_table,
    exercised across every endpoint that threads a `db` session into
    `get_universe_snapshot`/durable_cache - a caught exception that skips
    `db.rollback()` doesn't just fail closed on its own cache lookup, it
    poisons every later query in that request too."""
    ComputationCacheORM.__table__.drop(bind=engine)
    try:
        assert client.get("/api/v1/market/screener").status_code == 200
        assert client.get("/api/v1/market/movers").status_code == 200
        assert client.get("/api/v1/market/trend").status_code == 200
        assert client.get("/api/v1/market/trend/detail").status_code == 200
        assert client.get("/api/v1/market/levels/proximity").status_code == 200
        assert client.get("/api/v1/market/context").status_code == 200
    finally:
        ComputationCacheORM.__table__.create(bind=engine)


def test_universe_snapshot_recomputes_when_cached_payload_shape_is_stale(
    client: TestClient, db_session: Session
) -> None:
    db_session.add(
        ComputationCacheORM(
            cache_key="universe_snapshot:us",
            computed_at=datetime.now(UTC),
            payload=[{"ticker": "NOT_ENOUGH_FIELDS"}],  # not a valid TickerSnapshot dict
        )
    )
    db_session.commit()

    response = client.get("/api/v1/market/screener", params={"region": "us"})
    assert response.status_code == 200
    assert len(response.json()) > 0  # recomputed the real ~170-ticker universe, not an empty/broken list


def test_screener_uses_the_dynamic_universe_when_a_snapshot_is_on_file(
    client: TestClient, db_session: Session
) -> None:
    # Tercera auditoría, Bloque F-1: get_universe_snapshot used to always
    # read the curated ~216-ticker dict, no matter what
    # universe_memberships held - the one real consumer of the dynamic,
    # point-in-time universe was an offline script. Seeding a snapshot with
    # tickers that don't exist in the curated dict at all is the only way to
    # prove the live endpoint actually reads it now.
    db_session.add_all(
        [
            UniverseMembershipORM(
                region="us", ticker=f"DYNTICK{i}", sector="Tecnología", as_of_date=date.today(), source="live"
            )
            for i in range(5)
        ]
    )
    db_session.commit()

    response = client.get("/api/v1/market/screener", params={"region": "us"})
    assert response.status_code == 200
    tickers = {row["ticker"] for row in response.json()}
    assert "DYNTICK0" in tickers
    assert not (set(universe_tickers("us")) & tickers)  # the curated list was never consulted
