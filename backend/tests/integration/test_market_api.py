from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.infrastructure.db.models import ComputationCacheORM
from app.services.market_universe import INDUSTRIES, SECTOR_ETFS, universe_tickers


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


def test_sector_performance_covers_every_sector_etf(client: TestClient) -> None:
    response = client.get("/api/v1/market/sectors")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(SECTOR_ETFS)
    assert {row["sector"] for row in body} == set(SECTOR_ETFS.keys())


def test_sector_performance_rs_rank_is_a_percentile_across_all_sectors(client: TestClient) -> None:
    response = client.get("/api/v1/market/sectors")
    body = response.json()
    ranks = [row["rs_rank"] for row in body if row["rs_rank"] is not None]
    assert len(ranks) == len(SECTOR_ETFS)
    assert all(1 <= r <= 99 for r in ranks)
    assert len(set(ranks)) == len(ranks)  # each sector gets a distinct rank, no ties collapsed


def test_sector_forecast_covers_every_sector_with_a_markov_projection(client: TestClient) -> None:
    response = client.get("/api/v1/market/sectors/forecast")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(SECTOR_ETFS)
    assert {row["sector"] for row in body} == set(SECTOR_ETFS.keys())
    for row in body:
        assert isinstance(row["has_statistical_structure"], bool)
        assert 0.0 <= row["prob_bullish_21d"] <= 1.0
        assert isinstance(row["top_stocks"], list)
    # Sectors with genuine statistical structure are surfaced before those without.
    structured_flags = [row["has_statistical_structure"] for row in body]
    assert structured_flags == sorted(structured_flags, reverse=True)


def test_sector_rotation_returns_a_cycle_read(client: TestClient) -> None:
    response = client.get("/api/v1/market/sectors/rotation")
    assert response.status_code == 200
    body = response.json()
    assert len(body["leaders"]) == 3
    assert len(body["laggards"]) == 3
    assert isinstance(body["defensive_leadership"], bool)
    assert 0.0 <= body["cycle_confidence"] <= 1.0


def test_industry_performance_covers_every_industry(client: TestClient) -> None:
    response = client.get("/api/v1/market/industries")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(INDUSTRIES)
    semis = next(row for row in body if row["industry"] == "Semiconductores")
    assert semis["sector"] == "Tecnología"
    assert len(semis["leaders"]) > 0
    assert semis["leaders"][0]["industry"] == "Semiconductores"


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


def test_watchlist_returns_items_with_reasons(client: TestClient) -> None:
    response = client.get("/api/v1/market/watchlist")
    assert response.status_code == 200
    body = response.json()
    assert "computed_at" in body
    for item in body["items"]:
        assert item["horizon"] in {"short", "medium", "long"}
        assert len(item["reasons"]) > 0
        assert item["snapshot"]["ticker"] == item["ticker"]
        assert item["sector_rs_rank"] is None or 1 <= item["sector_rs_rank"] <= 99


def test_watchlist_filters_by_horizon(client: TestClient) -> None:
    response = client.get("/api/v1/market/watchlist", params={"horizon": "short"})
    assert response.status_code == 200
    assert all(item["horizon"] == "short" for item in response.json()["items"])


def test_watchlist_rejects_invalid_horizon(client: TestClient) -> None:
    response = client.get("/api/v1/market/watchlist", params={"horizon": "eternal"})
    assert response.status_code == 422


def test_premium_watchlist_returns_only_approved_tiered_candidates(client: TestClient) -> None:
    response = client.get("/api/v1/market/watchlist/premium")
    assert response.status_code == 200
    body = response.json()
    assert "computed_at" in body
    items = body["items"]
    assert isinstance(items, list)
    assert len(items) <= 30  # 3 tiers x 10 max approved per tier
    for item in items:
        assert item["tier"] in {"daily", "weekly", "monthly"}
        assert len(item["reasons"]) > 0
        # Only genuinely endorsed candidates make the list - see premium_watchlist_service.py
        assert item["signals"]["recommendation"]["verdict"] == "comprar"
        assert {"garch", "markov", "monte_carlo", "backtest", "position_sizing"} <= item["signals"].keys()


def test_premium_watchlist_reports_discard_stats_and_setup_type(client: TestClient) -> None:
    # Segunda auditoría, Bloque 3: MAX_CANDIDATES_PER_TIER used to truncate
    # the cheap pre-filter's matches silently - this is "15 de 47 candidatos
    # analizados" as a real, surfaced number.
    response = client.get("/api/v1/market/watchlist/premium")
    assert response.status_code == 200
    body = response.json()
    assert "discard_stats" in body
    tiers_seen = {s["tier"] for s in body["discard_stats"]}
    assert tiers_seen <= {"daily", "weekly", "monthly"}
    for stats in body["discard_stats"]:
        assert stats["analyzed"] <= stats["prefilter_matches"]
        assert stats["approved"] <= stats["analyzed"]

    for item in body["items"]:
        if item["tier"] == "daily":
            assert item["setup"] in {
                "oversold_bounce", "breakout_volume", "trend_continuation", "pullback_to_support",
            }
        else:
            assert item["setup"] is None


def test_premium_watchlist_filters_by_tier(client: TestClient) -> None:
    response = client.get("/api/v1/market/watchlist/premium", params={"tier": "daily"})
    assert response.status_code == 200
    assert all(item["tier"] == "daily" for item in response.json()["items"])


def test_premium_watchlist_rejects_invalid_tier(client: TestClient) -> None:
    response = client.get("/api/v1/market/watchlist/premium", params={"tier": "yearly"})
    assert response.status_code == 422


def test_premium_watchlist_europe_region_tags_items_with_that_region(client: TestClient) -> None:
    response = client.get("/api/v1/market/watchlist/premium", params={"region": "europe"})
    assert response.status_code == 200
    assert all(item["region"] == "europe" for item in response.json()["items"])


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


def test_market_context_has_indices_vix_fear_greed_and_liquidity(client: TestClient) -> None:
    response = client.get("/api/v1/market/context")
    assert response.status_code == 200
    body = response.json()
    assert len(body["indices"]) == 6
    assert {"level", "sma50", "regime", "term_structure"} <= body["vix"].keys()
    assert 0 <= body["fear_greed"]["score"] <= 100
    assert body["fear_greed"]["label"]
    assert {"proxy_ticker", "trend", "headwind"} <= body["liquidity"].keys()
    assert body["regime"]["verdict"] in {"favorable", "precaucion", "evitar"}
    assert body["regime"]["headline"]
    assert len(body["regime"]["reasons"]) > 0
    assert isinstance(body["news"], list)
    assert "macro" in body  # None without a configured FRED_API_KEY - key must still be present


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
        assert client.get("/api/v1/market/industries").status_code == 200
        assert client.get("/api/v1/market/levels/proximity").status_code == 200
        assert client.get("/api/v1/market/context").status_code == 200

        watchlist = client.get("/api/v1/market/watchlist")
        assert watchlist.status_code == 200
        assert "items" in watchlist.json()

        premium = client.get("/api/v1/market/watchlist/premium", params={"tier": "daily"})
        assert premium.status_code == 200
        assert "items" in premium.json()

        forecast = client.get("/api/v1/market/sectors/forecast")
        assert forecast.status_code == 200
        assert len(forecast.json()) == len(SECTOR_ETFS)
    finally:
        ComputationCacheORM.__table__.create(bind=engine)


def test_premium_watchlist_recomputes_when_cached_payload_shape_is_stale(
    client: TestClient, db_session: Session
) -> None:
    """The real incident this locks in: `imminent_cross` was added to
    CoreSignalsResponse after some premium-watchlist rows were already
    cached, and every read of those rows 500ed on Pydantic validation until
    they naturally expired (up to 30 days later, for the monthly tier). A
    cached payload that's fresh by age but the wrong shape must be treated
    as a miss and recomputed, not crash the endpoint."""
    db_session.add(
        ComputationCacheORM(
            cache_key="premium_watchlist:us:daily",
            computed_at=datetime.now(UTC),  # fresh by age - this must fail on *shape*, not staleness
            payload={"items": []},  # missing `computed_at` - an older response shape
        )
    )
    db_session.commit()

    response = client.get("/api/v1/market/watchlist/premium", params={"tier": "daily"})
    assert response.status_code == 200
    assert "items" in response.json()


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
