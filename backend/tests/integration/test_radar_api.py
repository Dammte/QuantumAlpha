"""Reconstruction (2026-09), Fase 5: GET /market/radar - a pure read over
`daily_close.py`'s own precomputed `ticker_daily_states`, never a live
universe scan. See docs/quant_methodology.md §25 and the endpoint's own
docstring."""

from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.trading_params import RISK_PER_TRADE_PCT
from app.domain.models.ticker_daily_state import TickerDailyState
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.services.portfolio_construction_service import MAX_SECTOR_CONCENTRATION_PCT

_VIABLE_GEOMETRY = {
    "entry_price": 50.0,
    "stop_price": 45.0,
    "stop_basis": "bajo el soporte en 45.00",
    "entry_type": "pullback_support",
    "risk_pct": 0.10,
    "risk_atr": 2.5,
    "risk_ceiling_pct": 0.07,
    "target_price": 56.0,
    "target_basis": "objetivo 2:1 sobre el riesgo",
    "reward_pct": 0.12,
    "risk_reward_gross": 2.0,
    "risk_reward_net": 1.9,
    "shares_for_risk_budget": None,
    "position_value": None,
    "pct_of_portfolio": None,
    "viable": True,
    "rejection_reason": None,
}


def _seed_state(db: Session, **overrides) -> TickerDailyState:
    defaults = dict(
        id=None,
        region="us",
        ticker="AAPL",
        trade_date=date.today(),
        computed_at=datetime.now(UTC),
        price=120.0,
        currency="USD",
        trend="uptrend",
        stage="stage2",
        rs_rating=85,
        adx14=28.0,
        atr_multiple=1.2,
        rsi14=55.0,
        gate_passes=True,
        gate_conditions=[
            {"label": "Tendencia alcista o Fase 2 de Weinstein", "passed": True},
            {"label": "Sin extensión parabólica (ATR múltiplo <= 4)", "passed": True},
        ],
        gate_version="2026-09-levels-v1",
        entry_trigger_type="breakout",
        entry_trigger_price=125.0,
        entry_already_triggered=False,
        stop_loss=110.0,
        take_profit=140.0,
        take_profit_method="objetivo 2:1 sobre el riesgo",
        risk_reward=2.0,
    )
    defaults.update(overrides)
    return TickerDailyStateRepository(db).upsert(TickerDailyState(**defaults))


def test_radar_returns_a_passing_gate_ticker(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session)

    response = client.get("/api/v1/market/radar?region=us")

    assert response.status_code == 200
    body = response.json()
    assert body["computed_at"] is not None
    tickers = {item["ticker"] for item in body["items"]}
    assert "AAPL" in tickers
    aapl = next(item for item in body["items"] if item["ticker"] == "AAPL")
    assert aapl["gate_passes"] is True
    assert len(aapl["gate_conditions"]) == 2
    assert aapl["entry_trigger"]["trigger_type"] == "breakout"
    assert aapl["stop_and_target"]["stop_loss"] == 110.0


def test_radar_includes_a_failing_gate_ticker_with_an_active_trigger(
    client: TestClient, db_session: Session
) -> None:
    # "About to trigger" (Parte 0, pregunta 2) is about the trigger, not
    # necessarily a fully-passing gate - a ticker approaching a breakout
    # level still belongs on the radar even if e.g. it's currently
    # overextended.
    _seed_state(
        db_session, ticker="MSFT", gate_passes=False, entry_trigger_type="breakout", entry_trigger_price=410.0
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    tickers = {item["ticker"] for item in body["items"]}
    assert "MSFT" in tickers
    msft = next(item for item in body["items"] if item["ticker"] == "MSFT")
    assert msft["gate_passes"] is False


def test_radar_excludes_a_ticker_with_no_trigger_and_a_failing_gate(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(
        db_session, ticker="XYZ", gate_passes=False, entry_trigger_type=None, entry_trigger_price=None,
        stop_loss=None, take_profit=None, take_profit_method=None, risk_reward=None,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    tickers = {item["ticker"] for item in body["items"]}
    assert "XYZ" not in tickers


def test_radar_computed_at_is_none_when_nothing_has_run_yet(client: TestClient) -> None:
    body = client.get("/api/v1/market/radar?region=us").json()
    assert body["items"] == []
    assert body["computed_at"] is None


def test_radar_is_scoped_by_region(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="AAPL", region="us")
    _seed_state(db_session, ticker="EURO.DE", region="europe")

    us_body = client.get("/api/v1/market/radar?region=us").json()
    europe_body = client.get("/api/v1/market/radar?region=europe").json()

    assert {item["ticker"] for item in us_body["items"]} == {"AAPL"}
    assert {item["ticker"] for item in europe_body["items"]} == {"EURO.DE"}


def test_radar_only_shows_the_latest_row_per_ticker(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="AAPL", trade_date=date(2020, 1, 1), gate_passes=False, entry_trigger_type=None)
    _seed_state(db_session, ticker="AAPL", trade_date=date.today(), gate_passes=True, price=150.0)

    body = client.get("/api/v1/market/radar?region=us").json()

    matches = [item for item in body["items"] if item["ticker"] == "AAPL"]
    assert len(matches) == 1
    assert matches[0]["price"] == 150.0
    assert matches[0]["gate_passes"] is True


def test_radar_row_with_no_entry_geometry_is_none_pre_migration_row(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(db_session)
    aapl = client.get("/api/v1/market/radar?region=us").json()["items"][0]
    assert aapl["entry_geometry"] is None


def test_radar_exposes_the_unsized_entry_geometry_without_a_portfolio(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    geometry = nvda["entry_geometry"]
    assert geometry is not None
    assert geometry["viable"] is True
    assert geometry["entry_type"] == "pullback_support"
    assert geometry["stop_price"] == 45.0
    # Parte 7: never sized without a specific portfolio's capital in view.
    assert geometry["shares_for_risk_budget"] is None
    assert geometry["position_value"] is None


def test_radar_row_with_no_grade_is_none_pre_migration_row(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session)
    aapl = client.get("/api/v1/market/radar?region=us").json()["items"][0]
    assert aapl["grade"] is None


def test_radar_exposes_the_persisted_grade(client: TestClient, db_session: Session) -> None:
    # Parte 5.3 (later pass): `daily_close.py` persists `levels_engine.GradeResult`
    # as a plain dict - the Radar just reads it back, no recomputation.
    _seed_state(
        db_session,
        ticker="NVDA",
        entry_geometry=_VIABLE_GEOMETRY,
        grade={"grade": "A", "reasons": ["Cerca del nivel del disparador", "Semanal alcista"]},
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    # `distance_atr` (Parte 9.1) siempre se serializa aunque la fila
    # persistida sea anterior a ese campo - `None` es su valor por defecto.
    assert nvda["grade"] == {
        "grade": "A", "reasons": ["Cerca del nivel del disparador", "Semanal alcista"], "distance_atr": None,
    }


def test_radar_row_with_no_setups_is_none_pre_migration_row(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session)
    aapl = client.get("/api/v1/market/radar?region=us").json()["items"][0]
    assert aapl["setups"] is None


def test_radar_exposes_the_persisted_setups(client: TestClient, db_session: Session) -> None:
    # Biblioteca de setups del Radar (en curso, quant_methodology.md §28):
    # `daily_close.py` persiste `setups.types.SetupMatch` como dicts planos -
    # el Radar los lee tal cual, sin recomputar nada.
    match = {
        "family": "ma_cross",
        "name": "ma_cross_confirmado",
        "label_es": "Cruce rápido confirmado",
        "stage": "triggered",
        "bars_in_stage": 2,
        "timeframe": "daily",
        "trigger_price": None,
        "trigger_condition": "",
        "invalidation_price": None,
        "invalidation_condition": "",
        "evidence": {"separation_atr": 0.3},
        "narrative_es": "La EMA21 cruzó por encima de la EMA55 hace 2 sesiones.",
        "confidence": "unvalidated",
    }
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY, setups=[match])

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    # `measured_stats`/`ticker_history`/`horizon`/`expected_sessions_to_trigger`
    # siempre se serializan - `None` sin ninguna fila de `setup_performance`/
    # `setup_ticker_history`, o sin haber pasado por `assign_horizon` (este
    # fixture siembra el dict directamente, sin pasar por `daily_close.py`).
    assert nvda["setups"] == [
        {
            **match,
            "horizon": None,
            "expected_sessions_to_trigger": None,
            "measured_stats": None,
            "ticker_history": None,
        }
    ]


def test_radar_attaches_measured_stats_from_setup_performance(client: TestClient, db_session: Session) -> None:
    # Parte 10.2/11.1: la fila SIN segmentar de setup_performance (si
    # scripts/setup_replay_study.py ya corrió para este nombre) se adjunta
    # al setup correspondiente - una sola fila en toda la base de datos,
    # nunca copiada dentro de cada TickerDailyState.
    from app.domain.models.setup_performance import SetupPerformance
    from app.infrastructure.db.repositories.setup_performance_repository import SetupPerformanceRepository

    SetupPerformanceRepository(db_session).replace_all(
        [
            SetupPerformance(
                id=None, setup_name="vcp_3_contracciones", family="vcp", grade=None, market_regime=None,
                n_observations=35, trigger_rate=0.6, win_rate=0.55, expectancy_r=0.42,
                median_bars_held=6.0, mae_p80_pct=-0.03, failure_rate_3d=0.1, confidence="measured",
                computed_at=datetime.now(UTC),
            )
        ]
    )
    _seed_state(db_session, ticker="NVDA", setups=[_setup("ready", name="vcp_3_contracciones")])

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    stats = nvda["setups"][0]["measured_stats"]
    assert stats is not None
    assert stats["n_observations"] == 35
    assert stats["win_rate"] == 0.55
    assert stats["expectancy_r"] == 0.42


def test_radar_attaches_ticker_history_from_setup_ticker_history(client: TestClient, db_session: Session) -> None:
    # Parte 11.2: "este valor ha formado 4 VCP en 5 años; 3 dispararon y 2
    # alcanzaron objetivo" - la fila de setup_ticker_history para ESTE
    # ticker (no la agregada sobre el universo) se adjunta al setup
    # correspondiente.
    from app.domain.models.setup_ticker_history import SetupTickerHistory
    from app.infrastructure.db.repositories.setup_ticker_history_repository import SetupTickerHistoryRepository

    SetupTickerHistoryRepository(db_session).replace_all(
        [
            SetupTickerHistory(
                id=None, ticker="NVDA", region="us", setup_name="vcp_3_contracciones", family="vcp",
                n_observations=4, n_triggered=3, n_target_hit=2,
                first_ready_date=date(2020, 1, 1), last_ready_date=date(2024, 6, 1),
                computed_at=datetime.now(UTC),
            )
        ]
    )
    _seed_state(db_session, ticker="NVDA", setups=[_setup("ready", name="vcp_3_contracciones")])

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    history = nvda["setups"][0]["ticker_history"]
    assert history is not None
    assert history["n_observations"] == 4
    assert history["n_triggered"] == 3
    assert history["n_target_hit"] == 2
    assert history["first_ready_date"] == "2020-01-01"
    assert history["last_ready_date"] == "2024-06-01"


def test_radar_never_attaches_ticker_history_from_a_different_ticker(
    client: TestClient, db_session: Session
) -> None:
    # La misma fila de setup_ticker_history para AAPL no debe "filtrarse" al
    # mismo nombre de setup en NVDA - la clave es (ticker, region, nombre).
    from app.domain.models.setup_ticker_history import SetupTickerHistory
    from app.infrastructure.db.repositories.setup_ticker_history_repository import SetupTickerHistoryRepository

    SetupTickerHistoryRepository(db_session).replace_all(
        [
            SetupTickerHistory(
                id=None, ticker="AAPL", region="us", setup_name="vcp_3_contracciones", family="vcp",
                n_observations=4, n_triggered=3, n_target_hit=2,
                first_ready_date=date(2020, 1, 1), last_ready_date=date(2024, 6, 1),
                computed_at=datetime.now(UTC),
            )
        ]
    )
    _seed_state(db_session, ticker="NVDA", setups=[_setup("ready", name="vcp_3_contracciones")])

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    assert nvda["setups"][0]["ticker_history"] is None


def test_radar_sizes_the_entry_geometry_against_a_portfolios_capital(
    client: TestClient, db_session: Session
) -> None:
    # `cash_balance` only ever reflects deposits/sale proceeds (see
    # `Portfolio.cash_balance`'s own docstring) - a buy alone never reduces
    # it, so `total_portfolio_value` here is simply the market value of what
    # was bought: 100 shares at the fake provider's fixed 150.0 AAPL quote.
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 100, "price": 100},
    )
    capital_total = 100 * 150.0
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    geometry = nvda["entry_geometry"]
    # _VIABLE_GEOMETRY's 10% risk_pct (entry 50, stop 45) keeps the fixed-risk
    # position comfortably under MAX_POSITION_PCT (0.01/0.10 = 10% < 15%), so
    # this exercises the plain `capital * RISK_PER_TRADE_PCT / risk_per_share`
    # formula, not the position-value cap.
    expected_shares = (capital_total * RISK_PER_TRADE_PCT) / (50.0 - 45.0)
    assert geometry["shares_for_risk_budget"] == pytest.approx(expected_shares)
    assert geometry["position_value"] == pytest.approx(expected_shares * 50.0)
    assert geometry["pct_of_portfolio"] == pytest.approx(geometry["position_value"] / capital_total)
    # Stop/target/entry_type themselves are untouched by sizing.
    assert geometry["stop_price"] == 45.0
    assert geometry["entry_type"] == "pullback_support"


def test_radar_with_portfolio_id_leaves_a_non_viable_geometry_unsized(
    client: TestClient, db_session: Session
) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    rejected = dict(_VIABLE_GEOMETRY, viable=False, rejection_reason="riesgo demasiado alto")
    _seed_state(db_session, ticker="NVDA", entry_geometry=rejected)

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    assert nvda["entry_geometry"]["viable"] is False
    assert nvda["entry_geometry"]["shares_for_risk_budget"] is None


def test_radar_with_unknown_portfolio_id_returns_404(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)
    response = client.get("/api/v1/market/radar?region=us&portfolio_id=999")
    assert response.status_code == 404


def _setup(stage: str = "ready", name: str = "x") -> dict:
    return {
        "family": "stage_transition", "name": name, "label_es": "Prueba", "stage": stage, "bars_in_stage": 1,
        "timeframe": "daily", "trigger_price": None, "trigger_condition": "", "invalidation_price": None,
        "invalidation_condition": "", "evidence": {}, "narrative_es": "", "confidence": "unvalidated",
    }


def _grade(value: str, distance_atr: float | None = None) -> dict:
    return {"grade": value, "reasons": [], "distance_atr": distance_atr}


# --- Parte 8/9 (§28.x): ordenación, agrupación por sector y cortes --------


def test_radar_sorts_triggered_before_ready_before_forming(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="FORM", setups=[_setup("forming")], grade=_grade("A"))
    _seed_state(db_session, ticker="TRIG", setups=[_setup("triggered")], grade=_grade("A"))
    _seed_state(db_session, ticker="RDY", setups=[_setup("ready")], grade=_grade("A"))

    body = client.get("/api/v1/market/radar?region=us").json()

    order = [item["ticker"] for item in body["items"]]
    assert order.index("TRIG") < order.index("RDY") < order.index("FORM")


def test_radar_sorts_by_grade_within_the_same_stage(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="GC", setups=[_setup("ready")], grade=_grade("C"))
    _seed_state(db_session, ticker="GA", setups=[_setup("ready")], grade=_grade("A"))
    _seed_state(db_session, ticker="GB", setups=[_setup("ready")], grade=_grade("B"))

    body = client.get("/api/v1/market/radar?region=us").json()

    order = [item["ticker"] for item in body["items"]]
    assert order.index("GA") < order.index("GB") < order.index("GC")


def test_radar_sorts_by_sector_percentile_descending(client: TestClient, db_session: Session) -> None:
    _seed_state(
        db_session, ticker="WEAK", setups=[_setup("ready")], grade=_grade("A"), sector="Salud",
        sector_rs_percentile=22,
    )
    _seed_state(
        db_session, ticker="STRONG", setups=[_setup("ready")], grade=_grade("A"), sector="Tecnología",
        sector_rs_percentile=88,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    order = [item["ticker"] for item in body["items"]]
    assert order.index("STRONG") < order.index("WEAK")


def test_radar_sorts_by_distance_atr_ascending_as_the_final_tiebreak(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(db_session, ticker="FAR", setups=[_setup("ready")], grade=_grade("A", distance_atr=1.8))
    _seed_state(db_session, ticker="NEAR", setups=[_setup("ready")], grade=_grade("A", distance_atr=0.2))

    body = client.get("/api/v1/market/radar?region=us").json()

    order = [item["ticker"] for item in body["items"]]
    assert order.index("NEAR") < order.index("FAR")


def test_radar_drops_a_forming_setup_graded_c(client: TestClient, db_session: Session) -> None:
    # Parte 9.2, literal: "los FORMING de grado C no se muestran".
    _seed_state(db_session, ticker="DROP", setups=[_setup("forming")], grade=_grade("C"))
    _seed_state(db_session, ticker="KEEP_B", setups=[_setup("forming")], grade=_grade("B"))
    _seed_state(db_session, ticker="KEEP_READY_C", setups=[_setup("ready")], grade=_grade("C"))

    body = client.get("/api/v1/market/radar?region=us").json()

    tickers = {item["ticker"] for item in body["items"]}
    assert "DROP" not in tickers
    assert "KEEP_B" in tickers
    assert "KEEP_READY_C" in tickers


def test_radar_caps_candidates_per_sector(client: TestClient, db_session: Session) -> None:
    for i in range(6):
        _seed_state(
            db_session, ticker=f"TEC{i}", setups=[_setup("ready")], grade=_grade("A", distance_atr=float(i)),
            sector="Tecnología", sector_rs_percentile=80,
        )

    body = client.get("/api/v1/market/radar?region=us").json()

    tec_items = [item for item in body["items"] if item["sector"] == "Tecnología"]
    assert len(tec_items) == 4
    # Se queda con los 4 mejores (menor distancia al gatillo), no cualquier 4.
    assert {item["ticker"] for item in tec_items} == {"TEC0", "TEC1", "TEC2", "TEC3"}


def test_radar_caps_the_total_at_radar_max_items(client: TestClient, db_session: Session) -> None:
    sectors = ["Tecnología", "Salud", "Financiero", "Industrial", "Energía", "Consumo discrecional", "Materiales"]
    for i in range(30):
        _seed_state(
            db_session, ticker=f"T{i}", setups=[_setup("ready")], grade=_grade("A", distance_atr=float(i)),
            sector=sectors[i % len(sectors)], sector_rs_percentile=50,
        )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert len(body["items"]) == 25


def test_radar_shows_a_clear_message_when_nothing_qualifies_after_the_cuts(
    client: TestClient, db_session: Session
) -> None:
    # El propio gate ya excluye este candidato (sin trigger, gate en falso)
    # - el job SÍ corrió (computed_at existe) pero nada cumple hoy.
    _seed_state(
        db_session, ticker="XYZ", gate_passes=False, entry_trigger_type=None, entry_trigger_price=None,
        stop_loss=None, take_profit=None, take_profit_method=None, risk_reward=None,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["items"] == []
    assert body["computed_at"] is not None
    assert body["message"] == "Ningún setup cumple los criterios hoy. Es un resultado normal en esta operativa."


def test_radar_computed_at_none_never_shows_the_empty_setups_message(client: TestClient) -> None:
    # Distingue "todavía no hay datos" (computed_at=None) de "hoy no hay
    # nada que cumpla" (Parte 9.2) - no son el mismo mensaje.
    body = client.get("/api/v1/market/radar?region=us").json()
    assert body["computed_at"] is None
    assert body["message"] is None


def test_radar_exposes_sector_and_sector_rs_percentile(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="NVDA", sector="Tecnología", sector_rs_percentile=88)

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    assert nvda["sector"] == "Tecnología"
    assert nvda["sector_rs_percentile"] == 88


def test_radar_a_ticker_matching_several_setups_appears_only_once(
    client: TestClient, db_session: Session
) -> None:
    # Parte 13.2, literal: "un ticker que cumple 4 setups aparece una sola
    # vez" - una fila por ticker con `setups` como lista, nunca una fila
    # por setup que cumple.
    four_setups = [
        _setup("triggered", name="ma_cross_confirmado"),
        _setup("ready", name="vcp_3_contracciones"),
        _setup("ready", name="pullback_a_media_o_soporte"),
        _setup("forming", name="stage1_base_confirmada"),
    ]
    _seed_state(db_session, ticker="NVDA", setups=four_setups, grade=_grade("A"))

    body = client.get("/api/v1/market/radar?region=us").json()

    matches = [item for item in body["items"] if item["ticker"] == "NVDA"]
    assert len(matches) == 1
    assert len(matches[0]["setups"]) == 4


def test_radar_total_analyzed_counts_every_row_before_the_gate_trigger_filter(
    client: TestClient, db_session: Session
) -> None:
    # Parte 12.1, literal: el contador de cabecera («8 de 412 analizados») -
    # cuenta TODO lo que daily_close.py calculó hoy, no solo lo que pasó el
    # filtro de gate/disparador ni lo que sobrevivió a los cortes.
    _seed_state(db_session, ticker="PASS", gate_passes=True)
    _seed_state(
        db_session, ticker="EXCLUDED", gate_passes=False, entry_trigger_type=None, entry_trigger_price=None,
        stop_loss=None, take_profit=None, take_profit_method=None, risk_reward=None,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert len(body["items"]) == 1  # solo PASS sobrevive al filtro de gate/disparador
    assert body["total_analyzed"] == 2  # pero ambos contaron como analizados


def test_radar_total_analyzed_is_zero_when_nothing_has_run_yet(client: TestClient) -> None:
    body = client.get("/api/v1/market/radar?region=us").json()
    assert body["total_analyzed"] == 0


def test_radar_stays_within_its_latency_budget_over_a_realistic_universe(
    client: TestClient, db_session: Session
) -> None:
    # Parte 13.2, literal: "GET /market/radar sigue respondiendo en <
    # 500 ms... es una lectura de tabla; si sube, es que se ha colado
    # cómputo en el endpoint." Presupuesto generoso (5x lo literal), mismo
    # criterio que test_latency_budgets.py: atrapar una regresión real (un
    # cálculo colado en el propio request), no perseguir una cifra de
    # milisegundos concreta en hardware de CI variable.
    import time

    sectors = ["Tecnología", "Salud", "Financiero", "Industrial", "Energía", "Consumo discrecional"]
    for i in range(60):
        _seed_state(
            db_session, ticker=f"T{i}", setups=[_setup("ready")], grade=_grade("A", distance_atr=float(i)),
            sector=sectors[i % len(sectors)], sector_rs_percentile=50,
        )

    start = time.perf_counter()
    response = client.get("/api/v1/market/radar?region=us")
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 2.5  # 5x el presupuesto literal de 500 ms


def test_radar_narrows_a_sized_candidate_by_the_portfolios_sector_concentration(
    client: TestClient, db_session: Session
) -> None:
    # `size_position` alone (the previous test) is blind to every *other*
    # open position - `portfolio_construction_service.apply_portfolio_limits`
    # is the "one layer up" narrowing its own docstring points to. MSFT is
    # "Tecnología", the same curated sector as the NVDA candidate below; TSLA
    # (a different sector) only dilutes total capital, at the fake provider's
    # shared 150.0 quote for both.
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "MSFT", "transaction_type": "buy", "quantity": 25, "price": 100},
    )
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "TSLA", "transaction_type": "buy", "quantity": 75, "price": 100},
    )
    capital_total = 150.0 * (25 + 75)  # = 15,000; MSFT is exactly 25% of it.
    _seed_state(db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY)

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    geometry = nvda["entry_geometry"]
    # size_position alone would size this at capital*RISK_PER_TRADE_PCT/5 = 30
    # shares - but Tecnología is already at 25% of capital, leaving only a 5%
    # sector headroom (30% cap), narrower than the risk-based 30 shares.
    sector_headroom_pct = MAX_SECTOR_CONCENTRATION_PCT - 0.25
    expected_shares = (sector_headroom_pct * capital_total) / 50.0
    assert expected_shares < (capital_total * RISK_PER_TRADE_PCT) / 5.0  # confirms the sector cap is what binds
    assert geometry["shares_for_risk_budget"] == pytest.approx(expected_shares)
    assert geometry["position_value"] == pytest.approx(expected_shares * 50.0)
    assert geometry["viable"] is True
