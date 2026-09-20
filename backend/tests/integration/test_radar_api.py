"""Reconstruction (2026-09), Fase 5: GET /market/radar - a pure read over
`daily_close.py`'s own precomputed `ticker_daily_states`, never a live
universe scan. See docs/quant_methodology.md §25 and the endpoint's own
docstring."""

from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_llm_narrator
from app.api.v1.endpoints import market
from app.core.trading_params import RISK_PER_TRADE_PCT
from app.domain.interfaces.llm_narrator import LLMNarrator
from app.domain.models.ticker_daily_state import TickerDailyState
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.main import app
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


def test_radar_falls_back_to_live_computation_when_nothing_has_run_yet(client: TestClient) -> None:
    # Auditoria del Radar, bloque B/bloque I, literal: "con
    # ticker_daily_states vacía, el endpoint devuelve o bien resultados del
    # fallback o bien un message explicativo no nulo. Nunca
    # {items: [], message: null}" - este es exactamente el bug reportado.
    # Con el `FakeMarketDataProvider` (determinista, sin red) de por medio,
    # el fallback SÍ produce resultados reales - el caso más honesto de
    # probar, no un mock que finja que "algo" pasó.
    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["computed_at"] is not None
    assert body["source"] == "live_fallback"
    assert body["coverage"]["universe"] > 0
    assert body["coverage"]["analyzed"] > 0
    assert body["coverage"]["analyzed"] <= body["coverage"]["universe"]
    # Nunca la combinación muda que reportó el bug original.
    assert not (body["items"] == [] and body["message"] is None)


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


def _setup(stage: str = "ready", name: str = "x", horizon: str | None = None) -> dict:
    return {
        "family": "stage_transition", "name": name, "label_es": "Prueba", "stage": stage, "bars_in_stage": 1,
        "timeframe": "daily", "trigger_price": None, "trigger_condition": "", "invalidation_price": None,
        "invalidation_condition": "", "evidence": {}, "narrative_es": "narrativa de prueba.",
        "confidence": "unvalidated", "horizon": horizon, "expected_sessions_to_trigger": None,
    }


def _grade(value: str, distance_atr: float | None = None) -> dict:
    return {"grade": value, "reasons": [], "distance_atr": distance_atr}


# --- Parte 8/9 (§28.x): ordenación, agrupación por sector y cortes --------


def test_radar_no_longer_sorts_by_stage_alone_ties_go_alphabetical(
    client: TestClient, db_session: Session
) -> None:
    # Auditoria del Radar, bloque E2: la tupla lexicográfica vieja ordenaba
    # SIEMPRE triggered < ready < forming, aunque el resto de la fila fuera
    # idéntico - el score compuesto lo sustituye. Con todo lo demás igual
    # (sin relative_volume, que es lo único que distinguiría la etapa en el
    # score), las tres puntúan exactamente igual, así que el desempate es
    # alfabético por ticker, no la etapa - un TRIGGERED no adelanta a un
    # FORMING solo por estar disparado si no aporta nada más.
    _seed_state(db_session, ticker="FORM", setups=[_setup("forming")], grade=_grade("A"))
    _seed_state(db_session, ticker="TRIG", setups=[_setup("triggered")], grade=_grade("A"))
    _seed_state(db_session, ticker="RDY", setups=[_setup("ready")], grade=_grade("A"))

    body = client.get("/api/v1/market/radar?region=us").json()

    order = [item["ticker"] for item in body["items"]]
    assert order == ["FORM", "RDY", "TRIG"]


def test_radar_sorts_by_score_descending_when_volume_confirms_a_trigger(
    client: TestClient, db_session: Session
) -> None:
    # Con relative_volume real, un TRIGGERED con volumen fuerte SÍ puntúa
    # más alto que un FORMING sin ese dato - la etapa influye a través del
    # score (componente de confirmación de volumen), no como un rango aparte.
    _seed_state(
        db_session, ticker="QUIET_FORM", setups=[_setup("forming")], grade=_grade("A"), relative_volume=1.0,
    )
    _seed_state(
        db_session, ticker="LOUD_TRIG", setups=[_setup("triggered")], grade=_grade("A"), relative_volume=3.0,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    order = [item["ticker"] for item in body["items"]]
    assert order.index("LOUD_TRIG") < order.index("QUIET_FORM")


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


# --- Auditoria del Radar, bloque E2: score compuesto -----------------------


def test_radar_exposes_a_score_breakdown_for_a_full_candidate(client: TestClient, db_session: Session) -> None:
    _seed_state(
        db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A", distance_atr=0.3),
        setups=[_setup("triggered")], relative_volume=2.0, rs_rating=90, sector_rs_percentile=85,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    assert nvda["score"] is not None
    assert 0.0 <= nvda["score"]["total"] <= 100.0
    # El desglose viaja completo - "sin desglose, esto vuelve a ser una caja
    # negra" (literal, bloque E2).
    assert set(nvda["score"].keys()) == {
        "total", "setup_quality", "relative_strength", "trigger_proximity", "geometry_quality",
        "volume_confirmation", "earnings_penalty", "high_atr_penalty",
    }
    assert nvda["score"]["setup_quality"] > 0  # grado A, aunque unvalidated


def test_radar_score_is_none_for_a_candidate_with_nothing_to_score(
    client: TestClient, db_session: Session
) -> None:
    # Un candidato que entra al Radar solo por gate_passes, sin grado, sin
    # geometría y sin ningún setup de la biblioteca - no hay nada con lo que
    # construir un score real, y `None` es honesto (nunca un cero fabricado
    # indistinguible de un score real bajo).
    _seed_state(db_session, ticker="BARE")

    body = client.get("/api/v1/market/radar?region=us").json()

    bare = next(item for item in body["items"] if item["ticker"] == "BARE")
    assert bare["score"] is None


def test_radar_high_atr_penalty_applies_relative_to_todays_candidates(
    client: TestClient, db_session: Session
) -> None:
    # El percentil 90 se mide sobre los propios candidatos del día - con
    # varios valores tranquilos y uno mucho más volátil, ese último debe
    # llevar la penalización y los demás no. Sectores distintos para que el
    # tope por sector (RADAR_MAX_PER_SECTOR) no interfiera con el score.
    for i in range(9):
        _seed_state(
            db_session, ticker=f"CALM{i}", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("B"),
            setups=[_setup("ready")], atr_pct=0.02, sector=f"Sector{i}",
        )
    _seed_state(
        db_session, ticker="WILD", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("B"),
        setups=[_setup("ready")], atr_pct=0.40, sector="SectorWild",
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    wild = next(item for item in body["items"] if item["ticker"] == "WILD")
    calm = next(item for item in body["items"] if item["ticker"] == "CALM0")
    assert wild["score"]["high_atr_penalty"] < 0
    assert calm["score"]["high_atr_penalty"] == 0


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


# --- Auditoria del Radar, bloque E3/E4: dos listas y "principal a entrar" --


def test_radar_splits_candidates_into_short_and_medium_term_by_horizon(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(
        db_session, ticker="NEAR", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A"),
        setups=[_setup("ready", horizon="short")],
    )
    _seed_state(
        db_session, ticker="FAR", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A"),
        setups=[_setup("forming", horizon="medium")],
    )
    _seed_state(db_session, ticker="NO_HORIZON", grade=_grade("A"))  # sin setup, sin horizonte

    body = client.get("/api/v1/market/radar?region=us").json()

    assert [item["ticker"] for item in body["short_term"]] == ["NEAR"]
    assert [item["ticker"] for item in body["medium_term"]] == ["FAR"]


def test_radar_short_term_list_has_its_own_stricter_sector_cap(client: TestClient, db_session: Session) -> None:
    # RADAR_SHORT_TERM_MAX_PER_SECTOR=3, más estricto que el tope general (4)
    # - "no quiero que un sector caliente me ocupe media lista" (literal).
    for i in range(5):
        _seed_state(
            db_session, ticker=f"TEC{i}", entry_geometry=_VIABLE_GEOMETRY,
            grade=_grade("A", distance_atr=float(i)), setups=[_setup("ready", horizon="short")],
            sector="Tecnología",
        )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert len(body["short_term"]) == 3
    assert {item["ticker"] for item in body["short_term"]} == {"TEC0", "TEC1", "TEC2"}


def test_radar_short_term_shows_fewer_than_ten_with_an_honest_message_never_padded(
    client: TestClient, db_session: Session
) -> None:
    for i in range(3):
        _seed_state(
            db_session, ticker=f"S{i}", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A"),
            setups=[_setup("ready", horizon="short")], sector=f"Sector{i}",
        )
    # Candidatos de sobra en medium_term - nunca deben "prestarse" a short_term
    # para rellenar hasta 10.
    for i in range(10):
        _seed_state(
            db_session, ticker=f"M{i}", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A"),
            setups=[_setup("forming", horizon="medium")], sector=f"SectorM{i}",
        )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert len(body["short_term"]) == 3
    assert body["short_term_message"] == "Solo 3 valores cumplen los criterios de corto plazo hoy."
    assert body["medium_term_message"] is None  # esa sí llega a 10


def test_radar_marks_the_top_short_term_candidate_as_primary_above_the_threshold(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(
        db_session, ticker="STRONG", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A", distance_atr=0.1),
        setups=[_setup("triggered", horizon="short")], rs_rating=95, sector_rs_percentile=90,
        relative_volume=2.5,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    strong = body["short_term"][0]
    assert strong["ticker"] == "STRONG"
    assert strong["is_primary"] is True
    assert strong["thesis"] is not None
    assert "STRONG" in strong["thesis"] or "narrativa de prueba" in strong["thesis"]


def test_radar_primary_thesis_is_overridden_by_gemini_when_configured(
    client: TestClient, db_session: Session
) -> None:
    # Auditoria del Radar, bloque 12: "el LLM redacta; no decide" - Gemini
    # solo reemplaza el TEXTO de la tesis ya puesta por `_mark_primary`
    # (plantilla determinista), nunca decide quién es primario. Con la
    # llamada real a Gemini fuera de alcance en un test (sin clave en el
    # entorno de pruebas - ver test_gemini_degradation.py), se sustituye la
    # dependencia por un `LLMNarrator` falso, mismo patrón que
    # `RadarFallbackService`/`FakeMarketDataProvider` ya usan en conftest.py.
    class _FakeNarrator(LLMNarrator):
        def explain_gate(self, **kwargs):
            raise AssertionError("not exercised by this test")

        def explain_radar_primary(self, **kwargs):
            return "Tesis redactada por Gemini sobre los mismos hechos ya calculados."

    _seed_state(
        db_session, ticker="STRONG", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A", distance_atr=0.1),
        setups=[_setup("triggered", horizon="short")], rs_rating=95, sector_rs_percentile=90,
        relative_volume=2.5,
    )

    app.dependency_overrides[get_llm_narrator] = lambda: _FakeNarrator()
    try:
        body = client.get("/api/v1/market/radar?region=us").json()
    finally:
        del app.dependency_overrides[get_llm_narrator]

    strong = body["short_term"][0]
    assert strong["is_primary"] is True
    assert strong["thesis"] == "Tesis redactada por Gemini sobre los mismos hechos ya calculados."


def test_radar_primary_keeps_the_deterministic_thesis_when_gemini_returns_none(
    client: TestClient, db_session: Session
) -> None:
    class _FailingNarrator(LLMNarrator):
        def explain_gate(self, **kwargs):
            raise AssertionError("not exercised by this test")

        def explain_radar_primary(self, **kwargs):
            return None  # sin configurar, o la llamada falló - ver GeminiNarrator

    _seed_state(
        db_session, ticker="STRONG", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A", distance_atr=0.1),
        setups=[_setup("triggered", horizon="short")], rs_rating=95, sector_rs_percentile=90,
        relative_volume=2.5,
    )

    app.dependency_overrides[get_llm_narrator] = lambda: _FailingNarrator()
    try:
        body = client.get("/api/v1/market/radar?region=us").json()
    finally:
        del app.dependency_overrides[get_llm_narrator]

    strong = body["short_term"][0]
    assert strong["is_primary"] is True
    assert strong["thesis"] is not None
    assert strong["thesis"] != "Tesis redactada por Gemini sobre los mismos hechos ya calculados."


def test_radar_marks_no_primary_when_the_best_score_is_below_the_threshold(
    client: TestClient, db_session: Session
) -> None:
    # Setup sin geometría viable, sin RS, grado C - un score bajo de sobra.
    # `stage="ready"` (no "forming") a propósito: un FORMING de grado C ya
    # se descarta por la Parte 9.2 antes de llegar siquiera a puntuarse -
    # este test quiere el caso "puntúa bajo", no "se descarta antes".
    _seed_state(
        db_session, ticker="WEAK", setups=[_setup("ready", horizon="short")], grade=_grade("C"),
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert len(body["short_term"]) == 1
    assert body["short_term"][0]["is_primary"] is False
    assert body["short_term"][0]["thesis"] is None


def test_radar_medium_term_never_gets_a_primary_even_with_a_high_score(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(
        db_session, ticker="MEDSTRONG", entry_geometry=_VIABLE_GEOMETRY, grade=_grade("A", distance_atr=0.1),
        setups=[_setup("triggered", horizon="medium")], rs_rating=95, sector_rs_percentile=90,
        relative_volume=2.5,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["medium_term"][0]["is_primary"] is False
    assert body["medium_term"][0]["thesis"] is None


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


def test_radar_shows_the_no_close_data_message_when_the_fallback_also_fails(
    client: TestClient, monkeypatch
) -> None:
    # Auditoria del Radar, bloque B.7, caso 1: nunca hubo cierre Y el
    # fallback tampoco produjo nada - distinto de "hoy no hay nada que
    # cumpla" (Parte 9.2, que sí tiene datos frescos detrás). Forzado con
    # monkeypatch porque `FakeMarketDataProvider` por sí solo SIEMPRE
    # produce algo - este es el caso "todo falló", no el camino feliz.
    from app.services.radar_fallback_service import LiveRadarSnapshot, RadarFallbackService

    def _empty_fallback(self, *args, **kwargs):
        return LiveRadarSnapshot(states=[], analyzed=0, universe=0, partial=False, computed_at=datetime.now(UTC))

    monkeypatch.setattr(RadarFallbackService, "get_or_compute", _empty_fallback)

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["items"] == []
    assert body["computed_at"] is None
    assert body["message"] == market.RADAR_MESSAGE_NO_CLOSE_DATA
    assert body["source"] == "daily_close"  # el fallback no llegó a producir nada, no hubo cambio de fuente


def test_radar_shows_the_stale_message_when_old_data_exists_and_the_fallback_also_fails(
    client: TestClient, db_session: Session, monkeypatch
) -> None:
    # Bloque B.7, caso 2: SÍ hubo un cierre, pero es viejo (> 36h), y el
    # fallback tampoco produjo nada - se sigue sirviendo esa foto vieja
    # (mejor un dato real viejo que ninguno), pero el mensaje aclara que no
    # es de hoy - distinto tanto del caso 1 (nunca hubo nada) como del caso
    # 3 (fresco, simplemente nada cumple).
    from app.services.radar_fallback_service import LiveRadarSnapshot, RadarFallbackService

    stale_at = datetime.now(UTC) - timedelta(hours=48)
    _seed_state(
        db_session, ticker="OLD", gate_passes=False, entry_trigger_type=None, entry_trigger_price=None,
        stop_loss=None, take_profit=None, take_profit_method=None, risk_reward=None, computed_at=stale_at,
    )

    def _empty_fallback(self, *args, **kwargs):
        return LiveRadarSnapshot(states=[], analyzed=0, universe=0, partial=False, computed_at=datetime.now(UTC))

    monkeypatch.setattr(RadarFallbackService, "get_or_compute", _empty_fallback)

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["items"] == []
    assert body["source"] == "daily_close"
    assert body["message"] is not None
    assert body["message"] != market.RADAR_EMPTY_MESSAGE
    assert body["message"] != market.RADAR_MESSAGE_NO_CLOSE_DATA


def test_radar_refreshes_via_fallback_when_the_daily_close_data_is_stale(
    client: TestClient, db_session: Session
) -> None:
    # Bloque B: datos viejos (> 36h) SÍ disparan el fallback, y si éste
    # tiene éxito (aquí, con el FakeMarketDataProvider real), la respuesta
    # pasa a `source == "live_fallback"` con un `computed_at` fresco -
    # nunca se queda pegada a la foto vieja si hay una alternativa mejor.
    stale_at = datetime.now(UTC) - timedelta(hours=48)
    _seed_state(db_session, ticker="OLD", computed_at=stale_at)

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["source"] == "live_fallback"
    assert body["computed_at"] is not None
    refreshed_at = datetime.fromisoformat(body["computed_at"])
    assert refreshed_at > stale_at


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


def test_radar_total_analyzed_is_zero_when_nothing_has_run_and_the_fallback_also_fails(
    client: TestClient, monkeypatch
) -> None:
    # Antes del bloque B, "nada ha corrido" implicaba `total_analyzed == 0`
    # sin condiciones. Ahora eso solo es cierto si el fallback TAMBIÉN
    # falla - si tiene éxito, `total_analyzed` refleja lo que sí analizó
    # (ver `test_radar_falls_back_to_live_computation_when_nothing_has_run_yet`).
    from app.services.radar_fallback_service import LiveRadarSnapshot, RadarFallbackService

    def _empty_fallback(self, *args, **kwargs):
        return LiveRadarSnapshot(states=[], analyzed=0, universe=0, partial=False, computed_at=datetime.now(UTC))

    monkeypatch.setattr(RadarFallbackService, "get_or_compute", _empty_fallback)

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


# --- Auditoria del Radar, bloque G/10: régimen, "a punto de disparar", -------
# --- "rompiendo por abajo" ----------------------------------------------------


def _timeframe_strip(weekly_price_vs_ma: str) -> dict:
    return {
        "monthly": {"bias": "unknown", "stage": None, "price_vs_ma": None, "note": ""},
        "weekly": {"bias": "bullish", "stage": "stage2", "price_vs_ma": weekly_price_vs_ma, "note": ""},
        "daily": {"bias": "bullish", "stage": "stage2", "price_vs_ma": "above", "note": ""},
    }


def test_radar_regime_is_bullish_when_the_index_holds_and_breadth_clears_the_bar(
    client: TestClient, db_session: Session, monkeypatch
) -> None:
    monkeypatch.setattr(market, "_index_above_weekly_ma30", lambda market_data, region: True)
    _seed_state(db_session, ticker="A", timeframe_strip=_timeframe_strip("above"))
    _seed_state(db_session, ticker="B", timeframe_strip=_timeframe_strip("above"))

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["regime"]["status"] == "alcista"
    assert body["regime"]["index_above_weekly_ma30"] is True
    assert body["regime"]["breadth_pct"] == pytest.approx(1.0)


def test_radar_regime_is_bearish_when_breadth_falls_even_if_the_index_holds(
    client: TestClient, db_session: Session, monkeypatch
) -> None:
    monkeypatch.setattr(market, "_index_above_weekly_ma30", lambda market_data, region: True)
    _seed_state(db_session, ticker="A", timeframe_strip=_timeframe_strip("below"))
    _seed_state(db_session, ticker="B", timeframe_strip=_timeframe_strip("below"))

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["regime"]["status"] == "bajista"


def test_radar_regime_bearish_halves_a_sized_candidates_position(
    client: TestClient, db_session: Session, monkeypatch
) -> None:
    monkeypatch.setattr(market, "_index_above_weekly_ma30", lambda market_data, region: False)
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "AAPL", "transaction_type": "buy", "quantity": 100, "price": 100},
    )
    capital_total = 100 * 150.0  # mismo patrón que el test de sizing de arriba
    _seed_state(
        db_session, ticker="NVDA", entry_geometry=_VIABLE_GEOMETRY, timeframe_strip=_timeframe_strip("above")
    )

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    assert body["regime"]["status"] == "bajista"
    nvda = next(item for item in body["items"] if item["ticker"] == "NVDA")
    geometry = nvda["entry_geometry"]
    unhalved_shares = (capital_total * RISK_PER_TRADE_PCT) / (50.0 - 45.0)
    assert geometry["viable"] is True
    assert geometry["shares_for_risk_budget"] == pytest.approx(unhalved_shares / 2)
    assert geometry["position_value"] == pytest.approx((unhalved_shares / 2) * 50.0)


def test_radar_regime_unknown_without_enough_data(client: TestClient, db_session: Session, monkeypatch) -> None:
    monkeypatch.setattr(market, "_index_above_weekly_ma30", lambda market_data, region: None)
    _seed_state(db_session, ticker="A")  # sin timeframe_strip

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["regime"]["status"] == "desconocido"


def test_radar_about_to_trigger_lists_a_close_untriggered_candidate_outside_the_horizon_lists(
    client: TestClient, db_session: Session
) -> None:
    # Sin setups (por tanto sin horizon) - nunca puede caer en short_term ni
    # medium_term, así que solo "a punto de disparar" puede mostrarlo.
    _seed_state(
        db_session, ticker="CLOSECO", price=100.0, entry_trigger_price=101.0, entry_already_triggered=False,
        atr_pct=0.05,  # atr14 = 5.0 -> distancia = 1/5 = 0.2 ATR, dentro de 0.3
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    tickers = {item["ticker"] for item in body["about_to_trigger"]}
    assert "CLOSECO" in tickers


def test_radar_about_to_trigger_excludes_a_candidate_too_far_from_its_trigger(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(
        db_session, ticker="FARCO", price=100.0, entry_trigger_price=110.0, entry_already_triggered=False,
        atr_pct=0.05,  # distancia = 10/5 = 2.0 ATR, fuera de 0.3
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert "FARCO" not in {item["ticker"] for item in body["about_to_trigger"]}


def test_radar_about_to_trigger_excludes_one_already_triggered(client: TestClient, db_session: Session) -> None:
    _seed_state(
        db_session, ticker="GONE", price=100.0, entry_trigger_price=101.0, entry_already_triggered=True,
        atr_pct=0.05,
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert "GONE" not in {item["ticker"] for item in body["about_to_trigger"]}


def test_radar_breaking_down_lists_a_ticker_with_a_recent_level_loss(
    client: TestClient, db_session: Session
) -> None:
    _seed_state(
        db_session, ticker="MSFT", sector="Tecnología",
        broken_levels=[{"kind": "ema21", "price": 95.0, "bars_since_loss": 2}],
    )

    body = client.get("/api/v1/market/radar?region=us").json()

    assert len(body["breaking_down"]) == 1
    item = body["breaking_down"][0]
    assert item["ticker"] == "MSFT"
    assert item["broken_levels"] == [{"kind": "ema21", "price": 95.0, "bars_since_loss": 2}]
    assert item["held"] is False


def test_radar_breaking_down_marks_a_ticker_held_in_the_given_portfolio(
    client: TestClient, db_session: Session
) -> None:
    portfolio_id = client.post("/api/v1/portfolios", json={"name": "Main"}).json()["id"]
    client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={"ticker": "MSFT", "transaction_type": "buy", "quantity": 10, "price": 100},
    )
    _seed_state(
        db_session, ticker="MSFT", sector="Tecnología",
        broken_levels=[{"kind": "ema21", "price": 95.0, "bars_since_loss": 2}],
    )

    body = client.get(f"/api/v1/market/radar?region=us&portfolio_id={portfolio_id}").json()

    item = next(i for i in body["breaking_down"] if i["ticker"] == "MSFT")
    assert item["held"] is True


def test_radar_breaking_down_is_empty_when_nothing_broke_down(client: TestClient, db_session: Session) -> None:
    _seed_state(db_session, ticker="AAPL")

    body = client.get("/api/v1/market/radar?region=us").json()

    assert body["breaking_down"] == []
