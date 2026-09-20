import statistics
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (
    DbSession,
    get_llm_narrator,
    get_market_context_service,
    get_market_data_service,
    get_market_screener_service,
    get_portfolio_service,
    get_radar_fallback_service,
    get_setup_performance_repository,
    get_setup_ticker_history_repository,
    get_ticker_daily_state_repository,
    get_trade_plan_repository,
)
from app.core import trading_params as tp
from app.domain.interfaces.llm_narrator import LLMNarrator
from app.domain.models.setup_performance import SetupPerformance
from app.domain.models.setup_ticker_history import SetupTickerHistory
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.infrastructure.db.repositories.setup_performance_repository import SetupPerformanceRepository
from app.infrastructure.db.repositories.setup_ticker_history_repository import SetupTickerHistoryRepository
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.infrastructure.db.repositories.trade_plan_repository import TradePlanRepository
from app.schemas.market import (
    BreakingDownItemResponse,
    BrokenLevelResponse,
    EntryTriggerResponse,
    GateConditionResponse,
    GradeResponse,
    IndexSnapshotResponse,
    IndustryUniverseResponse,
    MarketContextResponse,
    MarketRegimeResponse,
    MoversResponse,
    NewsArticleResponse,
    PriceLevelResponse,
    ProximityItemResponse,
    RadarCoverageResponse,
    RadarItemResponse,
    RadarResponse,
    RadarScoreResponse,
    RelationshipMapResponse,
    SectorPeerResponse,
    SetupMatchResponse,
    SetupPerformanceStatsResponse,
    SetupTickerHistoryResponse,
    StatisticalRelationResponse,
    StopAndTargetResponse,
    SupportResistanceResponse,
    TickerSnapshotResponse,
    TimeframeStripResponse,
    TradeGeometryResponse,
    TrendBreadthResponse,
    TrendDetailResponse,
    TrendRegimeResponse,
    UniverseResponse,
    VixSnapshotResponse,
)
from app.services import market_regime_service as mrs
from app.services import multi_timeframe as mtf
from app.services import portfolio_construction_service as pcs
from app.services import technical_analysis as ta
from app.services.market_context_service import MarketContextService, assess_market_regime
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import (
    MarketScreenerService,
    ScreenerFilters,
    apply_filters,
    get_movers,
    get_trend_breadth,
    get_trend_detail,
)
from app.services.market_universe import (
    benchmark_for_region,
    currency_of,
    industries_by_sector,
    region_config,
    region_of,
    sector_of,
)
from app.services.portfolio_service import PortfolioNotFoundError, PortfolioService
from app.services.radar_fallback_service import RadarFallbackService, is_stale
from app.services.relationship_map_service import build_relationship_map
from app.services.setups import scoring as radar_scoring
from app.services.setups import thesis as radar_thesis
from app.services.trade_geometry import geometry_from_dict, geometry_to_dict, size_position

router = APIRouter(prefix="/market", tags=["market"])

# Reused across every universe-scoped endpoint (screener, movers, sectors,
# industries, trend, radar...) - "us" (default, backward compatible with
# every existing frontend call that doesn't pass it) or "europe". Deliberately
# NOT a single blended universe: see market_universe.py's module docstring.
RegionQuery = Query(default="us", pattern="^(us|europe)$")

VALID_SORT_FIELDS = {
    "change_1d",
    "change_1w",
    "change_1m",
    "change_3m",
    "change_6m",
    "change_1y",
    "price",
    "rsi14",
    "relative_volume",
    "dist_52w_high",
    "dist_52w_low",
    "atr_multiple",
    "adx14",
    "rs_rating",
    "mansfield_rs",
}


def _to_response(snapshot: TickerSnapshot) -> TickerSnapshotResponse:
    data = asdict(snapshot)
    data["trend"] = snapshot.trend.value
    data["stage"] = snapshot.stage.value if snapshot.stage else None
    return TickerSnapshotResponse(**data)


@router.get("/universe", response_model=UniverseResponse)
def get_universe(region: str = RegionQuery) -> UniverseResponse:
    sectors = {
        sector: [i.name for i in industries] for sector, industries in industries_by_sector(region).items()
    }
    industries = [
        IndustryUniverseResponse(name=i.name, sector=i.sector, etf=i.etf, tickers=list(i.tickers))
        for i in region_config(region).industries
    ]
    return UniverseResponse(sectors=sectors, industries=industries)


@router.get("/screener", response_model=list[TickerSnapshotResponse])
def screen_market(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    sector: str | None = None,
    industry: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_change_1d: float | None = None,
    max_change_1d: float | None = None,
    min_rsi: float | None = None,
    max_rsi: float | None = None,
    min_relative_volume: float | None = None,
    min_rs_rating: float | None = None,
    above_sma50: bool | None = None,
    above_sma200: bool | None = None,
    trend: str | None = None,
    stage: str | None = None,
    cap_tier: str | None = None,
    minervini_pass: bool | None = None,
    sort_by: str = "change_1d",
    sort_dir: str = "desc",
    refresh: bool = False,
) -> list[TickerSnapshotResponse]:
    if sort_by not in VALID_SORT_FIELDS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid sort_by: {sort_by}")
    if sort_dir not in {"asc", "desc"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sort_dir must be 'asc' or 'desc'")

    snapshots = service.get_universe_snapshot(region=region, force_refresh=refresh, db=db)
    filters = ScreenerFilters(
        sector=sector,
        industry=industry,
        min_price=min_price,
        max_price=max_price,
        min_change_1d=min_change_1d,
        max_change_1d=max_change_1d,
        min_rsi=min_rsi,
        max_rsi=max_rsi,
        min_relative_volume=min_relative_volume,
        min_rs_rating=min_rs_rating,
        above_sma50=above_sma50,
        above_sma200=above_sma200,
        trend=trend,
        stage=stage,
        cap_tier=cap_tier,
        minervini_pass=minervini_pass,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    filtered = apply_filters(snapshots, filters)
    return [_to_response(s) for s in filtered]


@router.get("/movers", response_model=MoversResponse)
def get_market_movers(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    refresh: bool = False,
) -> MoversResponse:
    snapshots = service.get_universe_snapshot(region=region, force_refresh=refresh, db=db)
    movers = get_movers(snapshots)
    return MoversResponse(**{group: [_to_response(s) for s in items] for group, items in movers.items()})


@router.get("/trend", response_model=TrendBreadthResponse)
def get_market_trend(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    refresh: bool = False,
) -> TrendBreadthResponse:
    snapshots = service.get_universe_snapshot(region=region, force_refresh=refresh, db=db)
    breadth = get_trend_breadth(snapshots)
    return TrendBreadthResponse(**asdict(breadth))


@router.get("/trend/detail", response_model=TrendDetailResponse)
def get_market_trend_detail(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    refresh: bool = False,
) -> TrendDetailResponse:
    snapshots = service.get_universe_snapshot(region=region, force_refresh=refresh, db=db)
    detail = get_trend_detail(snapshots)
    return TrendDetailResponse(**{group: [_to_response(s) for s in items] for group, items in detail.items()})


def _geometry_dict_to_response(data: dict | None) -> TradeGeometryResponse | None:
    return TradeGeometryResponse(**data) if data is not None else None


def _grade_dict_to_response(data: dict | None) -> GradeResponse | None:
    return GradeResponse(**data) if data is not None else None


def _performance_stats_response(performance: SetupPerformance) -> SetupPerformanceStatsResponse:
    return SetupPerformanceStatsResponse(
        n_observations=performance.n_observations,
        trigger_rate=performance.trigger_rate,
        win_rate=performance.win_rate,
        expectancy_r=performance.expectancy_r,
        median_bars_held=performance.median_bars_held,
        mae_p80_pct=performance.mae_p80_pct,
        failure_rate_3d=performance.failure_rate_3d,
    )


def _ticker_history_response(history: SetupTickerHistory) -> SetupTickerHistoryResponse:
    return SetupTickerHistoryResponse(
        n_observations=history.n_observations,
        n_triggered=history.n_triggered,
        n_target_hit=history.n_target_hit,
        first_ready_date=history.first_ready_date,
        last_ready_date=history.last_ready_date,
    )


def _setups_list_to_response(
    data: list[dict] | None,
    performance_by_name: dict[str, SetupPerformance],
    ticker: str,
    region: str,
    history_by_ticker_and_name: dict[tuple[str, str, str], SetupTickerHistory],
) -> list[SetupMatchResponse] | None:
    if data is None:
        return None
    matches = [SetupMatchResponse(**item) for item in data]
    for match in matches:
        performance = performance_by_name.get(match.name)
        if performance is not None:
            match.measured_stats = _performance_stats_response(performance)
        history = history_by_ticker_and_name.get((ticker, region, match.name))
        if history is not None:
            match.ticker_history = _ticker_history_response(history)
    return matches


def _timeframe_strip_dict_to_response(data: dict | None) -> TimeframeStripResponse | None:
    return TimeframeStripResponse(**data) if data is not None else None


# --- Ordenación, agrupación por sector y cortes (Parte 8/9, §28.x) ---------

RADAR_MAX_ITEMS = 25
RADAR_MAX_PER_SECTOR = 4
# Parte 9.2 también pide "RADAR_MIN_REWARD_RISK_NET = 1,5 (ya en
# trade_geometry)" - literal, no hace falta reimplementarlo aquí:
# `trading_params.MIN_RISK_REWARD_NET` (=1,5) ya es el umbral que
# `trade_geometry.py` exige para que `geometry.viable` sea `True` - un
# candidato con peor R:R neto ya se queda sin geometría viable, sin grado y
# sin disparador mucho antes de llegar a este endpoint. Repetirlo aquí
# sería la misma pieza dos veces.
RADAR_DROP_FORMING_BELOW_GRADE = "B"  # los FORMING de grado C no se muestran
RADAR_EMPTY_MESSAGE = "Ningún setup cumple los criterios hoy. Es un resultado normal en esta operativa."
# Auditoria del Radar, bloque B.7: los otros dos motivos honestos de un
# Radar vacío, distintos del anterior (que sí es un resultado normal).
RADAR_MESSAGE_NO_CLOSE_DATA = (
    "Todavía no hay datos de cierre para esta región, y el cálculo en vivo no ha podido "
    "completarse. Vuelve a intentarlo en unos minutos."
)


def _stale_fallback_failed_message(stale_since: datetime) -> str:
    return (
        f"El cierre diario no ha corrido desde el {stale_since:%Y-%m-%d} y el cálculo en vivo "
        "tampoco ha podido completarse esta vez."
    )

# Auditoria del Radar, bloque E3: las dos listas de horizonte, cada una con
# su propio tope - "no quiero que un sector caliente me ocupe media lista"
# (literal, motivo del tope de sector más estricto en corto plazo).
RADAR_LIST_MAX_ITEMS = 10
RADAR_SHORT_TERM_MAX_PER_SECTOR = 3

_GRADE_SORT_RANK = {"A": 0, "B": 1, "C": 2}


def _leading_setup(item: RadarItemResponse) -> SetupMatchResponse | None:
    """El primer elemento de `item.setups` ya es el ganador de
    `arbitration.order_by_rank` ("el mejor gana, los demás son contexto") -
    esto solo LEE esa posición, nunca vuelve a elegir entre setups."""
    return item.setups[0] if item.setups else None


def _atr_pct_p90_in_universe(items: list[RadarItemResponse]) -> float | None:
    """Percentil 90 de `atr_pct` entre los candidatos del día - "de su
    universo" (bloque E2) se interpreta como el propio conjunto de
    candidatos del Radar, no el universo completo (~400-1000 tickers) que
    ni siquiera llega a este endpoint. `None` sin al menos dos valores (un
    percentil sobre 0-1 puntos no es un percentil real)."""
    values = sorted(item.atr_pct for item in items if item.atr_pct is not None)
    if len(values) < 2:
        return None
    return statistics.quantiles(values, n=100)[89]


def _score_input_for_item(
    item: RadarItemResponse, atr_pct_p90: float | None, as_of: date
) -> radar_scoring.RadarScoreInput | None:
    leading = _leading_setup(item)
    geometry = item.entry_geometry
    if leading is None and item.grade is None and geometry is None:
        return None  # nada que puntuar en absoluto - ni setup, ni grado, ni geometría
    return radar_scoring.RadarScoreInput(
        grade=item.grade.grade if item.grade is not None else None,
        setup_confidence=leading.confidence if leading is not None else None,
        rs_rating=item.rs_rating,
        sector_rs_percentile=item.sector_rs_percentile,
        distance_atr=item.grade.distance_atr if item.grade is not None else None,
        risk_reward_net=geometry.risk_reward_net if geometry is not None else None,
        entry_type=geometry.entry_type if geometry is not None else None,
        relative_volume=item.relative_volume,
        setup_stage=leading.stage if leading is not None else None,
        horizon=leading.horizon if leading is not None else None,
        next_earnings_date=item.next_earnings_date,
        as_of=as_of,
        atr_pct=item.atr_pct,
        atr_pct_p90_in_universe=atr_pct_p90,
    )


def _score_response(breakdown: radar_scoring.RadarScoreBreakdown) -> RadarScoreResponse:
    return RadarScoreResponse(
        total=breakdown.total,
        setup_quality=breakdown.setup_quality,
        relative_strength=breakdown.relative_strength,
        trigger_proximity=breakdown.trigger_proximity,
        geometry_quality=breakdown.geometry_quality,
        volume_confirmation=breakdown.volume_confirmation,
        earnings_penalty=breakdown.earnings_penalty,
        high_atr_penalty=breakdown.high_atr_penalty,
    )


def _attach_scores(items: list[RadarItemResponse], as_of: date) -> None:
    """Muta `item.score` en cada elemento - el percentil 90 de ATR se mide
    una sola vez sobre TODO el lote, antes de puntuar el primero, para que
    el percentil de cada item sea consistente con el de los demás (no un
    percentil que se recalcula a medida que la lista "crece")."""
    atr_pct_p90 = _atr_pct_p90_in_universe(items)
    for item in items:
        score_input = _score_input_for_item(item, atr_pct_p90, as_of)
        if score_input is not None:
            item.score = _score_response(radar_scoring.compute_radar_score(score_input))


def _should_drop_forming_below_grade(item: RadarItemResponse) -> bool:
    # Parte 9.2, literal: "los FORMING de grado C no se muestran". Un
    # candidato sin ningún setup de la biblioteca nueva (la mayoría hoy,
    # ver §28.2) no tiene un FORMING que evaluar - la regla es sobre el
    # setup LÍDER, no sobre el ticker en sí, así que nunca se descarta por
    # esta vía.
    leading = _leading_setup(item)
    if leading is None or leading.stage != "forming":
        return False
    grade_value = item.grade.grade if item.grade is not None else None
    if grade_value is None:
        return False
    threshold_rank = _GRADE_SORT_RANK.get(RADAR_DROP_FORMING_BELOW_GRADE, 0)
    return _GRADE_SORT_RANK.get(grade_value, 0) > threshold_rank


def _radar_sort_key(item: RadarItemResponse) -> tuple:
    """Auditoria del Radar, bloque E2: reemplaza la tupla lexicográfica
    original (etapa/grado/`expectancy_rank` hardcodeado a 0/percentil de
    sector/distancia/RS) por el score compuesto explicable - "sustituye la
    tupla por un score numérico 0-100" (literal). `item.score` ya debe
    existir (`_attach_scores` corre antes que esto en
    `_rank_and_cut_radar_items`) - un candidato sin score en absoluto
    (`_score_input_for_item` devolvió `None`, sin setup/grado/geometría que
    puntuar) va al final, no a un cero indistinguible de un score real bajo.
    Empate exacto de score: orden alfabético de ticker, para que el
    resultado sea determinista y reproducible en los tests, no un orden de
    iteración de diccionario arbitrario."""
    score_total = item.score.total if item.score is not None else float("-inf")
    return (-score_total, item.ticker)


def _apply_sector_cap(items: list[RadarItemResponse], max_per_sector: int) -> list[RadarItemResponse]:
    """Aplicado DESPUÉS de ordenar - se queda con los primeros
    `max_per_sector` de cada sector en el orden ya decidido por
    `_radar_sort_key`, nunca una selección aparte."""
    counts: dict[str | None, int] = {}
    kept: list[RadarItemResponse] = []
    for item in items:
        count = counts.get(item.sector, 0)
        if count >= max_per_sector:
            continue
        counts[item.sector] = count + 1
        kept.append(item)
    return kept


def _score_and_sort_radar_items(items: list[RadarItemResponse], as_of: date) -> list[RadarItemResponse]:
    """El primer tramo, compartido por `items` y por las dos listas de
    horizonte (bloque E3): descarta los FORMING de grado bajo, puntúa el
    resto (el percentil 90 de ATR se mide sobre ESTOS supervivientes, no
    sobre el lote bruto - un candidato ya descartado por grado no debe
    arrastrar el percentil de nadie) y ordena por score. Los cortes
    (por sector, por tamaño de lista) son responsabilidad de quien llama
    esto, porque `items`/`short_term`/`medium_term` cada uno tiene los
    suyos propios, distintos entre sí."""
    survivors = [item for item in items if not _should_drop_forming_below_grade(item)]
    _attach_scores(survivors, as_of)
    survivors.sort(key=_radar_sort_key)
    return survivors


def _rank_and_cut_radar_items(scored_sorted_items: list[RadarItemResponse]) -> list[RadarItemResponse]:
    capped = _apply_sector_cap(scored_sorted_items, RADAR_MAX_PER_SECTOR)
    return capped[:RADAR_MAX_ITEMS]


def _build_horizon_list(
    scored_sorted_items: list[RadarItemResponse], horizon: str, max_per_sector: int
) -> tuple[list[RadarItemResponse], str | None]:
    """Auditoria del Radar, bloque E3: filtra por el horizonte del setup
    LÍDER (no de cualquier setup secundario del ticker - el líder ya es "el
    que gana", `arbitration.order_by_rank`), aplica el tope de sector propio
    de esta lista, y corta a `RADAR_LIST_MAX_ITEMS`. Nunca rellena de vuelta
    con peores candidatos si la lista queda corta - "un radar honesto con 3
    nombres vale más que uno con 10 de los cuales 7 son relleno" (literal);
    en ese caso devuelve el mensaje que lo explica."""
    matching = [
        item for item in scored_sorted_items if (leading := _leading_setup(item)) and leading.horizon == horizon
    ]
    capped = _apply_sector_cap(matching, max_per_sector)
    result = capped[:RADAR_LIST_MAX_ITEMS]
    if len(result) >= RADAR_LIST_MAX_ITEMS:
        return result, None
    label = "corto" if horizon == "short" else "medio"
    plural = "es" if len(result) != 1 else ""
    verb_plural = "n" if len(result) != 1 else ""
    message = f"Solo {len(result)} valor{plural} cumple{verb_plural} los criterios de {label} plazo hoy."
    return result, message


def _mark_primary(
    short_term: list[RadarItemResponse], threshold: float = tp.RADAR_PRIMARY_SCORE_THRESHOLD
) -> None:
    """Auditoria del Radar, bloque E4: el primero de `short_term` (ya
    ordenado por score) se marca `is_primary` solo si supera `threshold`
    ("si el mejor candidato del día no llega al umbral, ninguno es primario",
    literal - "un sistema que cada día me señala obligatoriamente un
    principal me empuja a operar por operar"). `threshold` sube a
    `RADAR_PRIMARY_SCORE_THRESHOLD_BEARISH` en régimen bajista (bloque G/10:
    "el contexto tiene que tener consecuencia") - el llamador decide cuál
    pasar, esta función no conoce el régimen. Muta en sitio, igual que
    `_attach_scores`."""
    if not short_term:
        return
    leader = short_term[0]
    if leader.score is None or leader.score.total < threshold:
        return
    leader.is_primary = True
    leading_setup = _leading_setup(leader)
    geometry = leader.entry_geometry
    leader.thesis = radar_thesis.generate_deterministic_thesis(
        ticker=leader.ticker,
        setup_narrative=leading_setup.narrative_es if leading_setup is not None else None,
        sector=leader.sector,
        rs_rating=leader.rs_rating,
        entry_price=geometry.entry_price if geometry is not None else None,
        stop_price=geometry.stop_price if geometry is not None else None,
        stop_basis=geometry.stop_basis if geometry is not None else None,
        target_price=geometry.target_price if geometry is not None else None,
        risk_reward_net=geometry.risk_reward_net if geometry is not None else None,
        score_total=leader.score.total,
    )


def _apply_gemini_radar_thesis(short_term: list[RadarItemResponse], narrator: LLMNarrator) -> None:
    """Auditoria del Radar, bloque E4/12, literal: "esa tesis se genera con
    Gemini... a partir de los números ya calculados - nunca pidiéndole a
    Gemini que analice ni que opine. Si la llamada falla, se muestra un
    resumen plantilla determinista. El LLM redacta; no decide." - `_mark_primary`
    de arriba ya decidió QUIÉN es el primario y ya dejó la tesis determinista
    puesta; esta función es un paso puramente aditivo que intenta
    reemplazarla por la redacción de Gemini sobre EXACTAMENTE los mismos
    hechos, y no hace nada (deja la determinista tal cual) si no hay
    candidato primario, o si `explain_radar_primary` devuelve `None` (sin
    configurar, o cualquier fallo de la llamada - ver `GeminiNarrator`)."""
    if not short_term or not short_term[0].is_primary:
        return
    leader = short_term[0]
    leading_setup = _leading_setup(leader)
    geometry = leader.entry_geometry
    thesis = narrator.explain_radar_primary(
        ticker=leader.ticker,
        setup_narrative=leading_setup.narrative_es if leading_setup is not None else None,
        sector=leader.sector,
        rs_rating=leader.rs_rating,
        entry_price=geometry.entry_price if geometry is not None else None,
        stop_price=geometry.stop_price if geometry is not None else None,
        stop_basis=geometry.stop_basis if geometry is not None else None,
        target_price=geometry.target_price if geometry is not None else None,
        risk_reward_net=geometry.risk_reward_net if geometry is not None else None,
        score_total=leader.score.total,
    )
    if thesis is not None:
        leader.thesis = thesis


def _trigger_distance_atr(item: RadarItemResponse) -> float | None:
    if item.entry_trigger is None or not item.atr_pct or item.price <= 0:
        return None
    atr14 = item.atr_pct * item.price
    if not atr14:
        return None
    return abs(item.price - item.entry_trigger.trigger_price) / atr14


def _build_about_to_trigger(
    scored_sorted_items: list[RadarItemResponse],
    short_term: list[RadarItemResponse],
    medium_term: list[RadarItemResponse],
) -> list[RadarItemResponse]:
    """Auditoria del Radar, bloque G/10: "valores a menos de 0.3 ATR de su
    disparador que no están todavía en las listas" (literal) - "es la lista
    de alarmas para mañana". Nunca uno ya disparado hoy (`already_triggered`
    - ese ya está, o debería estar, en una de las dos listas de horizonte si
    tiene un setup real detrás) ni uno que ya aparece en `short_term`/
    `medium_term`. Ordenado por distancia ascendente (el más inminente
    primero), tope `RADAR_ABOUT_TO_TRIGGER_MAX_ITEMS`."""
    excluded = {item.ticker for item in short_term} | {item.ticker for item in medium_term}
    ranked = []
    for item in scored_sorted_items:
        if item.ticker in excluded or item.entry_trigger is None or item.entry_trigger.already_triggered:
            continue
        distance = _trigger_distance_atr(item)
        if distance is not None and distance < tp.RADAR_ABOUT_TO_TRIGGER_MAX_DISTANCE_ATR:
            ranked.append((distance, item))
    ranked.sort(key=lambda pair: pair[0])
    return [item for _, item in ranked[: tp.RADAR_ABOUT_TO_TRIGGER_MAX_ITEMS]]


def _build_breaking_down(states: list[TickerDailyState], held_tickers: set[str]) -> list[BreakingDownItemResponse]:
    """Auditoria del Radar, bloque G/10: "valores que han perdido un soporte,
    la EMA21 o la EMA55 en las últimas 3 sesiones... marca visualmente los
    que están en mi cartera" (literal) - de TODO lo analizado (`states`), no
    solo lo que pasa el gate: una alarma bajista nunca pasaría el gate de
    compra de todos modos, así que filtrar por `candidates` la dejaría
    siempre vacía. Ordenado por la ruptura más reciente primero, tope
    `RADAR_BREAKING_DOWN_MAX_ITEMS`. `held_tickers` vacío (nunca `None`)
    cuando el request no trae `portfolio_id` - `held` sale `False` para
    todos, honestamente, no "no se sabe"."""
    matches = [s for s in states if s.broken_levels]
    matches.sort(key=lambda s: min(bl["bars_since_loss"] for bl in s.broken_levels))
    return [
        BreakingDownItemResponse(
            ticker=s.ticker,
            sector=s.sector,
            price=s.price,
            currency=s.currency,
            broken_levels=[BrokenLevelResponse(**bl) for bl in s.broken_levels],
            held=s.ticker in held_tickers,
        )
        for s in matches[: tp.RADAR_BREAKING_DOWN_MAX_ITEMS]
    ]


def _index_above_weekly_ma30(market_data: MarketDataService, region: str) -> bool | None:
    """Auditoria del Radar, bloque G/10: "estado del índice de la región...
    por encima/debajo de su MA de 30 semanas" (literal) - UNA sola lectura en
    vivo (el índice de referencia, nunca el universo) por request, la misma
    clase de excepción, acotada de la misma forma, que
    `radar_fallback_service.py` ya documenta para el propio Radar. `None`
    ante cualquier fallo (histórico insuficiente, proveedor caído) - nunca
    hace que el resto del Radar falle, el régimen simplemente queda
    "desconocido" (ver `market_regime_service.assess_trend_regime`)."""
    try:
        index_ticker = benchmark_for_region(region)
        end = date.today()
        start = end - timedelta(days=400)  # de sobra para 30 semanas de barras semanales
        df = market_data.get_bulk_ohlcv([index_ticker], start, end).get(index_ticker)
        if df is None or df.empty:
            return None
        weekly_close = ta.resample_ohlcv(df, mtf.WEEKLY_RULE)["close"]
        ma30 = ta.sma(weekly_close, mtf.WEEKLY_STAGE_MA_WINDOW)
        if ma30.empty or ma30.isna().all():
            return None
        latest_ma30 = ma30.iloc[-1]
        if pd.isna(latest_ma30):
            return None
        return float(weekly_close.iloc[-1]) > float(latest_ma30)
    except Exception:
        return None


def _daily_state_to_radar_item(
    state: TickerDailyState,
    performance_by_name: dict[str, SetupPerformance],
    history_by_ticker_and_name: dict[tuple[str, str, str], SetupTickerHistory],
) -> RadarItemResponse:
    return RadarItemResponse(
        ticker=state.ticker,
        region=state.region,
        trade_date=state.trade_date,
        computed_at=state.computed_at,
        price=state.price,
        currency=state.currency,
        trend=state.trend,
        stage=state.stage,
        rs_rating=state.rs_rating,
        adx14=state.adx14,
        atr_multiple=state.atr_multiple,
        rsi14=state.rsi14,
        gate_passes=state.gate_passes,
        gate_conditions=[GateConditionResponse(**c) for c in state.gate_conditions],
        gate_version=state.gate_version,
        entry_trigger=(
            EntryTriggerResponse(
                trigger_type=state.entry_trigger_type,
                trigger_price=state.entry_trigger_price,
                already_triggered=state.entry_already_triggered,
            )
            if state.entry_trigger_type is not None
            else None
        ),
        stop_and_target=(
            StopAndTargetResponse(
                stop_loss=state.stop_loss,
                take_profit=state.take_profit,
                take_profit_method=state.take_profit_method,
                risk_reward=state.risk_reward,
            )
            if state.stop_loss is not None
            else None
        ),
        entry_geometry=_geometry_dict_to_response(state.entry_geometry),
        grade=_grade_dict_to_response(state.grade),
        setups=_setups_list_to_response(
            state.setups, performance_by_name, state.ticker, state.region, history_by_ticker_and_name
        ),
        timeframe_strip=_timeframe_strip_dict_to_response(state.timeframe_strip),
        sector=state.sector,
        sector_rs_percentile=state.sector_rs_percentile,
        relative_volume=state.relative_volume,
        next_earnings_date=state.next_earnings_date,
        atr_pct=state.atr_pct,
    )


@router.get("/radar", response_model=RadarResponse)
def get_radar(
    ticker_daily_state_repo: Annotated[TickerDailyStateRepository, Depends(get_ticker_daily_state_repository)],
    setup_performance_repo: Annotated[SetupPerformanceRepository, Depends(get_setup_performance_repository)],
    setup_ticker_history_repo: Annotated[
        SetupTickerHistoryRepository, Depends(get_setup_ticker_history_repository)
    ],
    portfolio_service: Annotated[PortfolioService, Depends(get_portfolio_service)],
    trade_plan_repo: Annotated[TradePlanRepository, Depends(get_trade_plan_repository)],
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
    screener: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    radar_fallback: Annotated[RadarFallbackService, Depends(get_radar_fallback_service)],
    narrator: Annotated[LLMNarrator, Depends(get_llm_narrator)],
    region: str = RegionQuery,
    portfolio_id: int | None = None,
) -> RadarResponse:
    """Reconstruction (2026-09), Fase 5: "qué está a punto de disparar una
    entrada" (Parte 0, pregunta 2) - a pure read over `daily_close.py`'s own
    precomputed `ticker_daily_states` en el caso normal. Every ticker whose
    gate passes and/or already has an active entry trigger, as of the last
    nightly run - not the whole universe. `watchlist_service.py`'s own
    cheap-rule filter (2026-09, Fase 5 retirement) was retired outright once
    this endpoint existed to answer the same question with real evidence
    behind it - see docs/quant_methodology.md §25.

    Auditoria del Radar, bloque B: cuando esa lectura está vacía o
    demasiado vieja, cae a `RadarFallbackService` - un cómputo en vivo
    acotado, la única excepción documentada a la regla de "sin cómputo en
    el propio request" (§29.1). `source`/`coverage`/`partial` en la
    respuesta dicen sin ambigüedad de dónde salió el resultado.

    `computed_at` is the *latest* of the returned rows' own timestamps
    (`None` when there's nothing to show yet, e.g. before `daily_close.py`
    has ever run for this region AND el fallback tampoco produjo nada) - a
    caller-visible way to tell "empty because nothing qualifies right now"
    from "empty because there's no data at all"."""
    # Parte 10.2/11.1 (§28.x): la foto completa de setup_performance, leída
    # una sola vez por request - no por fila - y filtrada a la fila sin
    # segmentar de cada nombre (ver `SetupMatchResponse.measured_stats`).
    performance_by_name = {
        row.setup_name: row
        for row in setup_performance_repo.all()
        if row.grade is None and row.market_regime is None
    }
    # Parte 11.2 (§28.x): mismo criterio de lectura - la foto completa de
    # setup_ticker_history, una sola vez por request, indexada por
    # (ticker, nombre de setup).
    history_by_ticker_and_name = {
        (row.ticker, row.region, row.setup_name): row for row in setup_ticker_history_repo.all()
    }

    states = ticker_daily_state_repo.latest_by_region(region)
    original_computed_at = max((s.computed_at for s in states), default=None)
    now = datetime.now(UTC)

    # Auditoria del Radar, bloque B: fallback en vivo cuando no hay datos de
    # cierre o son demasiado viejos - ver `radar_fallback_service.py` para
    # las salvaguardas (tope de tickers, timeout, aislamiento por ticker,
    # caché de 15 min). `source`/`partial`/`coverage_universe` viajan tal
    # cual en la respuesta, sin ambigüedad sobre de dónde salió el dato.
    source = "daily_close"
    partial = False
    coverage_universe = len(states)
    # `fallback_attempted` es DISTINTO de `source != "live_fallback"` tras
    # el bloque de abajo: ambos casos ("nunca hizo falta intentarlo, los
    # datos están frescos" y "se intentó pero no produjo nada") dejan
    # `source == "daily_close"` - sin esta bandera propia, el mensaje de
    # abajo no podría distinguir "datos frescos, nada que puntúe hoy" (un
    # resultado normal) de "esto está viejo y el fallback tampoco funcionó"
    # (un aviso real). Bug encontrado escribiendo el test de este bloque.
    fallback_attempted = not states or is_stale(original_computed_at, now)
    if fallback_attempted:
        fallback_snapshot = radar_fallback.get_or_compute(region, market_data, screener, performance_by_name)
        if fallback_snapshot.states:
            states = fallback_snapshot.states
            source = "live_fallback"
            partial = fallback_snapshot.partial
            coverage_universe = fallback_snapshot.universe

    candidates = [s for s in states if s.gate_passes or s.entry_trigger_price is not None]
    computed_at = max((s.computed_at for s in states), default=None)
    all_items = [
        _daily_state_to_radar_item(s, performance_by_name, history_by_ticker_and_name) for s in candidates
    ]
    # Parte 8/9 (§28.x): descarta/puntúa/ordena ANTES de dimensionar contra
    # una cartera concreta más abajo - no tiene sentido gastar ese trabajo en
    # filas que los cortes de más abajo van a descartar de todas formas.
    # Auditoria del Radar, bloque E3: `scored_sorted` es la base ÚNICA
    # (descartada/puntuada/ordenada una sola vez) de la que `items` y las dos
    # listas de horizonte se derivan cada una con su propio corte - nunca
    # tres cálculos de score independientes.
    scored_sorted = _score_and_sort_radar_items(all_items, date.today())
    items = _rank_and_cut_radar_items(scored_sorted)
    short_term, short_term_message = _build_horizon_list(scored_sorted, "short", RADAR_SHORT_TERM_MAX_PER_SECTOR)
    medium_term, medium_term_message = _build_horizon_list(scored_sorted, "medium", RADAR_MAX_PER_SECTOR)

    # Auditoria del Radar, bloque G/10: cabecera de régimen de mercado - la
    # amplitud sale de `timeframe_strip` ya persistido (`states`/hace ~5
    # sesiones, ambos ya en BD, cero cómputo nuevo); el índice de la región
    # es la ÚNICA lectura en vivo nueva de este bloque, acotada a un ticker.
    # "7 días naturales" es una aproximación deliberada a "5 sesiones" (sin
    # calendario de mercado exacto a este nivel, mismo criterio que
    # `levels_engine.EVENT_RISK_WINDOW_DAYS` ya usa) - si esa fecha exacta no
    # tiene fila (festivo/fin de semana), `breadth_change_5d` sale `None`,
    # nunca un valor aproximado a partir de otra fecha.
    latest_trade_date = max((s.trade_date for s in states), default=None)
    states_5d_ago = (
        ticker_daily_state_repo.for_region_and_date(region, latest_trade_date - timedelta(days=7))
        if latest_trade_date is not None
        else None
    )
    regime = mrs.assess_trend_regime(states, states_5d_ago, _index_above_weekly_ma30(market_data, region))
    primary_threshold = (
        tp.RADAR_PRIMARY_SCORE_THRESHOLD_BEARISH
        if regime.status == mrs.REGIME_BEARISH
        else tp.RADAR_PRIMARY_SCORE_THRESHOLD
    )
    _mark_primary(short_term, primary_threshold)
    # Auditoria del Radar, bloque 12: Gemini redacta, nunca decide - intenta
    # reemplazar la tesis determinista de arriba por la de Gemini sobre los
    # mismos hechos exactos; sin clave configurada o ante cualquier fallo,
    # la determinista ya puesta por `_mark_primary` se queda tal cual.
    _apply_gemini_radar_thesis(short_term, narrator)
    about_to_trigger = _build_about_to_trigger(scored_sorted, short_term, medium_term)

    # Parte 7 (later pass): "the Radar rendering for one portfolio" -
    # `trade_geometry.py`'s own docstring names this as the natural place to
    # call `size_position`, once a *specific* portfolio's capital is known.
    # `compute_entry_geometry` (above, unsized) already did the one thing
    # that needs a fresh read of the ticker's own technicals; sizing it
    # against `capital_total` is a pure, in-memory function - no extra
    # network call per candidate, only the one already-accepted live read
    # `GET /portfolios/{id}/construction` also does for the same portfolio.
    # Optional and additive: omitting `portfolio_id` keeps this endpoint the
    # same zero-network, pure-DB-read path it always was.
    # Auditoria del Radar, bloque G/10: `held_tickers` se calcula ya mismo
    # (no solo dentro del bloque de dimensionado) porque `breaking_down` de
    # abajo también lo necesita, y ese bloque no depende de `portfolio_id`.
    held_tickers: set[str] = set()
    if portfolio_id is not None:
        try:
            portfolio = portfolio_service.get_portfolio_summary(portfolio_id)
        except PortfolioNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        capital_total = portfolio.total_portfolio_value

        # `size_position` alone only enforces this *one* candidate's own
        # fixed-risk/MAX_POSITION_PCT ceilings - blind, by its own docstring,
        # to every other open position. `portfolio_construction_service.
        # apply_portfolio_limits` is the "one layer up" narrowing that
        # function points to: how much of the 6% aggregate-risk budget and
        # the 30%-per-sector ceiling this specific candidate's sector still
        # has room for, given what the portfolio already holds. Weight/risk
        # here are measured against the same `capital_total` used for sizing
        # (not `/construction`'s own `total_market_value` basis) so the two
        # percentages stay comparable - a handful of DB reads (one per held
        # position's trade plan, the same pattern `/construction` already
        # uses), never one per Radar candidate.
        held_positions = [p for p in portfolio.positions if p.quantity > 0]
        held_tickers = {p.ticker for p in held_positions}
        weight_by_ticker = {
            p.ticker: p.market_value_base / capital_total
            for p in held_positions
            if p.market_value_base is not None and capital_total > 0
        }
        sector_by_ticker = {p.ticker: sector_of(p.ticker) for p in held_positions}
        sector_concentrations = pcs.compute_sector_concentration(weight_by_ticker, sector_by_ticker)

        position_risks: list[pcs.HeldPositionRisk] = []
        for p in held_positions:
            plan = trade_plan_repo.get_open(portfolio_id, p.ticker)
            if plan is None or plan.current_stop is None or p.current_price is None:
                continue
            position_risks.append(
                pcs.HeldPositionRisk(
                    ticker=p.ticker, price=p.current_price, stop=plan.current_stop, quantity=p.quantity
                )
            )
        aggregate_risk = pcs.compute_aggregate_risk(position_risks, capital_total)

        # `scored_sorted` (no `items`) - el superconjunto del que `items`,
        # `short_term` y `medium_term` derivan cada uno su propio subcorte;
        # dimensionar aquí, una sola vez por objeto, deja ya dimensionados
        # los tres, sin volver a procesar el mismo `entry_geometry` dos
        # veces (que aplicaría el techo de riesgo agregado por partida
        # doble sobre el mismo candidato).
        for item in scored_sorted:
            if item.entry_geometry is None or not item.entry_geometry.viable:
                continue
            geometry = geometry_from_dict(item.entry_geometry.model_dump())
            sized = size_position(geometry, capital_total)
            narrowed = pcs.apply_portfolio_limits(
                sized, sector_of(item.ticker), sector_concentrations, aggregate_risk, capital_total
            )
            # Auditoria del Radar, bloque G/10: "si el régimen es bajista...
            # el tamaño de posición sugerido se reduce a la mitad" (literal) -
            # se aplica DESPUÉS de todos los demás límites (nunca los sustituye,
            # solo los estrecha más), y vuelve a comprobar el mínimo viable -
            # una posición que ya cabía justo podría dejar de tener sentido
            # una vez reducida a la mitad.
            if regime.status == mrs.REGIME_BEARISH and narrowed.viable and narrowed.shares_for_risk_budget:
                halved_value = narrowed.position_value / 2 if narrowed.position_value is not None else None
                halved_pct = narrowed.pct_of_portfolio / 2 if narrowed.pct_of_portfolio is not None else None
                narrowed = replace(
                    narrowed,
                    shares_for_risk_budget=narrowed.shares_for_risk_budget / 2,
                    position_value=halved_value,
                    pct_of_portfolio=halved_pct,
                )
                if halved_value is not None and halved_value < tp.MIN_POSITION_USD:
                    reason = "posición demasiado pequeña una vez reducida a la mitad por régimen bajista"
                    narrowed = replace(narrowed, viable=False, rejection_reason=reason)
            item.entry_geometry = TradeGeometryResponse(**geometry_to_dict(narrowed))

    breaking_down = _build_breaking_down(states, held_tickers)

    # Auditoria del Radar, bloque B.7: "el estado vacío deja de ser mudo" -
    # tres motivos distintos para un `items` vacío, nunca `message: None` en
    # silencio cuando de verdad no hay nada que mostrar:
    #   1. Nunca hubo cierre Y el fallback tampoco produjo nada.
    #   2. Hubo cierre, pero está viejo, Y el fallback tampoco produjo nada
    #      (se sigue sirviendo esa foto vieja - mejor un dato viejo real que
    #      ninguno - pero el mensaje deja claro que no es de hoy).
    #   3. Hay datos frescos (de `daily_close.py` o del fallback) y
    #      sencillamente ningún candidato pasa el filtro hoy - un resultado
    #      legítimo, el único caso que usa `RADAR_EMPTY_MESSAGE`.
    message = None
    if not items:
        fallback_failed = fallback_attempted and source != "live_fallback"
        if original_computed_at is None and fallback_failed:
            message = RADAR_MESSAGE_NO_CLOSE_DATA
        elif original_computed_at is not None and fallback_failed:
            message = _stale_fallback_failed_message(original_computed_at)
        else:
            message = RADAR_EMPTY_MESSAGE
    return RadarResponse(
        items=items,
        computed_at=computed_at,
        message=message,
        total_analyzed=len(states),
        short_term=short_term,
        medium_term=medium_term,
        short_term_message=short_term_message,
        medium_term_message=medium_term_message,
        source=source,
        coverage=RadarCoverageResponse(analyzed=len(states), universe=coverage_universe),
        partial=partial,
        regime=TrendRegimeResponse(
            status=regime.status,
            index_above_weekly_ma30=regime.index_above_weekly_ma30,
            breadth_pct=regime.breadth_pct,
            breadth_change_5d=regime.breadth_change_5d,
            headline=regime.headline,
        ),
        about_to_trigger=about_to_trigger,
        breaking_down=breaking_down,
    )


@router.get("/levels/proximity", response_model=list[ProximityItemResponse])
def get_levels_proximity(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    threshold: float = 0.03,
    refresh: bool = False,
) -> list[ProximityItemResponse]:
    matches = service.get_proximity_matches(region=region, threshold=threshold, force_refresh=refresh, db=db)
    return [
        ProximityItemResponse(
            ticker=m["ticker"],
            sector=m["sector"],
            currency=m["currency"],
            price=m["price"],
            level=PriceLevelResponse(**asdict(m["level"])),
        )
        for m in matches
    ]


@router.get("/tickers/{ticker}/levels", response_model=SupportResistanceResponse)
def get_support_resistance(
    ticker: str,
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    start: date | None = Query(default=None, description="Defaults to one year before `end`"),
    end: date | None = Query(default=None, description="Defaults to today"),
) -> SupportResistanceResponse:
    end = end or date.today()
    start = start or (end - timedelta(days=365))
    try:
        price, levels = service.get_support_resistance(ticker.upper(), start, end)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return SupportResistanceResponse(
        ticker=ticker.upper(),
        currency=currency_of(ticker.upper()),
        price=price,
        levels=[PriceLevelResponse(**asdict(lv)) for lv in levels],
    )


@router.get("/tickers/{ticker}/relationships", response_model=RelationshipMapResponse)
def get_relationship_map(
    ticker: str,
    screener: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str | None = Query(default=None, pattern="^(us|europe)$"),
) -> RelationshipMapResponse:
    """Tercera auditoría, Bloque G: "que se muestren acciones relacionadas...
    un mapa completo del proceso para buscar nuevas opciones" - dos capas de
    fiabilidad decreciente (estadística, sector/industria), ver
    `relationship_map_service.py`. 2026-09: la tercera capa (menciones en
    documentos SEC EDGAR) se retiró - ver el módulo.

    Unlike every other market/ endpoint, `region` has no hardcoded default
    here: this is reached from a free-text "Analizar activo" search (same as
    `GET /tickers/{ticker}/analysis`), which doesn't necessarily know which
    universe the ticker belongs to. When the caller omits it, fall back to
    `market_universe.region_of` - the same best-effort guess already used for
    exactly this "ticker typed directly, region unknown" case elsewhere
    (`technical_analysis.closed_bars`, `benchmark_for_ticker`)."""
    resolved_region = region or region_of(ticker.upper())
    result = build_relationship_map(ticker.upper(), resolved_region, screener, db=db)
    return RelationshipMapResponse(
        ticker=result.ticker,
        region=result.region,
        statistical=[StatisticalRelationResponse(**asdict(r)) for r in result.statistical],
        sector_peers=[SectorPeerResponse(**asdict(p)) for p in result.sector_peers],
        computed_at=result.computed_at,
    )


@router.get("/context", response_model=MarketContextResponse)
def get_market_context(
    context_service: Annotated[MarketContextService, Depends(get_market_context_service)],
) -> MarketContextResponse:
    """2026-09: Fear & Greed, dollar liquidity and the FRED macro read were
    retired from this panel (see market_context_service.py's module
    docstring) - VIX is what's left driving `regime`."""
    indices = context_service.get_indices()
    vix = context_service.get_vix()
    regime = assess_market_regime(vix)
    news = context_service.get_market_news()

    return MarketContextResponse(
        indices=[IndexSnapshotResponse(**asdict(i)) for i in indices],
        vix=VixSnapshotResponse(**asdict(vix)),
        regime=MarketRegimeResponse(**asdict(regime)),
        news=[NewsArticleResponse(**asdict(n)) for n in news],
    )
