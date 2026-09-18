from dataclasses import asdict
from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (
    DbSession,
    get_market_context_service,
    get_market_screener_service,
    get_portfolio_service,
    get_setup_performance_repository,
    get_ticker_daily_state_repository,
    get_trade_plan_repository,
)
from app.domain.models.setup_performance import SetupPerformance
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.infrastructure.db.repositories.setup_performance_repository import SetupPerformanceRepository
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.infrastructure.db.repositories.trade_plan_repository import TradePlanRepository
from app.schemas.market import (
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
    RadarItemResponse,
    RadarResponse,
    RelationshipMapResponse,
    SectorPeerResponse,
    SetupMatchResponse,
    SetupPerformanceStatsResponse,
    StatisticalRelationResponse,
    StopAndTargetResponse,
    SupportResistanceResponse,
    TickerSnapshotResponse,
    TimeframeStripResponse,
    TradeGeometryResponse,
    TrendBreadthResponse,
    TrendDetailResponse,
    UniverseResponse,
    VixSnapshotResponse,
)
from app.services import portfolio_construction_service as pcs
from app.services.market_context_service import MarketContextService, assess_market_regime
from app.services.market_screener_service import (
    MarketScreenerService,
    ScreenerFilters,
    apply_filters,
    get_movers,
    get_trend_breadth,
    get_trend_detail,
)
from app.services.market_universe import currency_of, industries_by_sector, region_config, region_of, sector_of
from app.services.portfolio_service import PortfolioNotFoundError, PortfolioService
from app.services.relationship_map_service import build_relationship_map
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


def _setups_list_to_response(
    data: list[dict] | None, performance_by_name: dict[str, SetupPerformance]
) -> list[SetupMatchResponse] | None:
    if data is None:
        return None
    matches = [SetupMatchResponse(**item) for item in data]
    for match in matches:
        performance = performance_by_name.get(match.name)
        if performance is not None:
            match.measured_stats = _performance_stats_response(performance)
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

_SETUP_STAGE_SORT_RANK = {"triggered": 0, "ready": 1, "forming": 2}
_GRADE_SORT_RANK = {"A": 0, "B": 1, "C": 2}


def _leading_setup(item: RadarItemResponse) -> SetupMatchResponse | None:
    """El primer elemento de `item.setups` ya es el ganador de
    `arbitration.order_by_rank` ("el mejor gana, los demás son contexto") -
    esto solo LEE esa posición, nunca vuelve a elegir entre setups."""
    return item.setups[0] if item.setups else None


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
    """Parte 9.1, lexicográfica - de más a menos significativo: etapa del
    setup líder, grado, expectancy medida, percentil de sector, distancia
    al gatillo en ATR, percentil de fuerza relativa."""
    leading = _leading_setup(item)
    stage_rank = _SETUP_STAGE_SORT_RANK.get(leading.stage, 3) if leading is not None else 3
    grade_value = item.grade.grade if item.grade is not None else None
    grade_rank = _GRADE_SORT_RANK.get(grade_value, 3)
    # Expectancy medida del setup (Parte 10, `setup_replay.py`) - todavía no
    # existe, fase posterior. Sin esa medición, TODO empata en este escalón
    # - es exactamente "los setups sin muestra van al final de su grupo"
    # (el encargo ya lo contempla), no un hueco: el escalón ya está en su
    # sitio correcto en la clave para cuando exista de verdad.
    expectancy_rank = 0
    sector_percentile_rank = -(item.sector_rs_percentile or 0)
    distance_atr = (
        item.grade.distance_atr
        if item.grade is not None and item.grade.distance_atr is not None
        else float("inf")
    )
    rs_percentile_rank = -(item.rs_rating or 0)
    return (stage_rank, grade_rank, expectancy_rank, sector_percentile_rank, distance_atr, rs_percentile_rank)


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


def _rank_and_cut_radar_items(items: list[RadarItemResponse]) -> list[RadarItemResponse]:
    survivors = [item for item in items if not _should_drop_forming_below_grade(item)]
    survivors.sort(key=_radar_sort_key)
    survivors = _apply_sector_cap(survivors, RADAR_MAX_PER_SECTOR)
    return survivors[:RADAR_MAX_ITEMS]


def _daily_state_to_radar_item(
    state: TickerDailyState, performance_by_name: dict[str, SetupPerformance]
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
        setups=_setups_list_to_response(state.setups, performance_by_name),
        timeframe_strip=_timeframe_strip_dict_to_response(state.timeframe_strip),
        sector=state.sector,
        sector_rs_percentile=state.sector_rs_percentile,
    )


@router.get("/radar", response_model=RadarResponse)
def get_radar(
    ticker_daily_state_repo: Annotated[TickerDailyStateRepository, Depends(get_ticker_daily_state_repository)],
    setup_performance_repo: Annotated[SetupPerformanceRepository, Depends(get_setup_performance_repository)],
    portfolio_service: Annotated[PortfolioService, Depends(get_portfolio_service)],
    trade_plan_repo: Annotated[TradePlanRepository, Depends(get_trade_plan_repository)],
    region: str = RegionQuery,
    portfolio_id: int | None = None,
) -> RadarResponse:
    """Reconstruction (2026-09), Fase 5: "qué está a punto de disparar una
    entrada" (Parte 0, pregunta 2) - a pure read over `daily_close.py`'s own
    precomputed `ticker_daily_states`, never a live universe scan. Every
    ticker whose gate passes and/or already has an active entry trigger, as
    of the last nightly run - not the whole universe. `watchlist_service.py`'s
    own cheap-rule filter (2026-09, Fase 5 retirement) was retired outright
    once this endpoint existed to answer the same question with real
    evidence behind it - see docs/quant_methodology.md §25.

    `computed_at` is the *latest* of the returned rows' own timestamps
    (`None` when there's nothing to show yet, e.g. before `daily_close.py`
    has ever run for this region) - a caller-visible way to tell "empty
    because nothing qualifies right now" from "empty because there's no data
    at all"."""
    # Parte 10.2/11.1 (§28.x): la foto completa de setup_performance, leída
    # una sola vez por request - no por fila - y filtrada a la fila sin
    # segmentar de cada nombre (ver `SetupMatchResponse.measured_stats`).
    performance_by_name = {
        row.setup_name: row
        for row in setup_performance_repo.all()
        if row.grade is None and row.market_regime is None
    }

    states = ticker_daily_state_repo.latest_by_region(region)
    candidates = [s for s in states if s.gate_passes or s.entry_trigger_price is not None]
    computed_at = max((s.computed_at for s in states), default=None)
    items = [_daily_state_to_radar_item(s, performance_by_name) for s in candidates]
    # Parte 8/9 (§28.x): descarta/ordena/recorta ANTES de dimensionar contra
    # una cartera concreta más abajo - no tiene sentido gastar ese trabajo
    # en filas que el propio corte va a descartar de todas formas.
    items = _rank_and_cut_radar_items(items)

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

        for item in items:
            if item.entry_geometry is None or not item.entry_geometry.viable:
                continue
            geometry = geometry_from_dict(item.entry_geometry.model_dump())
            sized = size_position(geometry, capital_total)
            narrowed = pcs.apply_portfolio_limits(
                sized, sector_of(item.ticker), sector_concentrations, aggregate_risk, capital_total
            )
            item.entry_geometry = TradeGeometryResponse(**geometry_to_dict(narrowed))

    message = RADAR_EMPTY_MESSAGE if not items and computed_at is not None else None
    return RadarResponse(items=items, computed_at=computed_at, message=message, total_analyzed=len(states))


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
