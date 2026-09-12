from dataclasses import asdict
from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (
    DbSession,
    get_market_context_service,
    get_market_screener_service,
    get_ticker_daily_state_repository,
)
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_snapshot import IndustryPerformance, TickerSnapshot
from app.infrastructure.db.repositories.ticker_daily_state_repository import TickerDailyStateRepository
from app.schemas.market import (
    EntryTriggerResponse,
    GateConditionResponse,
    IndexSnapshotResponse,
    IndustryPerformanceResponse,
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
    SectorPerformanceResponse,
    StatisticalRelationResponse,
    StopAndTargetResponse,
    SupportResistanceResponse,
    TickerSnapshotResponse,
    TrendBreadthResponse,
    TrendDetailResponse,
    UniverseResponse,
    VixSnapshotResponse,
)
from app.services.market_context_service import MarketContextService, assess_market_regime
from app.services.market_screener_service import (
    MarketScreenerService,
    ScreenerFilters,
    apply_filters,
    get_movers,
    get_trend_breadth,
    get_trend_detail,
)
from app.services.market_universe import currency_of, industries_by_sector, region_config, region_of
from app.services.relationship_map_service import build_relationship_map

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


def _industry_to_response(perf: IndustryPerformance) -> IndustryPerformanceResponse:
    data = asdict(perf)
    data["leaders"] = [_to_response(leader) for leader in perf.leaders]
    return IndustryPerformanceResponse(**data)


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


@router.get("/sectors", response_model=list[SectorPerformanceResponse])
def get_sector_performance(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    refresh: bool = False,
) -> list[SectorPerformanceResponse]:
    performance = service.get_sector_performance(region=region, force_refresh=refresh, db=db)
    return [SectorPerformanceResponse(**asdict(p)) for p in performance]


@router.get("/industries", response_model=list[IndustryPerformanceResponse])
def get_industry_performance(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    refresh: bool = False,
) -> list[IndustryPerformanceResponse]:
    performance = service.get_industry_performance(region=region, force_refresh=refresh, db=db)
    return [_industry_to_response(p) for p in performance]


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


def _daily_state_to_radar_item(state: TickerDailyState) -> RadarItemResponse:
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
    )


@router.get("/radar", response_model=RadarResponse)
def get_radar(
    ticker_daily_state_repo: Annotated[TickerDailyStateRepository, Depends(get_ticker_daily_state_repository)],
    region: str = RegionQuery,
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
    states = ticker_daily_state_repo.latest_by_region(region)
    candidates = [s for s in states if s.gate_passes or s.entry_trigger_price is not None]
    computed_at = max((s.computed_at for s in states), default=None)
    return RadarResponse(
        items=[_daily_state_to_radar_item(s) for s in candidates],
        computed_at=computed_at,
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
