from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (
    DbSession,
    get_macro_data_service,
    get_market_context_service,
    get_market_screener_service,
    get_premium_watchlist_service,
)
from app.domain.models.ticker_snapshot import IndustryPerformance, TickerSnapshot
from app.schemas.market import (
    FearGreedResponse,
    IndexSnapshotResponse,
    IndustryPerformanceResponse,
    IndustryUniverseResponse,
    LiquidityResponse,
    MacroSnapshotResponse,
    MarketContextResponse,
    MarketRegimeResponse,
    MoversResponse,
    NewsArticleResponse,
    PremiumWatchlistItemResponse,
    PremiumWatchlistResponse,
    PriceLevelResponse,
    ProximityItemResponse,
    SectorForecastResponse,
    SectorPerformanceResponse,
    SectorRotationResponse,
    SupportResistanceResponse,
    TickerSnapshotResponse,
    TrendBreadthResponse,
    TrendDetailResponse,
    UniverseResponse,
    VixSnapshotResponse,
    WatchlistItemResponse,
    WatchlistResponse,
)
from app.schemas.quant_analysis import CoreSignalsResponse
from app.services import durable_cache
from app.services.macro_data_service import MacroDataService
from app.services.market_context_service import MarketContextService, assess_market_regime
from app.services.market_screener_service import (
    MarketScreenerService,
    ScreenerFilters,
    apply_filters,
    get_movers,
    get_trend_breadth,
    get_trend_detail,
)
from app.services.market_universe import currency_of, industries_by_sector, region_config
from app.services.premium_watchlist_service import (
    DAILY,
    MONTHLY,
    WEEKLY,
    PremiumWatchlistItem,
    PremiumWatchlistService,
)
from app.services.sector_rotation_service import assess_sector_rotation
from app.services.ticker_analysis_service import CoreTickerSignals
from app.services.watchlist_service import build_watchlist

router = APIRouter(prefix="/market", tags=["market"])

# Reused across every universe-scoped endpoint (screener, movers, sectors,
# industries, trend, watchlist...) - "us" (default, backward compatible with
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


def _core_signals_to_response(signals: CoreTickerSignals) -> CoreSignalsResponse:
    data = asdict(signals)
    data["trend"] = signals.trend.value
    data["stage"] = signals.stage.value if signals.stage else None
    data["market_trend"] = signals.market_trend.value if signals.market_trend else None
    return CoreSignalsResponse(**data)


def _premium_item_to_response(item: PremiumWatchlistItem) -> PremiumWatchlistItemResponse:
    return PremiumWatchlistItemResponse(
        ticker=item.ticker,
        sector=item.sector,
        industry=item.industry,
        cap_tier=item.cap_tier,
        currency=item.currency,
        region=item.region,
        tier=item.tier,
        reasons=item.reasons,
        premium_score=item.premium_score,
        signals=_core_signals_to_response(item.signals),
    )


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
    region: str = RegionQuery,
    refresh: bool = False,
) -> list[SectorPerformanceResponse]:
    performance = service.get_sector_performance(region=region, force_refresh=refresh)
    return [SectorPerformanceResponse(**asdict(p)) for p in performance]


@router.get("/sectors/forecast", response_model=list[SectorForecastResponse])
def get_sector_forecast(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    refresh: bool = False,
) -> list[SectorForecastResponse]:
    """Which sectors are statistically likely to lead over the next 5-21
    trading days, not just which already have (see `/sectors` for that) - a
    Markov chain fit per sector ETF, see `MarketScreenerService.get_sector_forecast`."""
    forecast = service.get_sector_forecast(region=region, force_refresh=refresh, db=db)
    return [SectorForecastResponse(**asdict(f)) for f in forecast]


@router.get("/sectors/rotation", response_model=SectorRotationResponse | None)
def get_sector_rotation(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    region: str = RegionQuery,
    refresh: bool = False,
) -> SectorRotationResponse | None:
    """Which sectors are actually leading right now, cross-referenced against
    the classic business-cycle rotation pattern - see `sector_rotation_service.py`.
    The cycle-phase model itself was built around the US business cycle; applied
    to Europe it's a reasonable, disclosed approximation (the two cycles are
    closely correlated), not a separately-researched European model."""
    performance = service.get_sector_performance(region=region, force_refresh=refresh)
    rotation = assess_sector_rotation(performance)
    return SectorRotationResponse(**asdict(rotation)) if rotation is not None else None


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


@router.get("/watchlist", response_model=WatchlistResponse)
def get_watchlist(
    service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    db: DbSession,
    region: str = RegionQuery,
    horizon: str | None = Query(default=None, pattern="^(short|medium|long)$"),
    refresh: bool = False,
) -> WatchlistResponse:
    """"Acciones a revisar": every ticker in the universe whose technicals match
    a cheap rule (see `watchlist_service.py`) - just a filter over the same
    universe snapshot `/screener` and `/movers` already share, so it's as fresh
    as that snapshot (see `computed_at`) and never independently recomputed."""
    snapshots = service.get_universe_snapshot(region=region, force_refresh=refresh, db=db)
    items = build_watchlist(snapshots, horizon=horizon)
    sector_rank_by_name = {
        s.sector: s.rs_rank for s in service.get_sector_performance(region=region, force_refresh=refresh)
    }
    computed_at = service.get_snapshot_computed_at(region) or datetime.now(UTC)
    return WatchlistResponse(
        items=[
            WatchlistItemResponse(
                ticker=item.ticker,
                sector=item.sector,
                industry=item.industry,
                cap_tier=item.cap_tier,
                horizon=item.horizon,
                reasons=item.reasons,
                snapshot=_to_response(item.snapshot),
                sector_rs_rank=sector_rank_by_name.get(item.sector),
            )
            for item in items
        ],
        computed_at=computed_at,
    )


# Matches PremiumWatchlistService.CACHE_TTL - the durable cache should never
# consider a tier "fresh" for longer than the in-process one already would.
_PREMIUM_DURABLE_TTL = {DAILY: timedelta(days=1), WEEKLY: timedelta(days=7), MONTHLY: timedelta(days=30)}


@router.get("/watchlist/premium", response_model=PremiumWatchlistResponse)
def get_premium_watchlist(
    service: Annotated[PremiumWatchlistService, Depends(get_premium_watchlist_service)],
    db: DbSession,
    region: str = RegionQuery,
    tier: str | None = Query(default=None, pattern="^(daily|weekly|monthly)$"),
    refresh: bool = False,
) -> PremiumWatchlistResponse:
    """A small, curated list (~10 per tier) of tickers that passed the *exact
    same* full analysis "Analizar activo" runs - not just a cheap rule match.
    See `premium_watchlist_service.py` for the approval bar and the per-tier
    refresh cadence (daily/weekly/monthly, matching each tier's own name).

    Backed by the durable cache in addition to the service's own per-tier
    in-process one, for the same restart-durability reason as portfolio risk
    (see `durable_cache.py`) - this is the single most expensive read in the
    app (the full quant suite over up to 15 candidates x however many tiers
    are requested), so it's the one that benefits the most from never being
    forced to recompute just because the process restarted."""
    tiers = [tier] if tier else [DAILY, WEEKLY, MONTHLY]
    cache_key = f"premium_watchlist:{region}:{tier or 'all'}"
    durable_ttl = min(_PREMIUM_DURABLE_TTL[t] for t in tiers)
    if not refresh:
        cached = durable_cache.load_fresh_as(db, cache_key, durable_ttl, PremiumWatchlistResponse.model_validate)
        if cached is not None:
            return cached

    items = service.get_premium_watchlist(region=region, tier=tier, force_refresh=refresh)
    computed_at = datetime.now(UTC)
    response = PremiumWatchlistResponse(
        items=[_premium_item_to_response(i) for i in items], computed_at=computed_at
    )
    durable_cache.save(db, cache_key, response.model_dump(mode="json"), computed_at=computed_at)
    return response


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


@router.get("/context", response_model=MarketContextResponse)
def get_market_context(
    context_service: Annotated[MarketContextService, Depends(get_market_context_service)],
    screener_service: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    macro_service: Annotated[MacroDataService, Depends(get_macro_data_service)],
    db: DbSession,
) -> MarketContextResponse:
    universe_snapshot = screener_service.get_universe_snapshot(db=db)
    indices = context_service.get_indices()
    vix = context_service.get_vix()
    fear_greed = context_service.get_fear_greed(universe_snapshot)
    liquidity = context_service.get_liquidity()
    macro = macro_service.get_macro_snapshot()
    regime = assess_market_regime(vix, fear_greed, liquidity, macro)
    news = context_service.get_market_news()

    return MarketContextResponse(
        indices=[IndexSnapshotResponse(**asdict(i)) for i in indices],
        vix=VixSnapshotResponse(**asdict(vix)),
        fear_greed=FearGreedResponse(
            score=fear_greed.score, label=fear_greed.label, components=fear_greed.components
        ),
        liquidity=LiquidityResponse(**asdict(liquidity)),
        regime=MarketRegimeResponse(**asdict(regime)),
        news=[NewsArticleResponse(**asdict(n)) for n in news],
        macro=MacroSnapshotResponse(**asdict(macro)) if macro is not None else None,
    )
