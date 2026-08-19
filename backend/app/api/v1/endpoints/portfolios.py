from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import (
    DbSession,
    get_asset_repository,
    get_market_data_service,
    get_market_screener_service,
    get_portfolio_repository,
    get_portfolio_risk_service,
    get_portfolio_service,
    get_position_signal_snapshot_repository,
    get_premium_watchlist_service,
    get_trade_plan_repository,
)
from app.domain.models.asset import AssetClass
from app.domain.models.trade_plan import TradePlan
from app.infrastructure.db.repositories.asset_repository import AssetRepository
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.infrastructure.db.repositories.position_signal_snapshot_repository import (
    PositionSignalSnapshotRepository,
)
from app.infrastructure.db.repositories.trade_plan_repository import TradePlanRepository
from app.schemas.market import (
    PortfolioRiskResponse,
    PositionRiskResponse,
    PriceLevelResponse,
    ScaledExitPlanResponse,
    SwapSuggestionResponse,
    TradePlanResponse,
)
from app.schemas.portfolio import PortfolioCreate, PortfolioRead, PortfolioSummary, PositionRead
from app.schemas.quant_analysis import CoreSignalsResponse, MultiTimeframeResponse, TimeframeReadResponse
from app.schemas.transaction import TransactionCreate, TransactionRead
from app.services import durable_cache
from app.services import multi_timeframe as mtf
from app.services import trade_manager as tm
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.opportunity_cost import find_swap_suggestions
from app.services.portfolio_risk_service import PortfolioRiskService
from app.services.portfolio_service import PortfolioNotFoundError, PortfolioService
from app.services.premium_watchlist_service import PremiumWatchlistService
from app.services.ticker_analysis_service import CoreTickerSignals

# Several hours, not minutes - see durable_cache.py's docstring. Portfolio risk
# runs on daily bars, so a handful of refreshes a day is already as fresh as
# the underlying signals actually get; "actualizar" (the `refresh=true` query
# param) always forces a live recompute regardless of this window.
PORTFOLIO_RISK_DURABLE_TTL = timedelta(hours=6)

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


def _core_signals_to_response(signals: CoreTickerSignals) -> CoreSignalsResponse:
    data = asdict(signals)
    data["trend"] = signals.trend.value
    data["stage"] = signals.stage.value if signals.stage else None
    data["market_trend"] = signals.market_trend.value if signals.market_trend else None
    return CoreSignalsResponse(**data)


def _timeframe_read_to_response(read: mtf.TimeframeRead) -> TimeframeReadResponse:
    data = asdict(read)
    data["trend"] = read.trend.value
    data["stage"] = read.stage.value if read.stage else None
    return TimeframeReadResponse(**data)


def _multi_timeframe_to_response(read: mtf.MultiTimeframeRead) -> MultiTimeframeResponse:
    return MultiTimeframeResponse(
        weekly=_timeframe_read_to_response(read.weekly) if read.weekly else None,
        daily=_timeframe_read_to_response(read.daily),
        intraday=_timeframe_read_to_response(read.intraday) if read.intraday else None,
        alignment=read.alignment,
        alignment_score=read.alignment_score,
        conflicts=read.conflicts,
    )


def _scaled_exit_to_response(plan: tm.ScaledExitPlan) -> ScaledExitPlanResponse:
    return ScaledExitPlanResponse(
        action=plan.action.value,
        shares_to_sell=plan.shares_to_sell,
        shares_remaining_after=plan.shares_remaining_after,
        suggested_new_stop=plan.suggested_new_stop,
        description=plan.description,
    )


def _trade_plan_to_response(plan: TradePlan) -> TradePlanResponse:
    return TradePlanResponse(
        entry_price=plan.entry_price,
        entry_date=plan.entry_date,
        initial_stop=plan.initial_stop,
        initial_target=plan.initial_target,
        current_stop=plan.current_stop,
        highest_close_since_entry=plan.highest_close_since_entry,
        thesis=plan.thesis,
        engine_version=plan.engine_version,
    )


@router.post("", response_model=PortfolioRead, status_code=status.HTTP_201_CREATED)
def create_portfolio(
    payload: PortfolioCreate,
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
) -> PortfolioRead:
    orm = repository.create(name=payload.name, base_currency=payload.base_currency)
    return PortfolioRead.model_validate(orm)


@router.get("", response_model=list[PortfolioRead])
def list_portfolios(
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
) -> list[PortfolioRead]:
    return [PortfolioRead.model_validate(orm) for orm in repository.list_all()]


@router.get("/{portfolio_id}", response_model=PortfolioRead)
def get_portfolio(
    portfolio_id: int,
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
) -> PortfolioRead:
    orm = repository.get(portfolio_id)
    if orm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return PortfolioRead.model_validate(orm)


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio(
    portfolio_id: int,
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
) -> None:
    repository.delete(portfolio_id)


@router.get("/{portfolio_id}/summary", response_model=PortfolioSummary)
def get_portfolio_summary(
    portfolio_id: int,
    service: Annotated[PortfolioService, Depends(get_portfolio_service)],
) -> PortfolioSummary:
    try:
        portfolio = service.get_portfolio_summary(portfolio_id)
    except PortfolioNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    positions = [
        PositionRead(
            ticker=p.ticker,
            currency=p.currency,
            quantity=p.quantity,
            average_cost=p.average_cost,
            current_price=p.current_price,
            previous_close=p.previous_close,
            market_value=p.market_value,
            market_value_base=p.market_value_base,
            cost_basis=p.cost_basis,
            unrealized_pnl=p.unrealized_pnl,
            unrealized_pnl_pct=p.unrealized_pnl_pct,
            day_change=p.day_change,
            day_change_pct=p.day_change_pct,
            realized_pnl=p.realized_pnl,
        )
        for p in portfolio.positions
    ]
    return PortfolioSummary(
        id=portfolio.id,
        name=portfolio.name,
        base_currency=portfolio.base_currency,
        positions=positions,
        total_market_value=portfolio.total_market_value,
        total_cost_basis=portfolio.total_cost_basis,
        total_unrealized_pnl=portfolio.total_unrealized_pnl,
        total_realized_pnl=portfolio.total_realized_pnl,
        total_pnl=portfolio.total_pnl,
        total_day_change=portfolio.total_day_change,
        total_day_change_pct=portfolio.total_day_change_pct,
        cash_balance=portfolio.cash_balance,
        total_portfolio_value=portfolio.total_portfolio_value,
    )


@router.post(
    "/{portfolio_id}/transactions", response_model=TransactionRead, status_code=status.HTTP_201_CREATED
)
def add_transaction(
    portfolio_id: int,
    payload: TransactionCreate,
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
    asset_repository: Annotated[AssetRepository, Depends(get_asset_repository)],
    db: DbSession,
) -> TransactionRead:
    if repository.get(portfolio_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")

    if payload.ticker is not None:
        asset_repository.get_or_create(payload.ticker, asset_class=AssetClass.EQUITY)

    try:
        transaction = repository.add_transaction(
            portfolio_id=portfolio_id,
            ticker=payload.ticker,
            transaction_type=payload.transaction_type,
            quantity=payload.quantity,
            price=payload.price,
            fees=payload.fees,
            executed_at=payload.executed_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return TransactionRead(**asdict(transaction))


@router.get("/{portfolio_id}/transactions", response_model=list[TransactionRead])
def list_transactions(
    portfolio_id: int,
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
) -> list[TransactionRead]:
    return [TransactionRead(**asdict(tx)) for tx in repository.get_transactions(portfolio_id)]


@router.delete("/{portfolio_id}/transactions/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transaction(
    portfolio_id: int,
    transaction_id: int,
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
) -> None:
    deleted = repository.delete_transaction(portfolio_id, transaction_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")


@router.get("/{portfolio_id}/risk", response_model=PortfolioRiskResponse)
def get_portfolio_risk(
    portfolio_id: int,
    repository: Annotated[PortfolioRepository, Depends(get_portfolio_repository)],
    market_data: Annotated[MarketDataService, Depends(get_market_data_service)],
    screener: Annotated[MarketScreenerService, Depends(get_market_screener_service)],
    risk_service: Annotated[PortfolioRiskService, Depends(get_portfolio_risk_service)],
    premium_service: Annotated[PremiumWatchlistService, Depends(get_premium_watchlist_service)],
    trade_plan_repo: Annotated[TradePlanRepository, Depends(get_trade_plan_repository)],
    position_signal_snapshot_repo: Annotated[
        PositionSignalSnapshotRepository, Depends(get_position_signal_snapshot_repository)
    ],
    db: DbSession,
    refresh: bool = False,
) -> PortfolioRiskResponse:
    """Full quant risk read on every held ticker - the same recommendation, GARCH,
    Markov, Monte Carlo, backtest and Kelly-sizing pipeline "Analizar activo" runs -
    boiled down to exit_warning / add_candidate / watch / hold so you can act on
    holdings that break down and size up ones whose setup is confirmed, without
    assuming anything a full analysis wouldn't back. See `portfolio_risk_service.py`.
    Also flags an opportunity cost when a materially stronger, already-vetted
    premium candidate exists for a position that isn't already the best use of
    its own capital - see `opportunity_cost.py`.

    Backed by the durable cache (`durable_cache.py`): a normal visit within
    `PORTFOLIO_RISK_DURABLE_TTL` gets back the last computed read instantly,
    however old (down to however stale a restart made it) - never a live,
    tens-of-seconds recompute just because someone looked at this portfolio
    twice in a row. Pass `refresh=true` (the frontend's "Actualizar ahora"
    button) to force a real recompute regardless of freshness."""
    if repository.get(portfolio_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")

    cache_key = f"portfolio_risk:{portfolio_id}"
    if not refresh:
        cached = durable_cache.load_fresh_as(
            db, cache_key, PORTFOLIO_RISK_DURABLE_TTL, PortfolioRiskResponse.model_validate
        )
        if cached is not None:
            return cached

    transactions = repository.get_transactions(portfolio_id)
    tickers = sorted({tx.ticker for tx in transactions if tx.ticker is not None})
    # A personal portfolio isn't confined to one market - combine both curated
    # universes so a European holding's RS Rating is found too, not just US ones.
    universe_snapshot = screener.get_universe_snapshot("us", db=db) + screener.get_universe_snapshot(
        "europe", db=db
    )
    # portfolio_id/transactions/trade_plan_repo enable the independent exit
    # engine (see exit_engine.py, portfolio_risk_service.assess_position_risk)
    # - each holding's own stop/target plan, not just its bare technical read.
    positions = risk_service.get_positions_risk(
        tickers,
        market_data,
        universe_snapshot,
        force_refresh=refresh,
        portfolio_id=portfolio_id,
        transactions=transactions,
        trade_plan_repo=trade_plan_repo,
        position_signal_snapshot_repo=position_signal_snapshot_repo,
    )

    # Same two curated universes as the RS lookup above - a premium candidate
    # from either region is a fair comparison against any held position.
    us_candidates = premium_service.get_premium_watchlist(region="us")
    europe_candidates = premium_service.get_premium_watchlist(region="europe")
    swap_suggestions = find_swap_suggestions(positions, us_candidates + europe_candidates)

    computed_at = datetime.now(UTC)
    response = PortfolioRiskResponse(
        positions=[
            PositionRiskResponse(
                ticker=p.ticker,
                currency=p.currency,
                price=p.price,
                trend=p.trend,
                stage=p.stage,
                ma_cross=p.ma_cross,
                rs_rating=p.rs_rating,
                nearest_support=PriceLevelResponse(**asdict(p.nearest_support)) if p.nearest_support else None,
                nearest_resistance=(
                    PriceLevelResponse(**asdict(p.nearest_resistance)) if p.nearest_resistance else None
                ),
                signal=p.signal,
                score=p.score,
                reasons=p.reasons,
                signals=_core_signals_to_response(p.signals),
                exit_urgency=p.exit_urgency,
                exit_reasons=p.exit_reasons,
                trade_plan=_trade_plan_to_response(p.trade_plan) if p.trade_plan else None,
                r_multiple=p.r_multiple,
                multi_timeframe=_multi_timeframe_to_response(p.multi_timeframe) if p.multi_timeframe else None,
                scaled_exit=_scaled_exit_to_response(p.scaled_exit) if p.scaled_exit else None,
            )
            for p in positions
        ],
        swap_suggestions=[SwapSuggestionResponse(**asdict(s)) for s in swap_suggestions],
        computed_at=computed_at,
    )
    durable_cache.save(db, cache_key, response.model_dump(mode="json"), computed_at=computed_at)
    return response
