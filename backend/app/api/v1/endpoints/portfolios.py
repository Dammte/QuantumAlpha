from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import (
    DbSession,
    get_asset_repository,
    get_market_data_service,
    get_market_screener_service,
    get_portfolio_repository,
    get_portfolio_service,
)
from app.domain.models.asset import AssetClass
from app.infrastructure.db.repositories.asset_repository import AssetRepository
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.schemas.market import PortfolioRiskResponse, PositionRiskResponse, PriceLevelResponse
from app.schemas.portfolio import PortfolioCreate, PortfolioRead, PortfolioSummary, PositionRead
from app.schemas.quant_analysis import CoreSignalsResponse
from app.schemas.transaction import TransactionCreate, TransactionRead
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.portfolio_risk_service import get_portfolio_positions_risk
from app.services.portfolio_service import PortfolioNotFoundError, PortfolioService
from app.services.ticker_analysis_service import CoreTickerSignals

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


def _core_signals_to_response(signals: CoreTickerSignals) -> CoreSignalsResponse:
    data = asdict(signals)
    data["trend"] = signals.trend.value
    data["stage"] = signals.stage.value if signals.stage else None
    return CoreSignalsResponse(**data)


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
) -> PortfolioRiskResponse:
    """Full quant risk read on every held ticker - the same recommendation, GARCH,
    Markov, Monte Carlo, backtest and Kelly-sizing pipeline "Analizar activo" runs -
    boiled down to exit_warning / add_candidate / watch / hold so you can act on
    holdings that break down and size up ones whose setup is confirmed, without
    assuming anything a full analysis wouldn't back. See `portfolio_risk_service.py`."""
    if repository.get(portfolio_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")

    tickers = sorted({tx.ticker for tx in repository.get_transactions(portfolio_id)})
    # A personal portfolio isn't confined to one market - combine both curated
    # universes so a European holding's RS Rating is found too, not just US ones.
    universe_snapshot = screener.get_universe_snapshot("us") + screener.get_universe_snapshot("europe")
    positions = get_portfolio_positions_risk(tickers, market_data, universe_snapshot)

    return PortfolioRiskResponse(
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
            )
            for p in positions
        ]
    )
