from datetime import UTC, date, datetime

from sqlalchemy import JSON, ForeignKey, Numeric, String, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.models.asset import AssetClass
from app.domain.models.transaction import TransactionType
from app.infrastructure.db.session import Base


class PortfolioORM(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    base_currency: Mapped[str] = mapped_column(String(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    transactions: Mapped[list["TransactionORM"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )


class AssetORM(Base):
    __tablename__ = "assets"

    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    asset_class: Mapped[AssetClass] = mapped_column(SAEnum(AssetClass))
    currency: Mapped[str] = mapped_column(String(3), default="USD")


class TransactionORM(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    ticker: Mapped[str | None] = mapped_column(ForeignKey("assets.ticker"), nullable=True)
    transaction_type: Mapped[TransactionType] = mapped_column(SAEnum(TransactionType))
    quantity: Mapped[float] = mapped_column(Numeric(20, 8))
    price: Mapped[float] = mapped_column(Numeric(20, 8))
    fees: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    executed_at: Mapped[datetime]

    portfolio: Mapped["PortfolioORM"] = relationship(back_populates="transactions")


class PriceBarORM(Base):
    __tablename__ = "price_bars"
    __table_args__ = (UniqueConstraint("ticker", "trade_date", name="uq_price_bar_ticker_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("assets.ticker"), index=True)
    trade_date: Mapped[date] = mapped_column(index=True)
    open: Mapped[float] = mapped_column(Numeric(20, 8))
    high: Mapped[float] = mapped_column(Numeric(20, 8))
    low: Mapped[float] = mapped_column(Numeric(20, 8))
    close: Mapped[float] = mapped_column(Numeric(20, 8))
    volume: Mapped[float] = mapped_column(Numeric(20, 4))


class RecommendationSnapshotORM(Base):
    """Immutable audit trail: one row per real "Analizar activo" call, capturing
    exactly what the system said at that moment - not a synthetic replay, the
    actual live verdict a user actually saw. This is the concrete answer to
    an external audit's "cómo demuestras que una alerta estuvo fundamentada":
    without a persisted record, there's no way to later check whether a past
    verdict panned out, or to tell a genuine improvement in the recommendation
    engine from selective memory of the hits. Deliberately has no foreign key
    to a portfolio/transaction - this logs *analysis*, not trades; a ticker
    can (and will) be looked up many times with no position ever taken.
    """

    __tablename__ = "recommendation_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), index=True)
    verdict: Mapped[str] = mapped_column(String(20))
    score: Mapped[int]
    price: Mapped[float] = mapped_column(Numeric(20, 8))
    currency: Mapped[str] = mapped_column(String(3))
    horizon: Mapped[str] = mapped_column(String(10))
    engine_version: Mapped[str] = mapped_column(String(40))
    # List of {"label": str, "points": int, "triggered": bool} - the exact
    # factor breakdown behind the score, not just the final number, so a
    # later reviewer can see *why*, not only *what*.
    factors: Mapped[list] = mapped_column(JSON)


class ComputationCacheORM(Base):
    """Durable, restart-proof companion to each service's own in-process cache
    (see `market_screener_service.py`, `portfolio_risk_service.py`,
    `premium_watchlist_service.py`) - keyed by an arbitrary string the caller
    controls (e.g. "portfolio_risk:3", "universe_snapshot:us"), holding
    whatever JSON-safe payload that cache slot last computed plus when.

    Exists because this is a `plan: starter` Render web service: it doesn't
    spin down between requests, but it does restart on every deploy, and a
    personal project under active development deploys often - without this,
    every single deploy would force the next visit to eat the full quant-suite
    compute cost synchronously (tens of seconds, blocking the whole
    single-process server for anyone else using it at that moment) purely
    because the in-memory cache was empty again, not because the data was
    actually stale. See `app/services/durable_cache.py`.
    """

    __tablename__ = "computation_cache"

    cache_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    computed_at: Mapped[datetime] = mapped_column(index=True)
    payload: Mapped[dict] = mapped_column(JSON)
