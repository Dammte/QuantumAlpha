from datetime import UTC, date, datetime

from sqlalchemy import JSON, ForeignKey, Index, Numeric, String, UniqueConstraint
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


class TradePlanORM(Base):
    """Persists what `recommendation_engine.compute_stop_and_target` proposed
    at (or reconstructed for, point-in-time, if the position predates this
    table) a position's entry - the exit engine's reference for judging that
    position, independent of today's buy-side checklist score. See
    `TradePlan` (domain) and `trade_plan_service.py`.

    Deliberately no unique constraint on (portfolio_id, ticker): a ticker can
    be bought, fully sold, and bought again, and each such lot gets its own
    row (`closed_at` marks a lot as done) rather than overwriting history -
    this doubles as a real trade log for the signal-performance work later.
    The *open* plan for a ticker is always the one row with `closed_at IS
    NULL` (`TradePlanRepository.get_open`), which a plain index on
    (portfolio_id, ticker) makes cheap to find without needing a DB-level
    uniqueness guarantee.
    """

    __tablename__ = "trade_plans"
    __table_args__ = (Index("ix_trade_plans_portfolio_ticker", "portfolio_id", "ticker"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    ticker: Mapped[str] = mapped_column(String(20))
    entry_price: Mapped[float] = mapped_column(Numeric(20, 8))
    entry_date: Mapped[date]
    initial_stop: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    initial_target: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    current_stop: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    highest_close_since_entry: Mapped[float] = mapped_column(Numeric(20, 8))
    initial_quantity: Mapped[float] = mapped_column(Numeric(20, 8))
    thesis: Mapped[str] = mapped_column(String(500), default="")
    engine_version: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    closed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class PositionSignalSnapshotORM(Base):
    """Fase 0 instrumentation (docs/quant_methodology.md): the audit trail
    `RecommendationSnapshotORM` never kept for *position-level* signals - the
    buy-side verdict from "Analizar activo" was recorded, but
    `portfolio_risk_service.py`'s own `signal`/`exit_urgency` for a held
    position never was, so "how many times did the system say hold and the
    stock fell" could never be answered even in principle. Written only on a
    genuinely fresh (non-cached) risk evaluation - see
    `PortfolioRiskService`'s cache and `assess_position_risk`'s
    `position_signal_snapshot_repo` parameter.
    """

    __tablename__ = "position_signal_snapshots"
    __table_args__ = (Index("ix_position_signal_snapshots_portfolio_ticker", "portfolio_id", "ticker"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), index=True)
    signal: Mapped[str] = mapped_column(String(20))
    exit_urgency: Mapped[str | None] = mapped_column(String(20), nullable=True)
    score: Mapped[int]
    price: Mapped[float] = mapped_column(Numeric(20, 8))
    r_multiple: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    engine_version: Mapped[str] = mapped_column(String(40))


class ComputationCacheORM(Base):
    """Durable, restart-proof companion to each service's own in-process cache
    (see `market_screener_service.py`, `portfolios.py`) - keyed by an
    arbitrary string the caller controls (e.g. "portfolio_risk:3",
    "universe_snapshot:us"), holding
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


class JobRunORM(Base):
    """Reconstruction (2026-09), Fase 2: one execution of a precompute cron
    job (`daily_close.py`, `intraday_refresh.py`,
    `refresh_universe_membership.py`) - see `JobRun` (domain) for why this
    exists."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_name: Mapped[str] = mapped_column(String(60), index=True)
    started_at: Mapped[datetime] = mapped_column(index=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(20))  # "running" | "success" | "failed"
    rows_processed: Mapped[int] = mapped_column(default=0)
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class TickerDailyStateORM(Base):
    """Reconstruction (2026-09), Fase 2: one ticker's precomputed
    levels/triggers read for one trading day - see `TickerDailyState`
    (domain) for the full reasoning. `gate_conditions` mirrors
    `levels_engine.GateCondition` as plain JSON-safe dicts, same choice
    `RecommendationSnapshotORM.factors` already made."""

    __tablename__ = "ticker_daily_states"
    __table_args__ = (
        UniqueConstraint("region", "ticker", "trade_date", name="uq_ticker_daily_state_region_ticker_date"),
        Index("ix_ticker_daily_states_region_date", "region", "trade_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    region: Mapped[str] = mapped_column(String(20))
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    trade_date: Mapped[date] = mapped_column()
    computed_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    price: Mapped[float] = mapped_column(Numeric(20, 8))
    currency: Mapped[str] = mapped_column(String(3))
    trend: Mapped[str] = mapped_column(String(20))
    stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    rs_rating: Mapped[int | None] = mapped_column(nullable=True)
    adx14: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    atr_multiple: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    rsi14: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    gate_passes: Mapped[bool] = mapped_column()
    gate_conditions: Mapped[list] = mapped_column(JSON)
    gate_version: Mapped[str] = mapped_column(String(40))
    entry_trigger_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    entry_trigger_price: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    entry_already_triggered: Mapped[bool] = mapped_column(default=False)
    stop_loss: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    take_profit_method: Mapped[str | None] = mapped_column(String(60), nullable=True)
    risk_reward: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    entry_geometry: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    grade: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    setups: Mapped[list | None] = mapped_column(JSON, nullable=True)
    timeframe_strip: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sector: Mapped[str | None] = mapped_column(String(60), nullable=True)
    sector_rs_percentile: Mapped[int | None] = mapped_column(nullable=True)
    relative_volume: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    next_earnings_date: Mapped[date | None] = mapped_column(nullable=True)
    atr_pct: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)


class TickerIntradayStateORM(Base):
    """Reconstruction (2026-09), Fase 2: the latest intraday quote/trigger
    re-check for one ticker - see `TickerIntradayState` (domain). Always
    overwritten in place (primary key is just the ticker), never a history -
    `TriggerEventORM` is where a genuine intraday trigger firing gets
    permanently recorded."""

    __tablename__ = "ticker_intraday_states"

    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    region: Mapped[str] = mapped_column(String(20))
    updated_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    price: Mapped[float] = mapped_column(Numeric(20, 8))
    entry_already_triggered: Mapped[bool | None] = mapped_column(nullable=True)


class PositionDailyStateORM(Base):
    """Reconstruction (2026-09), Fase 2: one open position's precomputed
    exit_engine read for one trading day - see `PositionDailyState`
    (domain)."""

    __tablename__ = "position_daily_states"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "ticker", "trade_date", name="uq_position_daily_state_portfolio_ticker_date"
        ),
        Index("ix_position_daily_states_portfolio_date", "portfolio_id", "trade_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    trade_date: Mapped[date] = mapped_column()
    computed_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    urgency: Mapped[str] = mapped_column(String(20))
    reasons: Mapped[list] = mapped_column(JSON)
    price: Mapped[float] = mapped_column(Numeric(20, 8))
    r_multiple: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    current_stop: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    engine_version: Mapped[str] = mapped_column(String(40))


class TriggerEventORM(Base):
    """Reconstruction (2026-09), Fase 2: append-only log of a detected state
    change (a gate flipping, an entry firing, an exit urgency escalating) -
    see `TriggerEvent` (domain) for why this is a separate table from the
    daily-state ones above."""

    __tablename__ = "trigger_events"
    __table_args__ = (
        Index("ix_trigger_events_entity", "entity_type", "entity_key"),
        Index("ix_trigger_events_occurred_at", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(20))  # "ticker" | "position"
    entity_key: Mapped[str] = mapped_column(String(60))
    event_type: Mapped[str] = mapped_column(String(40))
    previous_value: Mapped[str | None] = mapped_column(String(60), nullable=True)
    new_value: Mapped[str | None] = mapped_column(String(60), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    details: Mapped[dict] = mapped_column(JSON)


class DailyBriefORM(Base):
    """Reconstruction (2026-09), Fase 2: one portfolio's precomputed "what
    changed since yesterday" summary for one day - see `DailyBrief`
    (domain)."""

    __tablename__ = "daily_briefs"
    __table_args__ = (UniqueConstraint("portfolio_id", "brief_date", name="uq_daily_brief_portfolio_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    brief_date: Mapped[date] = mapped_column(index=True)
    computed_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    positions_needing_action: Mapped[int] = mapped_column(default=0)
    new_entry_triggers: Mapped[int] = mapped_column(default=0)
    new_gate_passes: Mapped[int] = mapped_column(default=0)
    headline: Mapped[str] = mapped_column(String(500))


class UniverseMembershipORM(Base):
    """One row per (region, ticker, as_of_date) - the D14 fix (Segunda
    auditoría, Bloque 3). See `UniverseMember`'s docstring for why this
    persists a dated snapshot instead of overwriting the previous one on
    every refresh."""

    __tablename__ = "universe_memberships"
    __table_args__ = (
        UniqueConstraint("region", "ticker", "as_of_date", name="uq_universe_membership_region_ticker_date"),
        Index("ix_universe_memberships_region_date", "region", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    region: Mapped[str] = mapped_column(String(20))
    ticker: Mapped[str] = mapped_column(String(20))
    sector: Mapped[str | None] = mapped_column(String(80), nullable=True)
    as_of_date: Mapped[date] = mapped_column()
    source: Mapped[str] = mapped_column(String(20))  # "live" | "curated_fallback"
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class SetupPerformanceORM(Base):
    """Parte 10.2 de la biblioteca de setups del Radar - ver
    `app.domain.models.setup_performance.SetupPerformance` para el porqué
    de que esta tabla no tenga una restricción UNIQUE declarada (una foto
    completa reemplazada entera por `SetupPerformanceRepository.replace_all`,
    no un histórico acumulado)."""

    __tablename__ = "setup_performance"

    id: Mapped[int] = mapped_column(primary_key=True)
    setup_name: Mapped[str] = mapped_column(String(60), index=True)
    family: Mapped[str] = mapped_column(String(30))
    grade: Mapped[str | None] = mapped_column(String(1), nullable=True)
    market_regime: Mapped[str | None] = mapped_column(String(30), nullable=True)
    n_observations: Mapped[int] = mapped_column()
    trigger_rate: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    expectancy_r: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    median_bars_held: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    mae_p80_pct: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    failure_rate_3d: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    confidence: Mapped[str] = mapped_column(String(20))
    computed_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class SetupTickerHistoryORM(Base):
    """Parte 11.2 de la biblioteca de setups del Radar - ver
    `app.domain.models.setup_ticker_history.SetupTickerHistory` para el
    porqué de que esta tabla guarde conteos literales (no una tasa con
    umbral de confianza) y de que tampoco declare una restricción UNIQUE
    (misma foto-completa-reemplazada-entera que `setup_performance`)."""

    __tablename__ = "setup_ticker_history"
    __table_args__ = (Index("ix_setup_ticker_history_ticker_region", "ticker", "region"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20))
    region: Mapped[str] = mapped_column(String(20))
    setup_name: Mapped[str] = mapped_column(String(60))
    family: Mapped[str] = mapped_column(String(30))
    n_observations: Mapped[int] = mapped_column()
    n_triggered: Mapped[int] = mapped_column()
    n_target_hit: Mapped[int] = mapped_column()
    first_ready_date: Mapped[date] = mapped_column()
    last_ready_date: Mapped[date] = mapped_column()
    computed_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
