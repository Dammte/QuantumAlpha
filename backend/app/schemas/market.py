from datetime import date, datetime

from pydantic import BaseModel

from app.schemas.common import PriceLevelResponse
from app.schemas.quant_analysis import (
    CoreSignalsResponse,
    EntryTriggerResponse,
    GateConditionResponse,
    GradeResponse,
    ImminentCrossResponse,
    MultiTimeframeResponse,
    StopAndTargetResponse,
    TradeGeometryResponse,
)


class NewsArticleResponse(BaseModel):
    title: str
    publisher: str | None
    link: str | None
    published_at: str | None


class TickerSnapshotResponse(BaseModel):
    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    currency: str
    price: float
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_6m: float | None
    change_1y: float | None
    volume: float
    relative_volume: float | None
    rsi14: float | None
    sma20: float | None
    sma50: float | None
    sma150: float | None
    sma200: float | None
    dist_52w_high: float | None
    dist_52w_low: float | None
    atr_multiple: float | None
    adx14: float | None
    plus_di: float | None
    minus_di: float | None
    mansfield_rs: float | None
    rs_rating: int | None
    trend: str
    stage: str | None
    ma_cross: str | None
    minervini_score: int
    minervini_pass: bool
    ma_cross_short: str | None = None
    imminent_cross_short_term: ImminentCrossResponse | None = None


class MoversResponse(BaseModel):
    gainers: list[TickerSnapshotResponse]
    losers: list[TickerSnapshotResponse]
    near_52w_high: list[TickerSnapshotResponse]
    near_52w_low: list[TickerSnapshotResponse]
    high_volume: list[TickerSnapshotResponse]
    oversold: list[TickerSnapshotResponse]
    overbought: list[TickerSnapshotResponse]
    golden_cross: list[TickerSnapshotResponse]
    death_cross: list[TickerSnapshotResponse]
    rs_leaders: list[TickerSnapshotResponse]
    strong_trend: list[TickerSnapshotResponse]


class TrendBreadthResponse(BaseModel):
    total: int
    pct_above_sma50: float
    pct_above_sma200: float
    count_uptrend: int
    count_downtrend: int
    count_sideways: int
    golden_crosses: int
    death_crosses: int
    count_overbought: int
    count_oversold: int
    count_stage2: int
    count_minervini_pass: int
    count_imminent_golden: int = 0
    count_imminent_death: int = 0


class TrendDetailResponse(BaseModel):
    uptrend: list[TickerSnapshotResponse]
    downtrend: list[TickerSnapshotResponse]
    golden_cross: list[TickerSnapshotResponse]
    death_cross: list[TickerSnapshotResponse]
    overbought: list[TickerSnapshotResponse]
    oversold: list[TickerSnapshotResponse]
    stage2: list[TickerSnapshotResponse]
    stage4: list[TickerSnapshotResponse]
    minervini_pass: list[TickerSnapshotResponse]
    strong_trend: list[TickerSnapshotResponse]
    imminent_cross: list[TickerSnapshotResponse] = []


class SupportResistanceResponse(BaseModel):
    ticker: str
    currency: str
    price: float
    levels: list[PriceLevelResponse]


class ProximityItemResponse(BaseModel):
    ticker: str
    sector: str
    currency: str
    price: float
    level: PriceLevelResponse


class IndustryUniverseResponse(BaseModel):
    name: str
    sector: str
    etf: str | None
    tickers: list[str]


class UniverseResponse(BaseModel):
    sectors: dict[str, list[str]]  # sector -> industry names
    industries: list[IndustryUniverseResponse]


class IndexSnapshotResponse(BaseModel):
    name: str
    ticker: str
    price: float | None
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_1y: float | None
    trend: str | None


class VixSnapshotResponse(BaseModel):
    level: float | None
    sma50: float | None
    regime: str
    term_structure: str | None


class MarketRegimeResponse(BaseModel):
    verdict: str
    headline: str
    reasons: list[str]


class MarketContextResponse(BaseModel):
    indices: list[IndexSnapshotResponse]
    vix: VixSnapshotResponse
    regime: MarketRegimeResponse
    news: list[NewsArticleResponse]


class TradePlanResponse(BaseModel):
    """See `domain.models.trade_plan.TradePlan` - the persisted (or, for a
    position opened before this existed, point-in-time reconstructed) stop/
    target/thesis the exit engine judges this position against. `None` when
    there's no currently-open lot to plan for (see `trade_plan_service.py`)."""

    entry_price: float
    entry_date: date
    initial_stop: float | None
    initial_target: float | None
    current_stop: float | None
    highest_close_since_entry: float
    thesis: str
    engine_version: str


class ScaledExitPlanResponse(BaseModel):
    """See `trade_manager.ScaledExitPlan` - a suggested (never automatically
    executed) partial exit at the +1R/+2R milestones, in concrete share
    quantities."""

    action: str  # "none" | "sell_at_1r" | "sell_at_2r"
    shares_to_sell: float
    shares_remaining_after: float
    suggested_new_stop: float | None
    description: str


class PositionRiskResponse(BaseModel):
    ticker: str
    currency: str
    price: float
    trend: str
    stage: str | None
    ma_cross: str | None
    rs_rating: int | None
    nearest_support: PriceLevelResponse | None
    nearest_resistance: PriceLevelResponse | None
    signal: str
    score: int
    reasons: list[str]
    signals: CoreSignalsResponse
    # Added for the independent exit engine (see exit_engine.py) - all
    # additive, `signal`/`score`/`reasons` above are unchanged and still mean
    # exactly what they meant before this existed.
    exit_urgency: str | None  # "exit_now"|"reduce"|"tighten_stop"|"watch"|"hold" - None if no open trade plan
    exit_reasons: list[str]
    trade_plan: TradePlanResponse | None
    r_multiple: float | None
    multi_timeframe: MultiTimeframeResponse | None
    scaled_exit: ScaledExitPlanResponse | None
    bars_held: int | None  # closed daily bars since trade_plan.entry_date - "sesiones mantenidas" in the UI


class PortfolioRiskResponse(BaseModel):
    positions: list[PositionRiskResponse]
    computed_at: datetime  # when this was last actually computed - may be hours old, see durable_cache.py


class PositionDailyStateResponse(BaseModel):
    """Reconstruction (2026-09), Fase 5: one row of `PositionDailyStateORM`,
    read as-is - the precomputed counterpart of `PositionRiskResponse` above,
    written once a day by `daily_close.py` instead of recomputed (and
    durably cached) per request. Deliberately narrower than
    `PositionRiskResponse` - no `signals`/`multi_timeframe`/`scaled_exit` -
    this is the fast "what needs attention today" read (Parte 0, pregunta 1);
    the richer live read at `/portfolios/{id}/risk` is unchanged and still
    the one the position detail card opens into."""

    ticker: str
    trade_date: date
    computed_at: datetime
    urgency: str  # "exit_now" | "reduce" | "tighten_stop" | "watch" | "hold"
    reasons: list[str]
    price: float
    r_multiple: float | None
    current_stop: float | None


class DailyBriefResponse(BaseModel):
    brief_date: date
    computed_at: datetime
    positions_needing_action: int
    new_entry_triggers: int
    new_gate_passes: int
    headline: str


class OpportunityCostNoteResponse(BaseModel):
    """Reconstruction (2026-09), Fase 5 (resto): see `opportunity_cost.py`'s
    own docstring - a holding whose own gate doesn't pass today, next to a
    same-sector Radar candidate whose gate does. Never a second opinion on
    whether to sell `held_ticker` (that's exit_engine.py's job alone)."""

    held_ticker: str
    sector: str
    alternative_ticker: str
    alternative_rs_rating: int | None


class PortfolioTodayResponse(BaseModel):
    # `None` when daily_close.py hasn't run for this portfolio yet - same
    # "empty vs. doesn't exist yet" distinction RadarResponse.computed_at
    # documents.
    brief: DailyBriefResponse | None
    positions: list[PositionDailyStateResponse]
    opportunity_cost: list[OpportunityCostNoteResponse]


class TimeframeCellResponse(BaseModel):
    """Ver `app.services.multi_timeframe.TimeframeCell` - Parte 7 de la
    biblioteca de setups del Radar. `bias`/`stage`/`price_vs_ma` son
    deliberadamente más simples que las lecturas semanal/diaria que ya
    alimentan el gate (`TimeframeReadResponse`, si existiera) - la mensual
    en particular es EXCLUSIVAMENTE informativa, nunca entra en el gate, el
    grado ni ningún gatillo."""

    bias: str  # "bullish" | "neutral" | "bearish" | "unknown"
    stage: str | None
    price_vs_ma: str | None  # "above" | "below"
    note: str


class TimeframeStripResponse(BaseModel):
    """Ver `app.services.multi_timeframe.TimeframeStrip`. `monthly` se
    muestra atenuada en la interfaz (Parte 7.2, literal) - la propia API no
    impone eso, es una convención de presentación en `RadarView.jsx`."""

    monthly: TimeframeCellResponse
    weekly: TimeframeCellResponse
    daily: TimeframeCellResponse


class SetupPerformanceStatsResponse(BaseModel):
    """Ver `app.domain.models.setup_performance.SetupPerformance` - siempre
    la fila SIN segmentar (`grade`/`market_regime` en `None`) de un nombre
    de setup, la única que `GET /market/radar` adjunta (Parte 10.2/11.1).
    `None` en `SetupMatchResponse.measured_stats` cuando
    `scripts/setup_replay_study.py` nunca corrió para este nombre - nunca
    una medición fabricada."""

    n_observations: int
    trigger_rate: float | None
    win_rate: float | None
    expectancy_r: float | None
    median_bars_held: float | None
    mae_p80_pct: float | None
    failure_rate_3d: float | None


class SetupTickerHistoryResponse(BaseModel):
    """Ver `app.domain.models.setup_ticker_history.SetupTickerHistory`
    (Parte 11.2) - "este valor ha formado 4 VCP en 5 años; 3 dispararon y 2
    alcanzaron objetivo". Conteos literales de ESTE ticker, sin umbral de
    muestra mínima (a diferencia de `SetupPerformanceStatsResponse`) - un
    n=1 es un hecho honesto, no algo que ocultar."""

    n_observations: int
    n_triggered: int
    n_target_hit: int
    first_ready_date: date
    last_ready_date: date


class SetupMatchResponse(BaseModel):
    """Ver `app.services.setups.types.SetupMatch` - una coincidencia de un
    detector de la biblioteca de setups del Radar (en curso,
    `docs/quant_methodology.md` §28). Ningún campo aquí es una puntuación:
    el propio `family`/`name` identifica qué setup es, y elegir entre varios
    a la vez para un mismo ticker es trabajo de `setups/arbitration.py`
    (fase posterior), no de esta respuesta."""

    family: str
    name: str
    label_es: str
    stage: str  # "forming" | "ready" | "triggered" | "failed"
    bars_in_stage: int
    timeframe: str  # "daily" | "weekly"
    trigger_price: float | None
    trigger_condition: str
    invalidation_price: float | None
    invalidation_condition: str
    evidence: dict[str, float | int | str]
    narrative_es: str
    confidence: str  # "measured" | "thin" | "unvalidated"
    # Auditoria del Radar, bloque D - ver `setups/horizon.py` para el
    # criterio. `horizon` puede ser `None` en filas persistidas antes de
    # este bloque, hasta que `daily_close.py` vuelva a correr para ese
    # ticker.
    horizon: str | None = None  # "short" (2-10 sesiones) | "medium" (3-10 semanas)
    expected_sessions_to_trigger: int | None = None
    # Parte 10.2/11.1 (§28.x) - adjuntado en el endpoint, no persistido
    # junto al resto de este dict (ver `setup_performance`'s propio
    # docstring: una sola fila por nombre de setup en toda la base de
    # datos, no una copia por cada fila de `TickerDailyState` que lo
    # muestre). `None` sin medición todavía.
    measured_stats: SetupPerformanceStatsResponse | None = None
    # Parte 11.2 - mismo criterio que measured_stats, pero por ticker
    # concreto en vez de agregado sobre el universo. `None` sin historial
    # todavía para este (ticker, nombre de setup).
    ticker_history: SetupTickerHistoryResponse | None = None


class RadarScoreResponse(BaseModel):
    """Ver `app.services.setups.scoring.RadarScoreBreakdown` - cada
    componente ya multiplicado por su peso (`app.core.trading_params.RADAR_SCORE_WEIGHT_*`)
    y cada penalización ya en negativo. `total` es la suma de todo,
    recortada a [0, 100]."""

    total: float
    setup_quality: float
    relative_strength: float
    trigger_proximity: float
    geometry_quality: float
    volume_confirmation: float
    earnings_penalty: float
    high_atr_penalty: float


class RadarItemResponse(BaseModel):
    """Reconstruction (2026-09), Fase 5: one row of `TickerDailyStateORM`,
    read as-is - no live computation behind this response at all, unlike
    every field above it in this file. See `docs/quant_methodology.md` §25
    and `ticker_daily_state.py`'s own docstring."""

    ticker: str
    region: str
    trade_date: date
    computed_at: datetime
    price: float
    currency: str
    trend: str
    stage: str | None
    rs_rating: int | None
    adx14: float | None
    atr_multiple: float | None
    rsi14: float | None
    gate_passes: bool
    gate_conditions: list[GateConditionResponse]
    gate_version: str
    entry_trigger: EntryTriggerResponse | None
    stop_and_target: StopAndTargetResponse | None
    # Parte 7 (later pass): the ticker-only half of the real design
    # (`trade_geometry.compute_entry_geometry`), persisted by `daily_close.py`
    # - `None` for rows computed before this column existed. Never sized here
    # (see `TradeGeometryResponse`'s own docstring) - pass `portfolio_id` to
    # size every viable one against that portfolio's capital instead.
    entry_geometry: TradeGeometryResponse | None = None
    # Parte 5.3 (later pass): `levels_engine.GradeResult`, persisted by
    # `daily_close.py` as a plain dict - `None` for rows computed before this
    # column existed, or when there was no viable entry geometry to grade.
    grade: GradeResponse | None = None
    # Biblioteca de setups del Radar (en curso, §28) - `[]` es "sin
    # coincidencias hoy", el resultado normal para la mayoría de tickers la
    # mayoría de días; `None` solo para una fila anterior a esta columna.
    setups: list[SetupMatchResponse] | None = None
    # Parte 7 (§28.x) - `None` solo para una fila anterior a esta columna.
    timeframe_strip: TimeframeStripResponse | None = None
    # Parte 8/9 (§28.x): agrupación/ordenación del Radar por sector. `sector`
    # ya en español (`market_universe.sector_of`) - `None` solo para una
    # fila anterior a esta columna, o un ticker sin sector conocido.
    sector: str | None = None
    sector_rs_percentile: int | None = None
    # Auditoria del Radar, bloque E2: entradas del score compuesto,
    # persistidas por `daily_close.py` desde datos que ya calculaba para
    # otros fines (nunca cómputo nuevo en el propio request). `None` para
    # filas anteriores a este bloque.
    relative_volume: float | None = None
    next_earnings_date: date | None = None
    atr_pct: float | None = None
    # Auditoria del Radar, bloque E2: "necesito saber por qué un valor está
    # el primero, viendo el desglose del score" - calculado en el propio
    # endpoint (`app.services.setups.scoring`), nunca persistido (depende
    # del resto de candidatos del día vía `atr_pct_p90_in_universe`, así que
    # no tiene un valor estable por sí solo fuera de un request concreto).
    # `None` solo si el ticker no tiene ningún setup ni geometría con la que
    # construir un score en absoluto.
    score: RadarScoreResponse | None = None


class RadarResponse(BaseModel):
    items: list[RadarItemResponse]
    # `None` when `daily_close.py` hasn't run for this region yet (a fresh
    # deploy, or the cron job not yet enabled - see render.yaml) - an empty
    # Radar and a Radar that simply doesn't exist yet are different states,
    # same "prefiero una lista vacía a datos a medias" rule CLAUDE.md already
    # applies elsewhere.
    computed_at: datetime | None
    # Parte 9.2, literal: "si no hay nada, el Radar lo dice con claridad" -
    # solo se rellena cuando `computed_at` existe (el job SÍ corrió) pero
    # `items` quedó vacío tras los cortes - nunca para rellenar el hueco de
    # "todavía no hay datos", que `computed_at is None` ya distingue.
    message: str | None = None
    # Parte 12.1, literal: el contador de cabecera («8 de 412 analizados») -
    # cuántas filas `daily_close.py` calculó hoy para esta región, ANTES del
    # filtro de gate/disparador y de los cortes de la Parte 9.2. `0` cuando
    # `computed_at is None` (el job no ha corrido todavía).
    total_analyzed: int = 0


class CorrelationWarningResponse(BaseModel):
    """See portfolio_construction_service.CorrelationWarning - two held
    tickers correlated at or beyond the threshold: one bet with two names,
    not two independent ones."""

    ticker_a: str
    ticker_b: str
    correlation: float


class SectorConcentrationResponse(BaseModel):
    sector: str
    weight_pct: float
    tickers: list[str]


class PositionRiskContributionResponse(BaseModel):
    ticker: str
    weight_pct: float
    risk_contribution_pct: float


class AggregateRiskReportResponse(BaseModel):
    total_risk_amount: float
    total_risk_pct_of_capital: float | None
    exceeds_limit: bool


class PortfolioConstructionResponse(BaseModel):
    """See portfolio_construction_service.py (D12) - portfolio-level risk no
    single position's own analysis can see: correlated bets, sector
    concentration, and the real money lost if every stop got hit at once.
    Positions with no return history yet (a brand-new listing) or no open
    trade plan (aggregate_risk only) are simply absent from the relevant
    section rather than assumed - see each pure function's own docstring."""

    # ticker -> ticker -> correlation, only tickers with return history; None
    # for a pair whose correlation can't be computed (e.g. one side has zero
    # variance over the window) rather than a silently wrong number.
    correlation_matrix: dict[str, dict[str, float | None]]
    correlated_pairs: list[CorrelationWarningResponse]
    sector_concentrations: list[SectorConcentrationResponse]
    concentrated_sectors: list[SectorConcentrationResponse]
    risk_contributions: list[PositionRiskContributionResponse]
    portfolio_volatility_pct: float | None  # annualized realized vol of the portfolio as a whole
    volatility_target_pct: float
    suggested_to_trim: list[PositionRiskContributionResponse]  # top contributors to trim if vol exceeds target
    aggregate_risk: AggregateRiskReportResponse
    tickers_without_trade_plan: list[str]  # excluded from aggregate_risk - no known stop to size risk against
    computed_at: datetime


# --- Relationship map (Tercera auditoría, Bloque G) --------------------------


class StatisticalRelationResponse(BaseModel):
    ticker: str
    sector: str | None
    correlation_60d: float | None
    correlation_250d: float | None
    relative_beta: float | None
    lead_lag_days: int | None
    lead_lag_correlation: float | None
    comovement_extreme_days_pct: float | None
    is_diverging: bool


class SectorPeerResponse(BaseModel):
    ticker: str
    sector: str
    industry: str
    rs_rating: int | None
    trend: str


class RelationshipMapResponse(BaseModel):
    ticker: str
    region: str
    statistical: list[StatisticalRelationResponse]
    sector_peers: list[SectorPeerResponse]
    computed_at: datetime
