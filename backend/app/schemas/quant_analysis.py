"""Response models for the quant signal suite (the levels/triggers gate,
triple-barrier backtest) - shared between the single-ticker deep dive
(`schemas/ticker_analysis.py`) and the portfolio position risk endpoint, so
both surfaces describe the exact same underlying analysis the exact same way.

2026-09: Markov chain, GARCH, Monte Carlo, Kelly sizing and the Hurst/ADF
statistical-structure read were removed from this suite - see
`recommendation_engine.py`'s module docstring and `docs/quant_methodology.md`.

2026-09 (reconstruction, Fase 4): `RecommendationResponse`/
`RecommendationFactorResponse` (the old weighted checklist's verdict/score/
factors) are replaced by `GateResultResponse`/`GateConditionResponse` below -
see `levels_engine.py`'s own docstring for why a pass/fail gate replaced a
score. `recommendation_engine.Recommendation` still exists (some of its
individual factors are still measured by `scripts/factor_ablation_study.py`
alongside the new gate's own conditions, since Fase 8's reorientation - see
that script's own docstring), it just no longer has an API response shape -
nothing in the live API serializes it anymore.
"""

from pydantic import BaseModel

from app.schemas.common import PriceLevelResponse


class GateConditionResponse(BaseModel):
    label: str
    passed: bool


class EntryTriggerResponse(BaseModel):
    trigger_type: str  # "breakout" | "pullback_bounce"
    trigger_price: float
    already_triggered: bool


class StopAndTargetResponse(BaseModel):
    stop_loss: float | None
    take_profit: float | None
    take_profit_method: str | None
    risk_reward: float | None


class GateResultResponse(BaseModel):
    passes: bool
    conditions: list[GateConditionResponse]
    entry_trigger: EntryTriggerResponse | None
    stop_and_target: StopAndTargetResponse | None


class ImminentCrossResponse(BaseModel):
    """See `technical_analysis.detect_imminent_cross` - a projected, not yet
    confirmed, MA50/MA200 crossover."""

    direction: str  # "golden" | "death"
    bars_until: int
    r_squared: float


class CrossQualityResponse(BaseModel):
    """See `technical_analysis.detect_cross_with_quality`."""

    direction: str  # "golden" | "death"
    bars_since: int
    separation_atr: float | None
    fast_slope: float
    slow_slope: float
    volume_confirmation: float | None
    quality: str  # "strong" | "weak" | "noise"


class TimeframeReadResponse(BaseModel):
    """See `multi_timeframe.TimeframeRead` - one timeframe's full technical
    read, off closed bars only."""

    timeframe: str  # "weekly" | "daily" | "intraday_1h"
    trend: str
    stage: str | None
    ma_cross_50_200: str | None
    ma_cross_20_50: str | None
    cross_quality_20_50: CrossQualityResponse | None
    imminent_cross_50_200: ImminentCrossResponse | None
    imminent_cross_20_50: ImminentCrossResponse | None
    macd_cross: str | None
    macd_histogram_turning: str | None
    rsi14: float | None
    adx14: float | None
    di_bias: str | None
    price_vs_sma20: str | None
    price_vs_sma50: str | None
    price_vs_sma200: str | None
    bars_since_cross: int | None


class MultiTimeframeResponse(BaseModel):
    """See `multi_timeframe.MultiTimeframeRead` - the weekly-bias/daily-
    execution top-down read, the semáforo multi-timeframe the frontend shows
    per position."""

    weekly: TimeframeReadResponse | None
    daily: TimeframeReadResponse
    intraday: TimeframeReadResponse | None
    alignment: str  # "bullish_aligned" | "bearish_aligned" | "conflicted" | "transitioning"
    alignment_score: float
    conflicts: list[str]


class TradingMetricsResponse(BaseModel):
    """See `backtest_engine.TradingMetrics` - every P&L figure here
    (win_rate, avg_win/avg_loss, expectancy, profit_factor, max_drawdown) is
    net of the round-trip cost estimate; avg_mae_pct/avg_mfe_pct stay gross
    on purpose (intrabar price action, not P&L - see that dataclass's own
    docstring)."""

    n_trades: int
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy_pct: float | None
    profit_factor: float | None
    avg_mae_pct: float | None
    avg_mfe_pct: float | None
    max_drawdown_pct: float | None
    avg_bars_held_winners: float | None
    avg_bars_held_losers: float | None
    gross_return_pct: float | None
    net_return_pct: float | None


class TripleBarrierBacktestResponse(BaseModel):
    """See `backtest_engine.TripleBarrierBacktestResult`: triple-barrier
    labeling (stop/target/vertical, gap-aware), replayed with the system's
    real Chandelier trailing, net of costs, against buy-and-hold and
    random-entry benchmarks at the exact same entry points. 2026-09: the
    legacy walk-forward backtest this used to sit alongside
    (`WalkForwardBacktestResponse`, naive fixed-horizon buy-and-hold, no
    stop/target awareness) was retired from the API surface - see
    quant_methodology.md."""

    horizon_days: int
    n_signals_evaluated: int
    strategy_fixed: TradingMetricsResponse
    strategy_trailing: TradingMetricsResponse
    buy_and_hold: TradingMetricsResponse
    random_entries: TradingMetricsResponse


class CoreSignalsResponse(BaseModel):
    """Everything the recommendation engine and its supporting quant models
    produce for one ticker at one point in time - the same bundle `TickerAnalysisService.analyze()`
    builds for "Analizar activo", reused as-is for portfolio positions and the premium watchlist
    so a signal is never computed two different ways in two different places."""

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
    macd_line: float | None
    macd_signal: float | None
    macd_histogram: float | None
    adx14: float | None
    plus_di: float | None
    minus_di: float | None
    atr14: float | None
    atr_multiple: float | None
    sma20: float | None
    sma50: float | None
    sma150: float | None
    sma200: float | None
    dist_52w_high: float | None
    dist_52w_low: float | None
    trend: str
    stage: str | None
    ma_cross: str | None
    imminent_cross: ImminentCrossResponse | None
    imminent_cross_short_term: ImminentCrossResponse | None
    candlestick_pattern: str | None
    mansfield_rs: float | None
    rs_rating: int | None
    minervini_score: int
    minervini_pass: bool
    support_resistance: list[PriceLevelResponse]
    nearest_support: PriceLevelResponse | None
    nearest_resistance: PriceLevelResponse | None
    obv_divergence: str | None
    market_trend: str | None
    vix_regime: str | None
    is_intraday_snapshot: bool
    gate: GateResultResponse
