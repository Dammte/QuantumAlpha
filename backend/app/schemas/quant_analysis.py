"""Response models for the quant signal suite (recommendation, Markov chain,
GARCH, Monte Carlo, walk-forward backtest, Kelly sizing) - shared between the
single-ticker deep dive (`schemas/ticker_analysis.py`), the portfolio position
risk endpoint, and the premium watchlist (`schemas/market.py`), so all three
surfaces describe the exact same underlying analysis the exact same way.
"""

from pydantic import BaseModel

from app.schemas.common import PriceLevelResponse


class RecommendationFactorResponse(BaseModel):
    label: str
    points: int
    triggered: bool


class RecommendationResponse(BaseModel):
    verdict: str
    score: int
    factors: list[RecommendationFactorResponse]
    stop_loss: float | None
    take_profit: float | None
    take_profit_method: str | None
    risk_reward: float | None


class MarkovChainResponse(BaseModel):
    current_state: int
    current_state_label: str
    state_labels: list[str]
    transition_matrix: list[list[float]]
    state_mean_returns: list[float]
    stationary_distribution: list[float]
    forecast_5d_return: float
    forecast_21d_return: float
    forecast_21d_distribution: list[float]
    prob_bullish_21d: float
    runs_test_z: float
    sequence_looks_random: bool
    order2_justified: bool
    order2_p_value: float


class GarchResponse(BaseModel):
    omega: float
    alpha: float
    beta: float
    persistence: float
    unconditional_vol_annualized: float
    current_vol_annualized: float
    forecast_vol_21d_annualized: float
    vol_percentile: float
    regime: str


class PricePercentilesResponse(BaseModel):
    day: int
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class MonteCarloResponse(BaseModel):
    n_simulations: int
    method: str
    percentiles: list[PricePercentilesResponse]
    probability_of_loss: float
    probability_stop_before_target: float | None
    probability_target_before_stop: float | None
    probability_neither_hit: float | None


class VerdictBucketStatsResponse(BaseModel):
    verdict: str
    n: int
    win_rate: float | None
    mean_return: float | None
    median_return: float | None


class SignificanceTestResponse(BaseModel):
    comparison: str
    mean_difference: float
    t_stat: float
    p_value: float
    p_value_bonferroni: float
    permutation_p_value: float
    significant_at_5pct: bool


class WalkForwardBacktestResponse(BaseModel):
    horizon_days: int
    n_samples: int
    bucket_stats: list[VerdictBucketStatsResponse]
    significance_tests: list[SignificanceTestResponse]
    overall_mean_return: float
    interpretation: str


class KellyPositionSizeResponse(BaseModel):
    win_probability: float
    reward_risk_ratio: float
    full_kelly_fraction: float
    fractional_kelly_fraction: float
    recommended_position_pct: float
    growth_rate_retained_pct: float
    rationale: str


class StatisticalStructureResponse(BaseModel):
    hurst_exponent: float | None
    regime: str
    adf_statistic: float | None
    adf_p_value: float | None
    is_stationary: bool | None


class EntryTimingResponse(BaseModel):
    """See `entry_timing.py` - a read on how much of this setup's move looks
    still-ahead vs already-behind, not a second buy/avoid verdict."""

    status: str  # "optimal" | "valid" | "late" | "extended"
    label: str
    description: str
    atr_multiple: float


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
    mansfield_rs: float | None
    rs_rating: int | None
    minervini_score: int
    minervini_pass: bool
    support_resistance: list[PriceLevelResponse]
    nearest_support: PriceLevelResponse | None
    nearest_resistance: PriceLevelResponse | None
    obv_divergence: str | None
    statistical_structure: StatisticalStructureResponse | None
    market_trend: str | None
    vix_regime: str | None
    is_intraday_snapshot: bool
    recommendation: RecommendationResponse
    entry_timing: EntryTimingResponse | None
    markov: MarkovChainResponse | None
    garch: GarchResponse | None
    monte_carlo: MonteCarloResponse | None
    backtest: WalkForwardBacktestResponse | None
    position_sizing: KellyPositionSizeResponse | None
