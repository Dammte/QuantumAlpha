"""Orchestrates the single-ticker "deep dive": every indicator this app knows how
to compute, a Gann fan, seasonality, historical analogs, news, and a rule-based
buy/wait/avoid recommendation, all for one ticker on demand.

Unlike the market screener (which scans ~170 tickers and has to stay fast and
cheap per-ticker), this runs once per user search, so it can afford a decade of
history and a couple of slower per-ticker calls (fundamentals, news).

`compute_core_signals()` holds the quant core of that deep dive (recommendation,
Markov chain, GARCH, Monte Carlo, walk-forward backtest, Kelly sizing) as a
function of a plain OHLCV frame, with none of the extra per-ticker network calls
(fundamentals/news/holders) or chart-only series. It exists so the premium
watchlist and the portfolio-position risk check can run the *exact same*
analysis "Analizar activo" would - not a cheaper approximation of it - which is
the whole point of both features: a ticker is never called "premium" or a
holding never flagged "sell" on a different, laxer basis than what you'd see by
searching it directly.

One deliberate, documented exception: the recommendation engine's fundamentals
factor (revenue growth, profit margin, leverage - see `recommendation_engine.py`)
needs a `TickerInfo.info()` call per ticker, which is exactly the N-extra-calls
cost that made the portfolio-risk endpoint hang in production once already (see
`PortfolioRiskService`'s docstring). `compute_core_signals()` accepts
`revenue_growth`/`profit_margins`/`debt_to_equity` as optional pre-fetched
inputs (default None, contributing nothing) rather than fetching them itself -
only `TickerAnalysisService.analyze()`, which already pays for a fundamentals
fetch for the info card, passes them in. Volume/OBV divergence has no such
cost (it's derived from the same OHLCV frame every caller already has) and is
always computed for everyone.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from app.domain.models.ticker_analysis import PricePoint, TickerAnalysis
from app.services import analysis_tools as at
from app.services import statistical_structure as stats_structure
from app.services import technical_analysis as ta
from app.services.entry_timing import EntryTiming, assess_entry_timing
from app.services.kelly_criterion import KellyResult, recommend_position_size, win_probability_from_barriers
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import VIX_TICKER, benchmark_for_ticker
from app.services.markov_chain_model import MarkovChainResult, analyze_markov_chain
from app.services.monte_carlo_simulation import MonteCarloResult, simulate_and_analyze
from app.services.recommendation_engine import Recommendation, build_recommendation
from app.services.statistical_structure import StatisticalStructure, compute_statistical_structure
from app.services.volatility_model import GarchResult, fit_garch
from app.services.walk_forward_backtest import WalkForwardBacktestResult, run_walk_forward_backtest

HISTORY_YEARS = 10
CHART_BARS = 504  # ~2 trading years
MIN_BARS_REQUIRED = 60

# Monte Carlo horizon presets, keyed by the API's `horizon` query param. Each
# checkpoint tuple picks meaningful sub-horizons for that preset (roughly
# 1 week / 1-3 months out) rather than mechanically quartering n_days.
MONTE_CARLO_HORIZON_PRESETS: dict[str, tuple[int, tuple[int, ...]]] = {
    "1m": (21, (5, 10, 15, 21)),
    "3m": (63, (5, 21, 42, 63)),
    "6m": (126, (21, 42, 84, 126)),
}
DEFAULT_HORIZON = "3m"


@dataclass(frozen=True, slots=True)
class CoreTickerSignals:
    """Everything computable from an OHLCV frame + a benchmark close series +
    a (possibly unknown) RS Rating - the same fields `TickerAnalysis` carries,
    minus identity/content fields (name, sector, chart series, news, fundamentals,
    holders, seasonality, historical analogs) that need extra network calls or
    per-bar chart data an at-a-glance signal has no use for."""

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
    trend: ta.TrendState
    stage: ta.Stage | None
    ma_cross: str | None
    imminent_cross: ta.ImminentCross | None  # a projected, not-yet-happened cross - see technical_analysis.py
    mansfield_rs: float | None
    rs_rating: int | None
    minervini_score: int
    minervini_pass: bool
    support_resistance: list[ta.PriceLevel]
    nearest_support: ta.PriceLevel | None
    nearest_resistance: ta.PriceLevel | None
    obv_divergence: str | None
    statistical_structure: StatisticalStructure | None
    market_trend: ta.TrendState | None  # informational only - see recommendation_engine.py docstring
    vix_regime: str | None  # informational only - see recommendation_engine.py docstring
    is_intraday_snapshot: bool
    recommendation: Recommendation
    entry_timing: EntryTiming | None  # see entry_timing.py - a timing read, not a second verdict
    markov: MarkovChainResult | None
    garch: GarchResult | None
    monte_carlo: MonteCarloResult | None
    backtest: WalkForwardBacktestResult | None
    position_sizing: KellyResult | None


def _last(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _safe_at(series: pd.Series | None, ts: pd.Timestamp) -> float | None:
    if series is None or ts not in series.index:
        return None
    value = series.loc[ts]
    return None if pd.isna(value) else float(value)


def _nearest_level(levels: list[ta.PriceLevel], kind: str) -> ta.PriceLevel | None:
    candidates = [lv for lv in levels if lv.kind == kind]
    return min(candidates, key=lambda lv: abs(lv.distance_pct)) if candidates else None


def compute_core_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    benchmark_close: pd.Series | None,
    rs_rating: int | None,
    horizon: str = DEFAULT_HORIZON,
    revenue_growth: float | None = None,
    profit_margins: float | None = None,
    debt_to_equity: float | None = None,
    vix_close: pd.Series | None = None,
) -> CoreTickerSignals | None:
    if len(close) < MIN_BARS_REQUIRED:
        return None

    mc_days, mc_checkpoints = MONTE_CARLO_HORIZON_PRESETS.get(
        horizon, MONTE_CARLO_HORIZON_PRESETS[DEFAULT_HORIZON]
    )
    price = float(close.iloc[-1])
    # yfinance includes today's bar as soon as the session opens, with a
    # "close" that's really just the latest traded price, not a confirmed
    # settlement - every indicator/verdict reading here is real-time, not
    # repainting after the fact, but this flag lets the caller disclose that
    # today's numbers can still move before the actual close (an external
    # audit's "no evalúes sobre velas no cerradas" concern, addressed as
    # transparency rather than by discarding same-day data - a swing trader
    # checking mid-session wants the live read, not yesterday's stale one).
    last_bar_date = close.index[-1]
    is_intraday_snapshot = bool(
        hasattr(last_bar_date, "date") and last_bar_date.date() == date.today()
    )

    sma20_s = ta.sma(close, 20)
    sma50_s = ta.sma(close, 50)
    sma150_s = ta.sma(close, 150)
    sma200_s = ta.sma(close, 200)
    rsi_s = ta.rsi(close)
    macd_line_s, macd_signal_s, macd_hist_s = ta.macd(close)
    adx_s = ta.adx(high, low, close)
    plus_di_s, minus_di_s = ta.dmi(high, low, close)
    atr_s = ta.atr(high, low, close)

    returns = close.pct_change()
    garch = fit_garch(returns)
    markov = analyze_markov_chain(returns)
    # horizon_days matches the same 1m/3m/6m horizon already selected for the
    # Monte Carlo simulation (mc_days), not the module's own 21-day default.
    # scripts/factor_ablation_study.py (2026-08, ~217 tickers x 10y) found
    # trend/stage/momentum factors show short-term *mean reversion* at 21
    # trading days - the reversal zone documented since Jegadeesh (1990) - and
    # only become directionally consistent with their intended
    # trend-following read at 63/126 days, matching the classic
    # Jegadeesh-Titman (1993) 3-12 month momentum window. Backtesting this
    # system's trend-following verdicts at a horizon shorter than its own
    # design intent would understate (or invert) its real edge.
    backtest = run_walk_forward_backtest(
        close,
        sma20_s,
        sma50_s,
        sma150_s,
        sma200_s,
        rsi_s,
        adx_s,
        plus_di_s,
        minus_di_s,
        atr_s,
        horizon_days=mc_days,
        volume=volume,
    )

    sma20, sma50, sma150, sma200 = _last(sma20_s), _last(sma50_s), _last(sma150_s), _last(sma200_s)
    trend = ta.classify_trend(price, sma20, sma50, sma200)

    stage = None
    ma_cross = None
    imminent_cross = None
    if len(close) >= 200:
        stage = ta.classify_stage(price, sma150_s)
        ma_cross = ta.detect_recent_cross(sma50_s, sma200_s, lookback=5)
        imminent_cross = ta.detect_imminent_cross(sma50_s, sma200_s)

    atr14 = _last(atr_s)
    atr_multiple = ta.atr_multiple_from_sma(close, high, low)
    mansfield = _last(ta.mansfield_rs(close, benchmark_close)) if benchmark_close is not None else None

    sma200_trending_up = ta.sma_slope_positive(sma200_s)
    price_52w_low = ta.rolling_extreme_price(close, 252, "low")
    price_52w_high = ta.rolling_extreme_price(close, 252, "high")
    criteria = ta.minervini_checklist(
        price, sma50, sma150, sma200, sma200_trending_up, price_52w_low, price_52w_high, rs_rating
    )
    minervini_score = sum(criteria.values())
    minervini_pass = all(criteria.values())
    # Scored independently in the recommendation engine - see its comment on
    # `minervini_range_confirmed` for why this specific pair of criteria is
    # pulled out of the 8/8 AND-gate rather than only counted as part of it.
    minervini_range_confirmed = (
        criteria["price_25pct_above_52w_low"] and criteria["price_within_25pct_of_52w_high"]
    )

    levels = ta.support_resistance_levels(high, low, close)
    nearest_support = _nearest_level(levels, "support")
    nearest_resistance = _nearest_level(levels, "resistance")

    # Free (same OHLCV frame every caller already has) - computed for everyone,
    # unlike fundamentals below. See module docstring.
    obv_div = ta.obv_divergence(close, volume)
    # Informational only (see recommendation_engine.py's docstring for why
    # this isn't scored): the benchmark/VIX regime at the moment of analysis,
    # surfaced for context but not fed into the verdict.
    market_trend, vix_regime_label = ta.market_regime_inputs(benchmark_close, vix_close)
    structure = compute_statistical_structure(close)
    mean_reverting_structure = structure.regime == stats_structure.REGIME_MEAN_REVERTING

    recommendation = build_recommendation(
        price=price,
        trend=trend,
        stage=stage,
        ma_cross=ma_cross,
        rsi14=_last(rsi_s),
        adx14=_last(adx_s),
        plus_di=_last(plus_di_s),
        minus_di=_last(minus_di_s),
        atr14=atr14,
        atr_multiple=atr_multiple,
        rs_rating=rs_rating,
        minervini_pass=minervini_pass,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        minervini_range_confirmed=minervini_range_confirmed,
        markov=markov,
        garch=garch,
        obv_divergence=obv_div,
        revenue_growth=revenue_growth,
        profit_margins=profit_margins,
        debt_to_equity=debt_to_equity,
        mean_reverting_structure=mean_reverting_structure,
    )

    entry_timing = assess_entry_timing(atr_multiple, nearest_support, trend)

    monte_carlo = simulate_and_analyze(
        returns,
        price,
        garch,
        stop_loss=recommendation.stop_loss,
        take_profit=recommendation.take_profit,
        n_days=mc_days,
        checkpoints=mc_checkpoints,
    )

    position_sizing = None
    has_trade_setup = recommendation.verdict == "comprar" and recommendation.risk_reward is not None
    if has_trade_setup and monte_carlo is not None:
        win_prob = None
        if monte_carlo.probability_target_before_stop is not None:
            win_prob = win_probability_from_barriers(
                monte_carlo.probability_target_before_stop, monte_carlo.probability_stop_before_target
            )
        if win_prob is not None:
            position_sizing = recommend_position_size(
                win_probability=win_prob,
                reward_risk_ratio=recommendation.risk_reward,
                vol_regime=garch.regime if garch is not None else None,
            )

    return CoreTickerSignals(
        price=price,
        change_1d=ta.pct_change_over(close, 1),
        change_1w=ta.pct_change_over(close, 5),
        change_1m=ta.pct_change_over(close, 21),
        change_3m=ta.pct_change_over(close, 63),
        change_6m=ta.pct_change_over(close, 126),
        change_1y=ta.pct_change_over(close, 252),
        volume=float(volume.iloc[-1]),
        relative_volume=ta.relative_volume(volume),
        rsi14=_last(rsi_s),
        macd_line=_last(macd_line_s),
        macd_signal=_last(macd_signal_s),
        macd_histogram=_last(macd_hist_s),
        adx14=_last(adx_s),
        plus_di=_last(plus_di_s),
        minus_di=_last(minus_di_s),
        atr14=atr14,
        atr_multiple=atr_multiple,
        sma20=sma20,
        sma50=sma50,
        sma150=sma150,
        sma200=sma200,
        dist_52w_high=ta.distance_to_rolling_extreme(close, 252, "high"),
        dist_52w_low=ta.distance_to_rolling_extreme(close, 252, "low"),
        trend=trend,
        stage=stage,
        ma_cross=ma_cross,
        imminent_cross=imminent_cross,
        mansfield_rs=mansfield,
        rs_rating=rs_rating,
        minervini_score=minervini_score,
        minervini_pass=minervini_pass,
        support_resistance=levels,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        obv_divergence=obv_div,
        statistical_structure=structure,
        market_trend=market_trend,
        vix_regime=vix_regime_label,
        is_intraday_snapshot=is_intraday_snapshot,
        recommendation=recommendation,
        entry_timing=entry_timing,
        markov=markov,
        garch=garch,
        monte_carlo=monte_carlo,
        backtest=backtest,
        position_sizing=position_sizing,
    )


class TickerAnalysisService:
    def __init__(self, market_data: MarketDataService, screener: MarketScreenerService | None = None) -> None:
        self.market_data = market_data
        self.screener = screener

    def _rs_rating_for(self, ticker: str) -> int | None:
        if self.screener is None:
            return None
        # A ticker searched directly could be in either curated universe (or
        # neither, e.g. a ticker outside both - then RS Rating is simply None).
        for region in ("us", "europe"):
            snapshot = next((s for s in self.screener.get_universe_snapshot(region) if s.ticker == ticker), None)
            if snapshot is not None:
                return snapshot.rs_rating
        return None

    def analyze(self, ticker: str, horizon: str = DEFAULT_HORIZON) -> TickerAnalysis:
        ticker = ticker.upper()
        end = date.today()
        start = end - timedelta(days=365 * HISTORY_YEARS)

        # Benchmarked against the S&P 500 or STOXX Europe 600 depending on which
        # market the ticker actually trades in - comparing a European stock's
        # relative strength to the wrong benchmark would misread it entirely.
        # VIX rides along in the same batched call (free - yfinance downloads
        # all three in one request) so historical_analogs() can match on the
        # market's fear/volatility regime at each candidate point, not just
        # this ticker's own price shape.
        benchmark_ticker = benchmark_for_ticker(ticker)
        ohlcv = self.market_data.get_bulk_ohlcv([ticker, benchmark_ticker, VIX_TICKER], start, end)
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_BARS_REQUIRED:
            raise ValueError(f"No hay suficientes datos de precio para {ticker}")

        benchmark_df = ohlcv.get(benchmark_ticker)
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
        vix_df = ohlcv.get(VIX_TICKER)
        vix_close = vix_df["close"] if vix_df is not None else None

        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        rs_rating = self._rs_rating_for(ticker)
        # Fetched here (not after, as it used to be) so the fundamentals factor
        # can actually feed into the recommendation - this is the one path that
        # supplies it, see compute_core_signals()'s docstring for why the other
        # two callers (portfolio risk, premium watchlist) deliberately don't.
        info = self.market_data.get_ticker_info(ticker)
        core = compute_core_signals(
            close,
            high,
            low,
            volume,
            benchmark_close,
            rs_rating,
            horizon,
            revenue_growth=info.revenue_growth if info else None,
            profit_margins=info.profit_margins if info else None,
            debt_to_equity=info.debt_to_equity if info else None,
            vix_close=vix_close,
        )
        if core is None:
            raise ValueError(f"No hay suficientes datos de precio para {ticker}")

        # Chart-only series: analyze() needs the full per-bar history for the price
        # chart/RSI-MACD panel, which compute_core_signals() doesn't expose (it only
        # returns final scalar values). Recomputing these is cheap (vectorized pandas,
        # not the GARCH/backtest/Monte Carlo work compute_core_signals already did once).
        sma20_s, sma50_s = ta.sma(close, 20), ta.sma(close, 50)
        sma150_s, sma200_s = ta.sma(close, 150), ta.sma(close, 200)
        rsi_s = ta.rsi(close)
        _, _, macd_hist_s = ta.macd(close)
        bb_mid_s, bb_up_s, bb_low_s = ta.bollinger_bands(close)
        atr_s = ta.atr(high, low, close)

        gann_lines = at.gann_fan(high, low, core.trend, atr_s)
        gann_by_label: dict[str, pd.Series] = {}
        if gann_lines:
            for line in gann_lines:
                gann_by_label[line.label] = pd.Series(line.values, index=close.index)

        chart_slice = df.iloc[-CHART_BARS:] if len(df) > CHART_BARS else df
        price_history = [
            PricePoint(
                date=ts.date(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                sma20=_safe_at(sma20_s, ts),
                sma50=_safe_at(sma50_s, ts),
                sma150=_safe_at(sma150_s, ts),
                sma200=_safe_at(sma200_s, ts),
                bb_upper=_safe_at(bb_up_s, ts),
                bb_middle=_safe_at(bb_mid_s, ts),
                bb_lower=_safe_at(bb_low_s, ts),
                gann_1x1=_safe_at(gann_by_label.get("1x1"), ts),
                gann_1x2=_safe_at(gann_by_label.get("1x2"), ts),
                gann_2x1=_safe_at(gann_by_label.get("2x1"), ts),
                rsi14=_safe_at(rsi_s, ts),
                macd_histogram=_safe_at(macd_hist_s, ts),
            )
            for ts, row in chart_slice.iterrows()
        ]

        news = self.market_data.get_ticker_news(ticker)
        holders = self.market_data.get_holders(ticker)
        seasonality = at.seasonality_by_month(close)
        historical_analogs = at.historical_analogs(close, vix_close=vix_close)

        return TickerAnalysis(
            ticker=ticker,
            name=info.name if info else None,
            sector=info.sector if info else None,
            industry=info.industry if info else None,
            currency=info.currency if info else None,
            market_cap=info.market_cap if info else None,
            price=core.price,
            change_1d=core.change_1d,
            change_1w=core.change_1w,
            change_1m=core.change_1m,
            change_3m=core.change_3m,
            change_6m=core.change_6m,
            change_1y=core.change_1y,
            volume=core.volume,
            relative_volume=core.relative_volume,
            rsi14=core.rsi14,
            macd_line=core.macd_line,
            macd_signal=core.macd_signal,
            macd_histogram=core.macd_histogram,
            adx14=core.adx14,
            plus_di=core.plus_di,
            minus_di=core.minus_di,
            atr14=core.atr14,
            atr_multiple=core.atr_multiple,
            sma20=core.sma20,
            sma50=core.sma50,
            sma150=core.sma150,
            sma200=core.sma200,
            dist_52w_high=core.dist_52w_high,
            dist_52w_low=core.dist_52w_low,
            trend=core.trend,
            stage=core.stage,
            ma_cross=core.ma_cross,
            imminent_cross=core.imminent_cross,
            mansfield_rs=core.mansfield_rs,
            rs_rating=core.rs_rating,
            minervini_score=core.minervini_score,
            minervini_pass=core.minervini_pass,
            support_resistance=core.support_resistance,
            obv_divergence=core.obv_divergence,
            statistical_structure=core.statistical_structure,
            market_trend=core.market_trend,
            vix_regime=core.vix_regime,
            is_intraday_snapshot=core.is_intraday_snapshot,
            price_history=price_history,
            news=news,
            fundamentals=info,
            holders=holders,
            seasonality=seasonality,
            historical_analogs=historical_analogs,
            recommendation=core.recommendation,
            entry_timing=core.entry_timing,
            markov=core.markov,
            garch=core.garch,
            monte_carlo=core.monte_carlo,
            backtest=core.backtest,
            position_sizing=core.position_sizing,
        )
