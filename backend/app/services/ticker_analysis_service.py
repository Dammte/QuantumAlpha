"""Orchestrates the single-ticker "deep dive": every indicator this app knows how
to compute, a Gann fan, seasonality, historical analogs, news, and the
levels/triggers gate's pass/fail read, all for one ticker on demand.

Unlike the market screener (which scans ~170 tickers and has to stay fast and
cheap per-ticker), this runs once per user search, so it can afford a decade of
history and a couple of slower per-ticker calls (fundamentals, news).

`compute_core_signals()` holds the quant core of that deep dive (the gate plus
the triple-barrier backtest) as a function of a plain OHLCV frame, with none of
the extra per-ticker network calls (fundamentals/news) or chart-only series.
It exists so the portfolio-position risk check can run the *exact same*
analysis "Analizar activo" would - not a cheaper approximation of it - so a
holding is never flagged "sell" on a different, laxer basis than what you'd
see by searching it directly.

2026-09 (reconstruction, Fase 4): the live verdict here is now
`levels_engine.evaluate_gate` (`GateResult` - `gate`/`confirmed_gate` below),
not the old weighted checklist. That checklist (`build_recommendation`,
`Recommendation`, `RecommendationFactor`) has now been retired outright from
`recommendation_engine.py` (Parte 2.3/6 of the reconstruction) - the module
is trimmed to the `trade_geometry.py` re-exports and `ENGINE_VERSION` only.
`scripts/factor_ablation_study.py` never actually imported the checklist
function; it mirrors the old point values as its own literal constants (see
that script's own docstring) precisely so retiring this one doesn't touch
it. See docs/quant_methodology.md.

2026-09: Markov chain, GARCH, Monte Carlo, Kelly sizing, the Hurst/ADF
statistical-structure read and the entry-timing badge were removed from this
pipeline (see `docs/quant_methodology.md` and `recommendation_engine.py`'s
module docstring) - none of them had cross-sectional evidence of predicting
anything at this portfolio's actual holding horizon, and Monte Carlo's own
"probability of hitting the stop/target" was circular (it simulated around
the exact stop/target the recommendation had already chosen). GARCH's one
remaining real use - bucketing the Chandelier Exit's volatility multiplier -
now comes from `technical_analysis.volatility_regime_from_atr_percentile`
instead of a per-ticker model fit.

2026-09 (reconstruction, Fase 1): the recommendation engine's fundamentals
factor (revenue growth, profit margin, leverage) was retired too - never
measured by `scripts/factor_ablation_study.py`, see
`recommendation_engine.py`'s module docstring - so `compute_core_signals()`
no longer accepts or forwards those inputs at all. `TickerAnalysisService.
analyze()` still fetches `TickerInfo` (shown as plain informational context
in the deep-dive's Fundamentals tab), it just no longer feeds anything into
the score. Institutional/insider holders data and analyst-consensus fields
were retired the same round, for lack of any decision this app actually
makes with them (see `docs/quant_methodology.md`).
"""

from dataclasses import dataclass
from datetime import date, time, timedelta

import pandas as pd

from app.domain.interfaces.llm_narrator import LLMNarrator
from app.domain.models.ticker_analysis import PricePoint, TickerAnalysis
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.backtest_engine import (
    VERTICAL_BARRIER_HORIZONS,
    TripleBarrierBacktestResult,
    run_triple_barrier_backtest,
)
from app.services.dynamic_universe_service import passes_liquidity_floor
from app.services.levels_engine import GateResult, GradeResult, compute_grade, evaluate_gate
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import VIX_TICKER, benchmark_for_ticker, closed_bar_cutoff_for_ticker
from app.services.trade_geometry import EntryTrigger, StopAndTarget

HISTORY_YEARS = 10
CHART_BARS = 504  # ~2 trading years
MIN_BARS_REQUIRED = 250  # Parte 3.2: por debajo, no hay SMA200 ni rango anual fiable - "datos insuficientes"

# 2026-09: used to also key a Monte Carlo horizon preset (1m/3m/6m -> a day
# count + a set of forecast checkpoints); Monte Carlo is gone (see module
# docstring), so `horizon` now only labels the persisted
# RecommendationSnapshotORM row - kept as a parameter through
# TickerAnalysisService.analyze()/compute_core_signals() for that reason and
# for API-contract stability, but it no longer changes what gets computed.
DEFAULT_HORIZON = "3m"

# Segunda auditoría, Bloque 2: fixed at this portfolio's actual holding
# horizon (backtest_engine.py's own validated range), never the Monte Carlo
# preset the caller picked (1m/3m/6m -> 21/63/126 days) - 63/126 is outside
# what run_triple_barrier_backtest was designed and documented against.
TRIPLE_BARRIER_HORIZON_DAYS = VERTICAL_BARRIER_HORIZONS[1]  # 21

# Segunda auditoría, Bloque 4: same reasoning as TRIPLE_BARRIER_HORIZON_DAYS -
# this portfolio's actual holding horizon, fixed, never whichever Monte Carlo
# preset the caller happened to pick.
SIGN_CHECK_HORIZON_DAYS = VERTICAL_BARRIER_HORIZONS[1]  # 21


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
    imminent_cross: ta.ImminentCross | None  # SMA50/SMA200, projected - see technical_analysis.py
    imminent_cross_short_term: ta.ImminentCross | None  # SMA20/SMA50, for a shorter-horizon-managed position
    candlestick_pattern: str | None  # "bullish_engulfing" | "bearish_engulfing" - see detect_engulfing_pattern
    mansfield_rs: float | None
    rs_rating: int | None
    minervini_score: int
    minervini_pass: bool
    support_resistance: list[ta.PriceLevel]
    # Parte 5.1 (later pass): the richer level engine (`technical_analysis.detect_levels`)
    # alongside `support_resistance` above, not replacing it - 14 existing
    # consumers still read the simpler `PriceLevel`/`nearest_support`/
    # `nearest_resistance` shape below unchanged. `distance_atr`/`state`
    # (FAR/APPROACHING/TESTING/BREAKING/BROKEN_CONFIRMED/LOST_CONFIRMED) are
    # what this adds that the older shape can't express.
    levels: list[ta.Level]
    nearest_support: ta.PriceLevel | None
    nearest_resistance: ta.PriceLevel | None
    obv_divergence: str | None
    market_trend: ta.TrendState | None  # informational only - see recommendation_engine.py docstring
    vix_regime: str | None  # informational only - see recommendation_engine.py docstring
    is_intraday_snapshot: bool
    gate: GateResult
    # Parte 5.3: geometría, no probabilidad - `None` cuando `gate.entry_trigger`
    # o `gate.entry_geometry` no existen, o la geometría no es viable (no hay
    # ningún disparador real que gradar). Sin los modificadores de cartera
    # (correlación/tope de posiciones, Parte 5.3) - ver
    # `levels_engine.apply_portfolio_grade_modifiers`, que necesita una
    # cartera específica y vive un nivel por encima de esta lectura, igual
    # que `trade_geometry.size_position`.
    grade: GradeResult | None
    # Segunda auditoría, Bloque 2: `gate` above (and every field on this
    # dataclass) is computed on the raw, possibly still-forming last bar -
    # live, real-time, but not repaint-proof (see D6/`closed_bars`'
    # docstring). `multi_timeframe` gives the weekly/daily read off *closed*
    # bars only, reusing the exact same machinery `portfolio_risk_service.py`
    # already runs for open positions - before this, "Analizar activo" never
    # referenced `analyze_multi_timeframe`/`closed_bars` at all, so a weekly
    # bearish crossover already visible on higher timeframes was invisible
    # here regardless of what the daily-only read said.
    multi_timeframe: mtf.MultiTimeframeRead
    # The same pass/fail and stop/target `gate` carries, recomputed with
    # every discrete technical input (trend, stage, support/resistance)
    # re-derived from `technical_analysis.closed_bars` instead of the live
    # frame - `None` when there's nothing to separate from (the last bar is
    # already settled, so `gate` itself is already the confirmed read; see
    # `is_intraday_snapshot`). Deliberately reuses the already-computed
    # obv_divergence read rather than refitting it on one bar less of
    # history - that's a continuous read, not a discrete signal that
    # repaints the way a moving-average cross does.
    confirmed_gate: GateResult | None
    # This is the honest backtest: triple-barrier labeling, real Chandelier
    # trailing, costs net, at this portfolio's actual holding horizon - see
    # `backtest_engine.py`'s own module docstring for the full reasoning,
    # including why the older, naive fixed-horizon buy-and-hold replay it
    # replaced (`walk_forward_backtest.py`) was retired outright rather than
    # kept as a second field (2026-09). `None` unless the caller opted into
    # `include_triple_barrier_backtest` - see `compute_core_signals`'s own
    # docstring for why this one, unlike every other field here, isn't
    # computed unconditionally.
    triple_barrier_backtest: TripleBarrierBacktestResult | None


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


def _confirmed_gate(
    daily_df: pd.DataFrame,
    weekly_stage: ta.Stage | None,
    liquidity_ok: bool,
    next_earnings_date: date | None,
    cutoff: time | None = None,
) -> GateResult | None:
    """Re-derives the gate from `technical_analysis.closed_bars` instead of
    the live frame - the same discrete-technical-inputs mirror
    `compute_core_signals` builds for the live read, just bound to settled
    bars. `None` when there aren't enough closed bars left to say anything
    (a data-thin ticker whose last closed bar is also its only usable one).
    `cutoff` - see `market_universe.closed_bar_cutoff_for_ticker`.

    `weekly_stage`/`liquidity_ok`/`next_earnings_date` are reused from the
    live read's own already-computed values, not rederived here: the weekly
    Stage already comes from *closed* weekly bars regardless of whether
    today's daily bar has settled (see `multi_timeframe.py`), and liquidity/
    earnings don't depend on whether today's bar is confirmed either - only
    `trend`/`atr14`/the nearest levels/the fast-pair veto genuinely differ
    between "as of the live price" and "as of the last confirmed close",
    which is the whole reason this function exists.

    2026-09 (reconstruction, Fase 4): no longer takes `rs_rating` - RS Rating
    was never wired into the gate as a hard condition (see
    `levels_engine.py`'s own docstring), so there's nothing here that needs
    it; the Minervini-checklist recomputation this used to do purely to feed
    `build_recommendation` is gone for the same reason - `evaluate_gate`
    doesn't take `minervini_pass`/`ma_cross` either."""
    closed = ta.closed_bars(daily_df, cutoff=cutoff)
    if len(closed) < MIN_BARS_REQUIRED:
        return None
    close, high, low = closed["close"], closed["high"], closed["low"]
    price = float(close.iloc[-1])

    sma20_s, sma50_s = ta.sma(close, 20), ta.sma(close, 50)
    sma200_s = ta.sma(close, 200)
    sma20, sma50, sma200 = _last(sma20_s), _last(sma50_s), _last(sma200_s)
    trend = ta.classify_trend(price, sma20, sma50, sma200)

    atr_s = ta.atr(high, low, close)
    levels = ta.support_resistance_levels(high, low, close)

    return evaluate_gate(
        price=price,
        trend=trend,
        atr14=_last(atr_s),
        nearest_support=_nearest_level(levels, "support"),
        nearest_resistance=_nearest_level(levels, "resistance"),
        weekly_stage=weekly_stage,
        liquidity_ok=liquidity_ok,
        fast_pair_bearish_signal=ta.detect_fast_pair_bearish_veto(close),
        next_earnings_date=next_earnings_date,
    )


def compute_core_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    open_: pd.Series,
    benchmark_close: pd.Series | None,
    rs_rating: int | None,
    horizon: str = DEFAULT_HORIZON,
    vix_close: pd.Series | None = None,
    ticker: str | None = None,
    include_triple_barrier_backtest: bool = False,
    next_earnings_date: date | None = None,
    sector_rs_percentile: int | None = None,
) -> CoreTickerSignals | None:
    """`ticker`, when given, picks a region-aware settlement cutoff for
    `multi_timeframe`/`confirmed_gate` (`market_universe.closed_bar_cutoff_for_ticker`
    - Segunda auditoría, Bloque 2) instead of the US-centric default. Optional
    (not every caller has traced a ticker string this far down, and every
    other field here is computable without one) - `None` just means "assume
    US settlement hours".

    `sector_rs_percentile` (Parte 2.5/5.3) - percentil de fuerza relativa del
    sector propio, ya calculado por `market_screener_service._sector_rs_percentiles`
    para el universo; no recalculado aquí (esta función no tiene, ni necesita,
    los ETFs de sector). Solo alimenta el grado A/B/C - `None` simplemente
    omite ese modificador concreto, nunca bloquea el resto.

    `next_earnings_date` (Parte 6.2's `no_event_risk` gate criterion) is
    optional and defaults to `None` (interpreted by `evaluate_gate` as "no
    known upcoming event", not "couldn't check" - see that function's own
    docstring) - not every caller has already paid for a per-ticker
    `get_next_earnings_date` network call (`TickerAnalysisService.analyze()`
    has; the universe-wide screener path deliberately hasn't, per-ticker
    network calls there being exactly what CLAUDE.md forbids in a hot path).

    `include_triple_barrier_backtest` defaults to `False`: measured at
    ~3x this function's own cost (a bar-by-bar Python simulation over the
    full grid, twice - fixed and trailing - plus the random-entry benchmark,
    unlike every other computation here, which is vectorized pandas). The
    brief's own ask for this (Segunda auditoría, Bloque 2) was specifically
    "Analizar activo" - `TickerAnalysisService.analyze()` is the only caller
    that opts in; `portfolio_risk_service` runs this same function per held
    position on every cache refresh, where that 3x would compound across
    the whole portfolio for a field that view doesn't show."""
    if len(close) < MIN_BARS_REQUIRED:
        return None
    closed_bar_cutoff = closed_bar_cutoff_for_ticker(ticker) if ticker else None

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

    # Reconstructed here (compute_core_signals gets pre-split series, not a
    # combined frame) - identical values/index to whatever `df` the caller
    # actually holds, since these are the exact series it sliced out of it.
    daily_df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    multi_timeframe = mtf.analyze_multi_timeframe(daily_df, cutoff=closed_bar_cutoff)

    sma20_s = ta.sma(close, 20)
    sma50_s = ta.sma(close, 50)
    sma150_s = ta.sma(close, 150)
    sma200_s = ta.sma(close, 200)
    rsi_s = ta.rsi(close)
    macd_line_s, macd_signal_s, macd_hist_s = ta.macd(close)
    adx_s = ta.adx(high, low, close)
    plus_di_s, minus_di_s = ta.dmi(high, low, close)
    atr_s = ta.atr(high, low, close)

    # 2026-09: this used to feed a per-ticker GARCH(1,1) fit, whose only
    # surviving consumer was the Chandelier Exit's volatility-regime bucket -
    # see technical_analysis.volatility_regime_from_atr_percentile.
    vol_regime = ta.volatility_regime_from_atr_percentile(ta.atr_percentile(atr_s / close))
    triple_barrier_backtest = None
    if include_triple_barrier_backtest:
        triple_barrier_backtest = run_triple_barrier_backtest(
            close, high, low, open_, sma20_s, sma50_s, sma150_s, sma200_s, rsi_s, adx_s, plus_di_s, minus_di_s,
            atr_s, horizon_days=TRIPLE_BARRIER_HORIZON_DAYS, volume=volume,
            vol_regime=vol_regime,
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

    # Short-term counterpart of the above (SMA20/SMA50, not SMA50/SMA200) -
    # for a position managed on a shorter horizon, this is the trend guide
    # that actually matters day to day: SMA50/SMA200 can stay bullish for
    # months after a short-term swing has already turned. Only needs 50 bars,
    # not 200, so it's gated separately.
    imminent_cross_short_term = None
    if len(close) >= 50:
        imminent_cross_short_term = ta.detect_imminent_cross(sma20_s, sma50_s)

    candlestick_pattern = ta.detect_engulfing_pattern(open_, close)

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

    levels = ta.support_resistance_levels(high, low, close)
    nearest_support = _nearest_level(levels, "support")
    nearest_resistance = _nearest_level(levels, "resistance")

    # Parte 5.1 (later pass): the real weekly close series (same in-memory
    # resample `multi_timeframe` above already paid for internally, no new
    # network call) lets `detect_levels` build a real WEEKLY_MA30 level
    # instead of omitting it (Parte 5.5's own documented gap, closed here).
    weekly_df_for_levels = ta.resample_ohlcv(daily_df, mtf.WEEKLY_RULE)
    weekly_close_for_levels = weekly_df_for_levels["close"] if len(weekly_df_for_levels) >= 2 else None
    detected_levels = ta.detect_levels(high, low, close, volume, weekly_close=weekly_close_for_levels)

    # Free (same OHLCV frame every caller already has) - computed for everyone,
    # unlike fundamentals below. See module docstring.
    obv_div = ta.obv_divergence(close, volume)
    # Informational only (see recommendation_engine.py's docstring for why
    # this isn't scored): the benchmark/VIX regime at the moment of analysis,
    # surfaced for context but not fed into the verdict.
    market_trend, vix_regime_label = ta.market_regime_inputs(benchmark_close, vix_close)
    fast_pair_veto = ta.detect_fast_pair_bearish_veto(close)
    # Same EMA21/55 pair multi_timeframe.py itself uses (Parte 3.2/6) - feeds
    # the gate's own Parte 7 geometry below, not a third independent read.
    ema21 = _last(ta.ema(close, mtf.FAST_MA_PERIOD))
    ema55 = _last(ta.ema(close, mtf.SLOW_MA_PERIOD))

    # Parte 6.2: `weekly_not_stage4` reads the *weekly* Stage `multi_timeframe`
    # already computed above (real MA30 semanal, Parte 5.5) - never the
    # daily-bar `stage` above (that one stays a plain informational field on
    # `CoreTickerSignals`, shown in the UI, no longer a gate input at all).
    weekly_stage = multi_timeframe.weekly.stage if multi_timeframe.weekly is not None else None
    # Divisa nativa, sin conversión a USD (a diferencia de daily_close.py/
    # market_screener_service.py, que sí manejan fx_rate para el universo
    # completo) - una simplificación deliberada para "Analizar activo": el
    # usuario ya eligió mirar este ticker en concreto, así que el suelo de
    # liquidez importa menos aquí que como filtro de qué aparece en el
    # universo/Radar, donde sí se convierte a USD.
    dollar_volume_20d = float((close.iloc[-20:] * volume.iloc[-20:]).mean()) if len(close) >= 20 else None
    liquidity_ok = passes_liquidity_floor(price, dollar_volume_20d)

    gate = evaluate_gate(
        price=price,
        trend=trend,
        atr14=atr14,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        weekly_stage=weekly_stage,
        liquidity_ok=liquidity_ok,
        fast_pair_bearish_signal=fast_pair_veto,
        next_earnings_date=next_earnings_date,
        ema21=ema21,
        ema55=ema55,
    )

    confirmed_gate = None
    if is_intraday_snapshot:
        confirmed_gate = _confirmed_gate(
            daily_df, weekly_stage, liquidity_ok, next_earnings_date, cutoff=closed_bar_cutoff
        )

    # Parte 5.3: solo tiene sentido gradar un disparador real - una entrada
    # geométricamente inviable, o un gate sin ningún nivel cerca todavía
    # (`entry_trigger is None`), no tiene nada que gradar.
    grade = None
    if gate.entry_trigger is not None and gate.entry_geometry is not None and gate.entry_geometry.viable:
        weekly_bullish = mtf.timeframe_bias(multi_timeframe.weekly) == "bullish"
        grade = compute_grade(
            price=price,
            atr14=atr14,
            entry_trigger=gate.entry_trigger,
            geometry=gate.entry_geometry,
            weekly_bullish=weekly_bullish,
            relative_volume=ta.relative_volume(volume),
            rs_percentile=rs_rating,
            sma200=sma200,
            sector_rs_percentile=sector_rs_percentile,
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
        imminent_cross_short_term=imminent_cross_short_term,
        candlestick_pattern=candlestick_pattern,
        mansfield_rs=mansfield,
        rs_rating=rs_rating,
        minervini_score=minervini_score,
        minervini_pass=minervini_pass,
        support_resistance=levels,
        levels=detected_levels,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        obv_divergence=obv_div,
        market_trend=market_trend,
        vix_regime=vix_regime_label,
        is_intraday_snapshot=is_intraday_snapshot,
        gate=gate,
        grade=grade,
        multi_timeframe=multi_timeframe,
        confirmed_gate=confirmed_gate,
        triple_barrier_backtest=triple_barrier_backtest,
    )


def _entry_trigger_summary(trigger: EntryTrigger | None) -> str | None:
    if trigger is None:
        return None
    kind = "ruptura" if trigger.trigger_type == "breakout" else "rebote en soporte"
    status = "ya disparado" if trigger.already_triggered else "todavía vigilando"
    return f"{kind} en {trigger.trigger_price:.2f} ({status})"


def _stop_and_target_summary(stop_and_target: StopAndTarget | None) -> str | None:
    if stop_and_target is None or stop_and_target.stop_loss is None:
        return None
    parts = [f"stop en {stop_and_target.stop_loss:.2f}"]
    if stop_and_target.take_profit is not None:
        parts.append(f"objetivo en {stop_and_target.take_profit:.2f} ({stop_and_target.take_profit_method})")
    if stop_and_target.risk_reward is not None:
        parts.append(f"relación beneficio:riesgo {stop_and_target.risk_reward:.1f}:1")
    return ", ".join(parts)


class TickerAnalysisService:
    def __init__(
        self,
        market_data: MarketDataService,
        screener: MarketScreenerService | None = None,
        narrator: LLMNarrator | None = None,
    ) -> None:
        self.market_data = market_data
        self.screener = screener
        # Reconstruction (2026-09), Fase 7: `None` by default (no live path
        # instantiates a narrator unless GEMINI_API_KEY is set - see
        # api/deps.py) - `_explain_gate` below already handles that case
        # gracefully, so this class never needs its own extra "is this
        # configured" branch beyond what the port itself guarantees.
        self.narrator = narrator

    def _explain_gate(
        self, ticker: str, gate: GateResult, trend: ta.TrendState, stage: ta.Stage | None
    ) -> str | None:
        if self.narrator is None:
            return None
        return self.narrator.explain_gate(
            ticker=ticker,
            gate_passes=gate.passes,
            conditions=[(c.label, c.passed) for c in gate.conditions],
            trend_label=trend.value,
            stage_label=stage.value if stage else None,
            entry_trigger_summary=_entry_trigger_summary(gate.entry_trigger),
            stop_and_target_summary=_stop_and_target_summary(gate.stop_and_target),
        )

    def _universe_snapshot_for(self, ticker: str) -> TickerSnapshot | None:
        if self.screener is None:
            return None
        # A ticker searched directly could be in either curated universe (or
        # neither, e.g. a ticker outside both).
        for region in ("us", "europe"):
            snapshot = next((s for s in self.screener.get_universe_snapshot(region) if s.ticker == ticker), None)
            if snapshot is not None:
                return snapshot
        return None

    def analyze(self, ticker: str, horizon: str = DEFAULT_HORIZON) -> TickerAnalysis:
        ticker = ticker.upper()
        end = date.today()
        start = end - timedelta(days=365 * HISTORY_YEARS)

        # Benchmarked against the S&P 500 or STOXX Europe 600 depending on which
        # market the ticker actually trades in - comparing a European stock's
        # relative strength to the wrong benchmark would misread it entirely.
        # VIX rides along in the same batched call (free - yfinance downloads
        # all three in one request).
        benchmark_ticker = benchmark_for_ticker(ticker)
        ohlcv = self.market_data.get_bulk_ohlcv([ticker, benchmark_ticker, VIX_TICKER], start, end)
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_BARS_REQUIRED:
            raise ValueError(f"No hay suficientes datos de precio para {ticker}")

        benchmark_df = ohlcv.get(benchmark_ticker)
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
        vix_df = ohlcv.get(VIX_TICKER)
        vix_close = vix_df["close"] if vix_df is not None else None

        close, high, low, volume, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
        universe_snapshot = self._universe_snapshot_for(ticker)
        rs_rating = universe_snapshot.rs_rating if universe_snapshot is not None else None
        sector_rs_percentile = universe_snapshot.sector_rs_percentile if universe_snapshot is not None else None
        # Used for the Fundamentals tab's plain informational display
        # (name/sector/industry/market cap/growth/margin/leverage) - no
        # longer feeds the recommendation score itself (see
        # recommendation_engine.py's module docstring).
        info = self.market_data.get_ticker_info(ticker)
        # Tercera auditoría, Bloque F-9: a breakout 3 days before earnings
        # isn't the same trade as one with no event risk in the holding
        # window - only fetched here (a single-ticker deep-dive already
        # paying for get_ticker_info), never for the whole universe screener.
        next_earnings = self.market_data.get_next_earnings_date(ticker)
        days_to_earnings = (next_earnings - date.today()).days if next_earnings else None
        core = compute_core_signals(
            close,
            high,
            low,
            volume,
            open_,
            benchmark_close,
            rs_rating,
            horizon,
            vix_close=vix_close,
            ticker=ticker,
            include_triple_barrier_backtest=True,
            next_earnings_date=next_earnings,
            sector_rs_percentile=sector_rs_percentile,
        )
        if core is None:
            raise ValueError(f"No hay suficientes datos de precio para {ticker}")

        # Chart-only series: analyze() needs the full per-bar history for the price
        # chart/RSI-MACD panel, which compute_core_signals() doesn't expose (it only
        # returns final scalar values). Recomputing these is cheap (vectorized pandas,
        # not the triple-barrier backtest work compute_core_signals already did once).
        sma20_s, sma50_s = ta.sma(close, 20), ta.sma(close, 50)
        sma150_s, sma200_s = ta.sma(close, 150), ta.sma(close, 200)
        # Parte 3.2/11: the real fast pair the gate/exit engine decide
        # against, drawn on the chart too - same mtf.FAST_MA_PERIOD/
        # SLOW_MA_PERIOD basis multi_timeframe.py itself uses.
        ema21_s = ta.ema(close, mtf.FAST_MA_PERIOD)
        ema55_s = ta.ema(close, mtf.SLOW_MA_PERIOD)
        rsi_s = ta.rsi(close)
        _, _, macd_hist_s = ta.macd(close)
        bb_mid_s, bb_up_s, bb_low_s = ta.bollinger_bands(close)

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
                ema21=_safe_at(ema21_s, ts),
                ema55=_safe_at(ema55_s, ts),
                bb_upper=_safe_at(bb_up_s, ts),
                bb_middle=_safe_at(bb_mid_s, ts),
                bb_lower=_safe_at(bb_low_s, ts),
                rsi14=_safe_at(rsi_s, ts),
                macd_histogram=_safe_at(macd_hist_s, ts),
            )
            for ts, row in chart_slice.iterrows()
        ]

        news = self.market_data.get_ticker_news(ticker)
        llm_narrative = self._explain_gate(ticker, core.gate, core.trend, core.stage)

        return TickerAnalysis(
            ticker=ticker,
            name=info.name if info else None,
            sector=info.sector if info else None,
            industry=info.industry if info else None,
            currency=info.currency if info else None,
            market_cap=info.market_cap if info else None,
            days_to_earnings=days_to_earnings,
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
            imminent_cross_short_term=core.imminent_cross_short_term,
            candlestick_pattern=core.candlestick_pattern,
            mansfield_rs=core.mansfield_rs,
            rs_rating=core.rs_rating,
            minervini_score=core.minervini_score,
            minervini_pass=core.minervini_pass,
            support_resistance=core.support_resistance,
            levels=core.levels,
            obv_divergence=core.obv_divergence,
            market_trend=core.market_trend,
            vix_regime=core.vix_regime,
            is_intraday_snapshot=core.is_intraday_snapshot,
            multi_timeframe=core.multi_timeframe,
            confirmed_gate=core.confirmed_gate,
            price_history=price_history,
            news=news,
            fundamentals=info,
            gate=core.gate,
            grade=core.grade,
            triple_barrier_backtest=core.triple_barrier_backtest,
            llm_narrative=llm_narrative,
        )
