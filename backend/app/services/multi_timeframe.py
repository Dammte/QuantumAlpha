"""Top-down, multi-timeframe read: the higher timeframe sets the directional
bias, the lower timeframe times entries/exits - never averaged together. That
hierarchy is the whole point (see `analyze_multi_timeframe`'s docstring for
the alignment rules); a weekly downtrend with a daily bounce inside it is a
bounce, not a trend change, and averaging the two into one blended number
would erase exactly that distinction.

Built entirely on `technical_analysis.py`'s pure functions plus
`resample_ohlcv`/`closed_bars` - no network call: weekly/monthly bars are
derived from the same daily OHLCV frame every caller (ticker analysis,
portfolio risk, the exit engine) already has in memory. This is the fix for
what the project's own audit calls D1: before this module, the whole system
only ever saw daily bars, so "the weekly chart already shows a bearish
crossover" - the exact complaint that motivated this work - was something the
system could not see, by construction, no matter how the daily-bar factors
were tuned.
"""

from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd

from app.services import technical_analysis as ta

WEEKLY_RULE = "W-FRI"

# Weinstein's own stage analysis is defined on a 30-week moving average on
# weekly bars. `technical_analysis.classify_stage`'s default (150-day MA) is
# documented there as a *daily-bar proxy* for that 30-week MA - useful when
# weekly bars aren't available, but on genuine weekly bars (which this module
# now derives via `resample_ohlcv`) the real 30-week parameter should be used
# directly instead of re-applying the daily proxy to weekly data.
WEEKLY_STAGE_MA_WINDOW = 30
DAILY_STAGE_MA_WINDOW = 150
# Round gates below which a stage/cross read on that many bars would mostly
# be reporting classify_stage's own no-data fallback rather than a real
# signal - not a strict data requirement of the functions themselves (they
# degrade to None/Stage.STAGE_1 gracefully), just a "don't bother" threshold
# matching what compute_core_signals already does for the daily case (>= 200).
MIN_DAILY_BARS_FOR_STAGE = 200
MIN_WEEKLY_BARS_FOR_STAGE = 60
MIN_BARS_FOR_50_200_CROSS = 200
MIN_BARS_FOR_20_50_CROSS = 50

# The propietario's own short/medium-term system times entries and exits off
# a 21-period fast MA against the 50-period, not the more common 20 (see
# docs/quant_methodology.md) - this is the actual period every "*_20_50"
# field below is computed with. Those field/constant names keep the
# historical "20_50" spelling deliberately (API/schema stability - a rename
# is pure churn with zero behavior change), but every one of them is real
# 21-period data, not 20 - this constant is the single source of truth for
# that period, never a bare literal.
FAST_MA_PERIOD = 21


@dataclass(frozen=True, slots=True)
class TimeframeRead:
    """One timeframe's full technical read, off *closed* bars only (see
    `technical_analysis.closed_bars`) - every field here backs a discrete
    decision (a cross, a stage, an alignment call), never a live, still-moving
    read."""

    timeframe: str  # "weekly" | "daily" | "intraday_1h"
    trend: ta.TrendState
    stage: ta.Stage | None
    ma_cross_50_200: str | None  # "golden" | "death" | None, confirmed within the last few closed bars
    ma_cross_20_50: str | None  # despite the name, the fast leg is FAST_MA_PERIOD (21), not 20 - see that constant
    # Full quality read (separation/slope/volume) for the fast/50 pair -
    # unlike the 50/200 pair, `exit_engine.py` needs `quality`, not just
    # direction, for its death-cross hard trigger, so this carries the whole
    # object rather than just bars_since like ma_cross_50_200 does.
    cross_quality_20_50: ta.CrossQuality | None
    imminent_cross_50_200: ta.ImminentCross | None
    imminent_cross_20_50: ta.ImminentCross | None  # fast/50 pair, see FAST_MA_PERIOD
    macd_cross: str | None  # "bullish" | "bearish" - MACD line vs its own signal line
    macd_histogram_turning: str | None  # "up" | "down" - histogram's own last-2-bar direction
    rsi14: float | None
    adx14: float | None
    di_bias: str | None  # "bullish" | "bearish" - +DI vs -DI, only reported when ADX shows a real trend (>=25)
    price_vs_sma20: str | None  # "above" | "below" - vs the fast MA (FAST_MA_PERIOD=21), not a literal SMA20
    price_vs_sma50: str | None
    price_vs_sma200: str | None
    bars_since_cross: int | None  # bars since ma_cross_50_200 confirmed, if any


@dataclass(frozen=True, slots=True)
class MultiTimeframeRead:
    weekly: TimeframeRead | None  # None only when there isn't yet a single closed week of history
    daily: TimeframeRead
    intraday: TimeframeRead | None  # not populated yet - see D1/Fase 1.2 in docs/quant_methodology.md
    alignment: str  # "bullish_aligned" | "bearish_aligned" | "conflicted" | "transitioning"
    alignment_score: float  # -1.0 (bearish_aligned) to +1.0 (bullish_aligned)
    conflicts: list[str]  # human-readable (Spanish) descriptions of each discrepancy found, for the UI


def _last(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _price_vs(price: float, moving_average: float | None) -> str | None:
    if moving_average is None or pd.isna(moving_average):
        return None
    return "above" if price > moving_average else "below"


def _read_timeframe(
    df: pd.DataFrame, timeframe: str, stage_ma_window: int, min_bars_for_stage: int,
    now: datetime | None = None, cutoff: time | None = None,
) -> TimeframeRead | None:
    closed = ta.closed_bars(df, now=now, cutoff=cutoff)
    if len(closed) < 2:
        return None
    close, high, low, volume = closed["close"], closed["high"], closed["low"], closed["volume"]
    price = float(close.iloc[-1])

    sma_fast_s, sma50_s, sma200_s = ta.sma(close, FAST_MA_PERIOD), ta.sma(close, 50), ta.sma(close, 200)
    sma_fast, sma50, sma200 = _last(sma_fast_s), _last(sma50_s), _last(sma200_s)
    trend = ta.classify_trend(price, sma_fast, sma50, sma200)

    stage = None
    if len(close) >= min_bars_for_stage:
        stage_ma = ta.sma(close, stage_ma_window)
        stage = ta.classify_stage(price, stage_ma)

    ma_cross_50_200, imminent_50_200, bars_since_cross = None, None, None
    if len(close) >= MIN_BARS_FOR_50_200_CROSS:
        quality = ta.detect_cross_with_quality(sma50_s, sma200_s, high, low, close, volume)
        if quality is not None:
            ma_cross_50_200, bars_since_cross = quality.direction, quality.bars_since
        imminent_50_200 = ta.detect_imminent_cross(sma50_s, sma200_s)

    ma_cross_20_50, cross_quality_20_50, imminent_20_50 = None, None, None
    if len(close) >= MIN_BARS_FOR_20_50_CROSS:
        cross_quality_20_50 = ta.detect_cross_with_quality(sma_fast_s, sma50_s, high, low, close, volume)
        if cross_quality_20_50 is not None:
            ma_cross_20_50 = cross_quality_20_50.direction
        imminent_20_50 = ta.detect_imminent_cross(sma_fast_s, sma50_s)

    macd_line, macd_signal, macd_hist = ta.macd(close)
    raw_macd_cross = ta.detect_recent_cross(macd_line, macd_signal, lookback=5)
    macd_cross = {"golden": "bullish", "death": "bearish"}.get(raw_macd_cross)
    macd_histogram_turning = None
    valid_hist = macd_hist.dropna()
    if len(valid_hist) >= 2:
        macd_histogram_turning = "up" if valid_hist.iloc[-1] > valid_hist.iloc[-2] else "down"

    adx14 = _last(ta.adx(high, low, close))
    plus_di, minus_di = ta.dmi(high, low, close)
    di_bias = None
    if adx14 is not None and adx14 >= 25:
        plus_last, minus_last = _last(plus_di), _last(minus_di)
        if plus_last is not None and minus_last is not None:
            di_bias = "bullish" if plus_last > minus_last else "bearish"

    return TimeframeRead(
        timeframe=timeframe,
        trend=trend,
        stage=stage,
        ma_cross_50_200=ma_cross_50_200,
        ma_cross_20_50=ma_cross_20_50,
        cross_quality_20_50=cross_quality_20_50,
        imminent_cross_50_200=imminent_50_200,
        imminent_cross_20_50=imminent_20_50,
        macd_cross=macd_cross,
        macd_histogram_turning=macd_histogram_turning,
        rsi14=_last(ta.rsi(close)),
        adx14=adx14,
        di_bias=di_bias,
        price_vs_sma20=_price_vs(price, sma_fast),
        price_vs_sma50=_price_vs(price, sma50),
        price_vs_sma200=_price_vs(price, sma200),
        bars_since_cross=bars_since_cross,
    )


def timeframe_bias(read: TimeframeRead | None) -> str:
    """"bullish" | "bearish" | "neutral" | "unknown" - the one shared
    definition of "which way is this timeframe leaning", for both weekly and
    daily reads. Before this existed, `combine_timeframes` below (drives
    `alignment`, shown in the UI) and `exit_engine.py`'s own
    `weekly_not_bullish` computed this independently and could disagree: this
    function already counts a Weinstein Stage 2 read as bullish even when
    `trend` alone reads SIDEWAYS, while `exit_engine.py`'s old check only
    ever looked at `trend` - stricter than what the UI showed for the exact
    same data, so a position could read "bullish_aligned" on screen while
    simultaneously arming the hardest EXIT_NOW trigger underneath it.

    "unknown" - not "neutral", and deliberately never folded into "bearish"
    - when `read is None` or `read.price_vs_sma200 is None`: fewer than
    ~200 bars of *this timeframe's own* history (~3.85 years on weekly bars)
    isn't enough for a real SMA200-based trend read yet. A hard trigger that
    needs "confirmed not bullish" to arm must never treat "we genuinely
    don't know" the same as "we know it's not bullish" - see
    `exit_engine.py`'s own use of this distinction."""
    if read is None or read.price_vs_sma200 is None:
        return "unknown"
    if read.trend == ta.TrendState.UPTREND or read.stage == ta.Stage.STAGE_2:
        return "bullish"
    if read.trend == ta.TrendState.DOWNTREND or read.stage == ta.Stage.STAGE_4:
        return "bearish"
    return "neutral"


def _empty_daily_read() -> TimeframeRead:
    # MIN_BARS_REQUIRED upstream (ticker_analysis_service.py) already guards
    # against a frame this thin in practice - this is a defensive fallback,
    # not an expected path.
    return TimeframeRead(
        timeframe="daily",
        trend=ta.TrendState.SIDEWAYS,
        stage=None,
        ma_cross_50_200=None,
        ma_cross_20_50=None,
        cross_quality_20_50=None,
        imminent_cross_50_200=None,
        imminent_cross_20_50=None,
        macd_cross=None,
        macd_histogram_turning=None,
        rsi14=None,
        adx14=None,
        di_bias=None,
        price_vs_sma20=None,
        price_vs_sma50=None,
        price_vs_sma200=None,
        bars_since_cross=None,
    )


def combine_timeframes(weekly: TimeframeRead | None, daily: TimeframeRead) -> tuple[str, float, list[str]]:
    """The alignment rules, isolated as a pure function of two already-computed
    reads (deliberately separate from `analyze_multi_timeframe`'s data
    plumbing, so this - the part that actually encodes the top-down
    hierarchy - can be unit-tested against hand-built `TimeframeRead`s
    without needing synthetic price series that happen to trip every
    SMA/stage boundary just right).

    Returns `(alignment, alignment_score, conflicts)`:
    - weekly and daily both bearish (trend or Weinstein stage) -> "bearish_aligned"
      (-1.0), the strongest exit signal this read can produce.
    - weekly and daily both bullish -> "bullish_aligned" (+1.0).
    - they disagree -> "conflicted", and the disagreement is scored *against*
      the setup (a negative or reduced alignment_score), never neutrally - a
      conflicted setup is worse than a neutral one, not equal to it. Which
      side the conflict scores toward mirrors which timeframe carries the
      bias: weekly-bearish-over-daily-bullish scores negative (a bounce
      inside a bigger downtrend, per Faber/Weinstein-style top-down analysis)
      even though the daily read alone would look constructive.
    - neither timeframe shows a clear trend, or the weekly read isn't
      confirmed yet (no weekly bars at all, or fewer than ~200 of them -
      ~3.85 years - for a real SMA200-based read; see `timeframe_bias`) ->
      "transitioning" / the daily-only read, neutral - never guessed from an
      unconfirmed weekly read.
    """
    conflicts: list[str] = []
    weekly_bias_read = timeframe_bias(weekly)
    if weekly_bias_read == "unknown":
        # No weekly read at all, or one that exists but isn't confirmed yet -
        # the daily read stands alone, honestly labeled as such rather than
        # guessing a bias from too little (or no) weekly data.
        daily_bias = timeframe_bias(daily)
        if daily_bias == "bullish":
            return "bullish_aligned", 1.0, conflicts
        if daily_bias == "bearish":
            return "bearish_aligned", -1.0, conflicts
        return "transitioning", 0.0, conflicts

    # weekly_bias_read != "unknown" guarantees weekly is not None from here on.
    daily_bias_read = timeframe_bias(daily)
    weekly_bullish, weekly_bearish = weekly_bias_read == "bullish", weekly_bias_read == "bearish"
    daily_bullish, daily_bearish = daily_bias_read == "bullish", daily_bias_read == "bearish"

    if weekly_bearish and daily_bearish:
        alignment, alignment_score = "bearish_aligned", -1.0
    elif weekly_bullish and daily_bullish:
        alignment, alignment_score = "bullish_aligned", 1.0
    elif weekly_bearish and daily_bullish:
        alignment, alignment_score = "conflicted", -0.3
        conflicts.append(
            "La semanal está en tendencia bajista pero la diaria muestra fuerza alcista - "
            "puede ser un rebote dentro de una tendencia bajista mayor, no un cambio de tendencia."
        )
    elif weekly_bullish and daily_bearish:
        alignment, alignment_score = "conflicted", 0.3
        conflicts.append(
            "La semanal sigue alcista pero la diaria ya se ha girado a la baja - "
            "posible corrección dentro de la tendencia mayor; vigilar si se confirma también en semanal."
        )
    else:
        alignment, alignment_score = "transitioning", 0.0

    if weekly.ma_cross_50_200 == "death" and daily.ma_cross_50_200 != "death":
        conflicts.append("Cruce de medias bajista (SMA50/SMA200) ya confirmado en semanal, todavía no en diario.")
    if weekly.macd_cross == "bearish" and daily.macd_cross == "bullish":
        conflicts.append("El MACD semanal ha girado bajista mientras el diario todavía marca un cruce alcista.")

    return alignment, alignment_score, conflicts


def analyze_multi_timeframe(
    daily_df: pd.DataFrame, now: datetime | None = None, cutoff: time | None = None
) -> MultiTimeframeRead:
    """Weekly bias, daily execution - see module docstring and
    `combine_timeframes` for the alignment rules. `daily_df` is the same
    daily OHLCV frame every caller already fetches; weekly is derived from it
    at zero extra network cost via `resample_ohlcv`. `now` passes straight
    through to `technical_analysis.closed_bars`/`resample_ohlcv` (both
    timeframes) - injectable for tests that need to control whether "today's"
    bar/period counts as settled; defaults to the real current UTC time in
    production, same as those functions themselves. `cutoff` similarly passes
    through to `closed_bars` - this module stays market-universe-agnostic on
    purpose, so a caller that knows the ticker should pass
    `market_universe.closed_bar_cutoff_for_ticker(ticker)` here instead of
    leaving the US-centric default to apply to every region (Segunda
    auditoría, Bloque 2)."""
    daily = (
        _read_timeframe(daily_df, "daily", DAILY_STAGE_MA_WINDOW, MIN_DAILY_BARS_FOR_STAGE, now=now, cutoff=cutoff)
        or _empty_daily_read()
    )

    weekly_df = ta.resample_ohlcv(daily_df, WEEKLY_RULE, now=now)
    weekly = (
        _read_timeframe(
            weekly_df, "weekly", WEEKLY_STAGE_MA_WINDOW, MIN_WEEKLY_BARS_FOR_STAGE, now=now, cutoff=cutoff
        )
        if len(weekly_df) >= 2
        else None
    )

    alignment, alignment_score, conflicts = combine_timeframes(weekly, daily)
    return MultiTimeframeRead(
        weekly=weekly, daily=daily, intraday=None, alignment=alignment, alignment_score=alignment_score,
        conflicts=conflicts,
    )
