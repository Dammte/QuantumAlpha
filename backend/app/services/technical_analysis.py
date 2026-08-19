"""Technical-analysis primitives shared by the market screener, movers, sector
strength and support/resistance features.

Pure functions over pandas/numpy, no framework or I/O dependencies, so they're
cheap to unit test in isolation - same philosophy as `metrics_service.py`.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def bollinger_bands(
    closes: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Middle band (SMA), upper and lower bands (+/- `num_std` standard deviations).
    Price hugging the upper band signals strength (not automatically "overbought");
    a squeeze (bands narrowing) often precedes a volatility expansion/breakout."""
    middle = sma(closes, window)
    std = closes.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def rsi(closes: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.where(avg_loss != 0, 100.0)


def macd(
    closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(closes, fast) - ema(closes, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def atr_multiple_from_sma(
    close: pd.Series, high: pd.Series, low: pd.Series, sma_window: int = 50, atr_window: int = 14
) -> float | None:
    """How many ATRs the latest close sits above/below its SMA - a simple
    overextension/"euphoria" gauge: a large positive value means the price has
    run far above its average trend and a mean-reversion or blow-off top is more
    likely than a continuation at the same pace."""
    if len(close) < max(sma_window, atr_window) + 1:
        return None
    moving_avg = sma(close, sma_window).iloc[-1]
    latest_atr = atr(high, low, close, atr_window).iloc[-1]
    if pd.isna(moving_avg) or pd.isna(latest_atr) or latest_atr == 0:
        return None
    return float((close.iloc[-1] - moving_avg) / latest_atr)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume, added on up days and subtracted on
    down days. Purely causal (each value only uses data up to that bar), so it
    can be sliced `[:i+1]` for point-in-time backtest replay with no lookahead."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def rolling_position_in_range(series: pd.Series, window: int) -> pd.Series:
    """Where the latest value sits within its own trailing `window`-bar
    min-max range, as a 0 (at the low) to 1 (at the high) fraction. Vectorized
    (rolling min/max), not a per-bar Python callback, so it stays cheap even
    over a decade of daily bars."""
    roll_min = series.rolling(window, min_periods=window).min()
    roll_max = series.rolling(window, min_periods=window).max()
    span = roll_max - roll_min
    return ((series - roll_min) / span.replace(0, np.nan)).clip(0, 1)


OBV_DIVERGENCE_WINDOW = 20
OBV_DIVERGENCE_GAP = 0.35


def obv_divergence(close: pd.Series, volume: pd.Series, window: int = OBV_DIVERGENCE_WINDOW) -> str | None:
    """Wyckoff's "effort vs result", made concrete: compares where price sits in
    its own trailing range against where On-Balance Volume sits in *its* own
    trailing range. Both normalized to 0-1 first so this compares relative
    position, not raw units that aren't otherwise comparable (price vs a
    cumulative volume count).

    Price near the top of its range while OBV is NOT near the top of its own
    range means the advance isn't backed by real net buying pressure - a
    Wyckoff distribution/"upthrust" warning ("bearish"). The mirror case (price
    near its range low, OBV holding up) reads as quiet accumulation
    ("bullish") - selling pressure drying up despite the falling price. A
    modest gap is normal noise; only a gap of `OBV_DIVERGENCE_GAP` or more
    (calibrated against the factor-ablation study, see
    `scripts/factor_ablation_study.py`) counts as a real divergence.
    """
    if len(close) < window:
        return None
    obv_series = obv(close, volume)
    price_pos = rolling_position_in_range(close, window).iloc[-1]
    obv_pos = rolling_position_in_range(obv_series, window).iloc[-1]
    if pd.isna(price_pos) or pd.isna(obv_pos):
        return None
    gap = price_pos - obv_pos
    if gap >= OBV_DIVERGENCE_GAP:
        return "bearish"
    if gap <= -OBV_DIVERGENCE_GAP:
        return "bullish"
    return None


def relative_volume(volume: pd.Series, window: int = 20) -> float | None:
    """Today's volume vs the average of the prior `window` sessions (excluding today)."""
    if len(volume) < window + 1:
        return None
    baseline = volume.iloc[-(window + 1) : -1].mean()
    if pd.isna(baseline) or baseline == 0:
        return None
    return float(volume.iloc[-1] / baseline)


def pct_change_over(closes: pd.Series, periods: int) -> float | None:
    if len(closes) <= periods:
        return None
    previous = closes.iloc[-(periods + 1)]
    if pd.isna(previous) or previous == 0:
        return None
    return float(closes.iloc[-1] / previous - 1)


def distance_to_rolling_extreme(
    closes: pd.Series, window: int, kind: str, min_periods: int | None = None
) -> float | None:
    """% distance of the latest close from its rolling max (`kind="high"`) or
    min (`kind="low"`) over `window` sessions - e.g. distance to the 52-week high.

    Returns `None`, not an approximation, when fewer than `window` bars are
    available: a "52-week high" computed on 60 days of history isn't a
    52-week high, it's a 3-month high mislabeled as an annual one - and this
    fed straight into the checklist's single most validated factor (see
    `recommendation_engine.py`'s comment on `minervini_range_confirmed`), so
    silently answering a different question than the one asked was a real
    bug, not a harmless fallback. `min_periods` lets a caller opt into a
    shorter minimum on purpose (e.g. a test exercising the max/min logic
    itself, not the "is this really annual" question) - it defaults to
    requiring the full `window`."""
    required = window if min_periods is None else min_periods
    if len(closes) < required:
        return None
    windowed = closes.iloc[-window:]
    extreme = windowed.max() if kind == "high" else windowed.min()
    if pd.isna(extreme) or extreme == 0:
        return None
    return float(closes.iloc[-1] / extreme - 1)


def rolling_extreme_price(
    closes: pd.Series, window: int, kind: str, min_periods: int | None = None
) -> float | None:
    """Same as `distance_to_rolling_extreme` but returns the raw price level
    instead of a % distance - used by the Minervini checklist, which tests
    against the actual 52-week high/low price. See `distance_to_rolling_extreme`
    for why `None` on a short series is the correct answer, not an
    approximation, and what `min_periods` is for."""
    required = window if min_periods is None else min_periods
    if len(closes) < required:
        return None
    windowed = closes.iloc[-window:]
    extreme = windowed.max() if kind == "high" else windowed.min()
    return None if pd.isna(extreme) else float(extreme)


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series]:
    """Wilder's +DI / -DI (Directional Indicators)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    smoothed_tr = true_range(high, low, close).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)
    return plus_di, minus_di


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's ADX: how strongly a market is trending (either direction) - see
    `dmi` for the directional component. Standard read: <20 choppy/range-bound,
    20-25 ambiguous, >25 trending, >40 a strong (possibly overextended) trend."""
    plus_di, minus_di = dmi(high, low, close, window)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def mansfield_rs(closes: pd.Series, benchmark_closes: pd.Series, window: int = 200) -> pd.Series:
    """Stan Weinstein's relative-strength line: the stock/benchmark ratio expressed
    as a % deviation from its own rolling average. Positive & rising = outperforming
    the benchmark; a cross from negative to positive is Weinstein's Stage 2
    confirmation, the reverse cross a Stage 4 warning. `window=200` daily sessions
    approximates his original 52-week-on-weekly-bars parameter.
    """
    aligned = pd.concat([closes, benchmark_closes], axis=1, join="inner")
    ratio = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    baseline = sma(ratio, window)
    return (ratio / baseline - 1) * 100


def rs_raw_score(closes: pd.Series) -> float | None:
    """IBD-style weighted composite return: the trailing quarter counts double
    each of the other three quarters, so a stock that's turned it on recently
    outranks one coasting on an old move. Feed this into a cross-sectional
    percentile rank (1-99) across your universe to get an RS Rating."""
    periods = {63: 0.4, 126: 0.2, 189: 0.2, 252: 0.2}
    components = {n: pct_change_over(closes, n) for n in periods}
    if any(v is None for v in components.values()):
        return None
    return sum(components[n] * weight for n, weight in periods.items())


def sma_slope_positive(series: pd.Series, lookback: int = 25) -> bool | None:
    """Whether a moving-average series has been rising over the last `lookback`
    bars - the "is the 200-day MA trending up" leg of the Minervini checklist."""
    valid = series.dropna()
    if len(valid) <= lookback:
        return None
    previous = valid.iloc[-lookback - 1]
    if pd.isna(previous) or previous == 0:
        return None
    return bool(valid.iloc[-1] > previous)


class Stage(str, Enum):
    STAGE_1 = "stage1"  # basing / accumulation
    STAGE_2 = "stage2"  # advancing / markup
    STAGE_3 = "stage3"  # topping / distribution
    STAGE_4 = "stage4"  # declining / markdown


def classify_stage(
    price: float, sma_trend: pd.Series, lookback: int = 20, flat_threshold: float = 0.005
) -> Stage | None:
    """Weinstein's 4-stage cycle, proxied with a daily `sma_trend` (canonically
    the 150-day MA, standing in for his 30-week MA on weekly bars). A clearly
    rising MA with price above it is Stage 2 (the only stage worth buying in);
    a clearly falling MA with price below it is Stage 4 (be out or short). A
    flat MA is either Stage 1 or Stage 3 depending on whether the flattening
    followed a prior decline (basing) or a prior advance (topping) - inferred
    from where the MA sat ~100 bars ago.
    """
    valid = sma_trend.dropna()
    if len(valid) <= lookback:
        return None
    sma_now = valid.iloc[-1]
    sma_prev = valid.iloc[-lookback - 1]
    if pd.isna(sma_now) or pd.isna(sma_prev) or sma_prev == 0:
        return None
    slope_pct = (sma_now - sma_prev) / sma_prev

    if slope_pct > flat_threshold and price > sma_now:
        return Stage.STAGE_2
    if slope_pct < -flat_threshold and price < sma_now:
        return Stage.STAGE_4

    long_lookback = 100
    if len(valid) > long_lookback:
        sma_long_ago = valid.iloc[-long_lookback - 1]
        if not pd.isna(sma_long_ago) and sma_long_ago != 0:
            if sma_now > sma_long_ago * 1.03:
                return Stage.STAGE_3
            if sma_now < sma_long_ago * 0.97:
                return Stage.STAGE_1
    return Stage.STAGE_1


def minervini_checklist(
    price: float,
    sma50: float | None,
    sma150: float | None,
    sma200: float | None,
    sma200_trending_up: bool | None,
    price_52w_low: float | None,
    price_52w_high: float | None,
    rs_rating: int | None,
) -> dict[str, bool]:
    """Mark Minervini's 8-point "Trend Template" - a stock must pass all 8 to
    count as a confirmed Stage 2 leader worth buying. Returns each criterion so
    the caller can show partial credit, not just pass/fail."""
    return {
        "price_above_150_and_200": sma150 is not None and sma200 is not None and price > sma150 and price > sma200,
        "150_above_200": sma150 is not None and sma200 is not None and sma150 > sma200,
        "200_trending_up": bool(sma200_trending_up),
        "50_above_150_and_200": (
            sma50 is not None and sma150 is not None and sma200 is not None and sma50 > sma150 and sma50 > sma200
        ),
        "price_above_50": sma50 is not None and price > sma50,
        "price_25pct_above_52w_low": (
            price_52w_low is not None and price_52w_low > 0 and price >= price_52w_low * 1.25
        ),
        "price_within_25pct_of_52w_high": (
            price_52w_high is not None and price_52w_high > 0 and price >= price_52w_high * 0.75
        ),
        "rs_rating_70_plus": rs_rating is not None and rs_rating >= 70,
    }


class TrendState(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


def vix_regime(level: float | None) -> str:
    """VIX level -> a named fear regime. Lives here (not in
    MarketContextService, which originally owned it) so both the live "Contexto"
    dashboard and anything computing a per-ticker/backtest market-regime read
    (recommendation_engine.py, walk_forward_backtest.py) can share one
    definition without a circular import - this module is the app's
    pure-function, no-I/O layer everything else is allowed to depend on."""
    if level is None:
        return "desconocido"
    if level < 12:
        return "complacencia"
    if level < 20:
        return "normal"
    if level < 30:
        return "miedo elevado"
    if level < 40:
        return "pánico"
    return "crisis"


def market_regime_inputs(
    benchmark_close: pd.Series | None, vix_close: pd.Series | None
) -> tuple["TrendState | None", str | None]:
    """Meb Faber's tactical SMA-200 filter on a benchmark index, plus the VIX
    fear regime - the two market-wide inputs `build_recommendation` uses to
    gate an individual ticker's verdict. A pure function of two price series
    so it can be reused identically by the live recommendation path and the
    point-in-time backtest replay (fed a `.iloc[:i+1]` slice of each series -
    still fully causal, no lookahead)."""
    market_trend = None
    if benchmark_close is not None and len(benchmark_close) >= 200:
        b_close = benchmark_close.dropna()
        if len(b_close) >= 200:
            b_sma20 = sma(b_close, 20).iloc[-1] if len(b_close) >= 20 else None
            b_sma50 = sma(b_close, 50).iloc[-1] if len(b_close) >= 50 else None
            b_sma200 = sma(b_close, 200).iloc[-1]
            if not pd.isna(b_sma200):
                market_trend = classify_trend(
                    float(b_close.iloc[-1]),
                    None if b_sma20 is None or pd.isna(b_sma20) else float(b_sma20),
                    None if b_sma50 is None or pd.isna(b_sma50) else float(b_sma50),
                    float(b_sma200),
                )

    vix_regime_label = None
    if vix_close is not None and not vix_close.empty:
        clean_vix = vix_close.dropna()
        if len(clean_vix):
            vix_regime_label = vix_regime(float(clean_vix.iloc[-1]))

    return market_trend, vix_regime_label


def classify_trend(price: float, sma20: float | None, sma50: float | None, sma200: float | None) -> TrendState:
    """Ordered moving averages (price > MA20 > MA50 > MA200, or the mirror) read as a
    clean trend; anything tangled in between reads as sideways/transition."""
    if sma20 is None or sma50 is None or sma200 is None:
        return TrendState.SIDEWAYS
    if price > sma20 > sma50 > sma200:
        return TrendState.UPTREND
    if price < sma20 < sma50 < sma200:
        return TrendState.DOWNTREND
    return TrendState.SIDEWAYS


def detect_recent_cross(fast: pd.Series, slow: pd.Series, lookback: int = 5) -> str | None:
    """"golden" if `fast` crossed above `slow` within the last `lookback` bars,
    "death" if it crossed below, else None. Used for MA50/MA200 and MACD/signal crosses.

    `lookback` counts bar-to-bar transitions, not points in the window: seeing
    a transition that happened `lookback` bars ago needs both that bar *and*
    the one right before it, so the slice pulled is `lookback + 1` points wide.
    (A prior off-by-one pulled only `lookback` points - `lookback - 1`
    transitions - so `lookback=5` actually only checked the last 4 bar-to-bar
    changes; a cross exactly 5 bars back was silently missed.)"""
    diff = (fast - slow).dropna()
    if len(diff) < 2:
        return None
    window = diff.iloc[-(lookback + 1) :] if len(diff) > lookback + 1 else diff
    signs = np.sign(window)
    changes = signs.diff().dropna()
    if (changes > 0).any() and signs.iloc[-1] > 0:
        return "golden"
    if (changes < 0).any() and signs.iloc[-1] < 0:
        return "death"
    return None


IMMINENT_CROSS_LOOKBACK = 15  # bars of recent gap history the trend line is fit over
IMMINENT_CROSS_HORIZON = 10  # only flag a projected cross within this many bars ahead
IMMINENT_CROSS_MIN_R2 = 0.5  # the gap's recent trajectory must actually look like a trend, not noise


@dataclass(frozen=True, slots=True)
class ImminentCross:
    """A cross `detect_recent_cross` hasn't seen yet because it hasn't happened -
    a projection, not a confirmation. See `detect_imminent_cross`."""

    direction: str  # "golden" | "death" - which way the projected cross points
    bars_until: int  # projected trading days until fast and slow would meet, at the current rate
    r_squared: float  # how well a straight line explains the recent gap trajectory (0-1)


def detect_imminent_cross(fast: pd.Series, slow: pd.Series) -> ImminentCross | None:
    """Fits a straight line to `fast - slow` over the last `IMMINENT_CROSS_LOOKBACK`
    bars and, if that line is actually heading toward zero (not away from it, not
    flat), projects how many bars until it gets there - a leading complement to
    `detect_recent_cross`'s lagging, already-happened read. Exists because a
    confirmed cross is, by construction, old news: two moving averages don't
    snap together instantly, they converge over days, and by the time SMA50
    has actually crossed SMA200 a meaningful chunk of the move behind it has
    usually already happened. Catching the convergence *before* the cross
    finishes gives a genuine head start to act on, at the cost of being a
    projection rather than a fact - which is why this is surfaced as a
    separate, clearly-labeled field (never folded into the scored
    `ma_cross` factor) and gated by `IMMINENT_CROSS_MIN_R2`: a choppy,
    directionless gap can trivially produce a spurious "3 bars until cross"
    from four noisy points, so this only fires when a straight line actually
    explains most of the recent trajectory (R² is exactly that goodness-of-fit
    check), and only within `IMMINENT_CROSS_HORIZON` bars - far enough out and
    a linear extrapolation stops being a reasonable assumption anyway."""
    gap = (fast - slow).dropna()
    if len(gap) < IMMINENT_CROSS_LOOKBACK:
        return None
    recent = gap.iloc[-IMMINENT_CROSS_LOOKBACK:].to_numpy()
    current_gap = float(recent[-1])
    if current_gap == 0:
        return None  # already crossed, or exactly on it - detect_recent_cross's territory

    x = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent, 1)
    predicted = slope * x + intercept
    residual_ss = float(np.sum((recent - predicted) ** 2))
    total_ss = float(np.sum((recent - recent.mean()) ** 2))
    r_squared = 1 - residual_ss / total_ss if total_ss > 0 else 0.0
    if r_squared < IMMINENT_CROSS_MIN_R2:
        return None

    if slope == 0:
        return None
    bars_until = -current_gap / slope
    if bars_until <= 0 or bars_until > IMMINENT_CROSS_HORIZON:
        return None  # moving away from zero, or too far out for a straight-line projection to mean much

    direction = "death" if current_gap > 0 else "golden"
    return ImminentCross(direction=direction, bars_until=round(bars_until), r_squared=round(r_squared, 3))


# A body at least this much bigger than the prior one's, in the *same*
# direction as the engulf, filters out a technically-qualifying but trivial
# one-tick engulf (a barely-bigger body on a low-volatility day) that pattern
# purists wouldn't call a real reversal signal.
ENGULFING_MIN_BODY_RATIO = 1.0


def detect_engulfing_pattern(open_: pd.Series, close: pd.Series) -> str | None:
    """Classic two-candle reversal: "bullish_engulfing" when a down candle is
    immediately followed by an up candle whose real body fully contains the
    prior one's (the buyers who showed up today didn't just stop the
    decline, they erased the whole prior session and then some) -
    "bearish_engulfing" is the mirror. Read purely off closing (settled)
    bars, same as everything else in this module - never off today's
    still-moving intraday candle, which isn't "engulfing" anything yet.
    """
    valid_open = open_.dropna()
    valid_close = close.dropna()
    if len(valid_open) < 2 or len(valid_close) < 2:
        return None

    prev_open, prev_close = float(valid_open.iloc[-2]), float(valid_close.iloc[-2])
    curr_open, curr_close = float(valid_open.iloc[-1]), float(valid_close.iloc[-1])
    prev_body = abs(prev_close - prev_open)
    if prev_body == 0:
        return None  # a doji has no real body to engulf

    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open
    curr_bullish = curr_close > curr_open
    curr_bearish = curr_close < curr_open
    curr_body = abs(curr_close - curr_open)

    if (
        prev_bearish
        and curr_bullish
        and curr_open <= prev_close
        and curr_close >= prev_open
        and curr_body >= prev_body * ENGULFING_MIN_BODY_RATIO
    ):
        return "bullish_engulfing"
    if (
        prev_bullish
        and curr_bearish
        and curr_open >= prev_close
        and curr_close <= prev_open
        and curr_body >= prev_body * ENGULFING_MIN_BODY_RATIO
    ):
        return "bearish_engulfing"
    return None


@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: float
    kind: str  # "support" | "resistance"
    strength: int  # how many pivots clustered into this level
    distance_pct: float  # % distance from the current price


def _fractal_pivots(series: pd.Series, left: int, right: int, kind: str) -> list[float]:
    """A bar is a pivot high if it's the max of its `left`+`right` neighborhood
    (mirror for pivot low) - the standard "fractal" swing-point definition."""
    values = series.to_numpy()
    pivots = []
    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        center = values[i]
        if kind == "high" and center == window.max() and np.count_nonzero(window == center) == 1:
            pivots.append(float(center))
        elif kind == "low" and center == window.min() and np.count_nonzero(window == center) == 1:
            pivots.append(float(center))
    return pivots


def _cluster_levels(prices: list[float], tolerance_pct: float) -> list[tuple[float, int]]:
    """Merges nearby pivot prices into levels (price = cluster average, strength =
    number of pivots in it) - several swing points bunched together make a level
    more significant than a single isolated touch."""
    if not prices:
        return []
    clusters: list[list[float]] = []
    for price in sorted(prices):
        if clusters and abs(price - np.mean(clusters[-1])) / np.mean(clusters[-1]) <= tolerance_pct / 100:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def support_resistance_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    left: int = 3,
    right: int = 3,
    tolerance_pct: float = 1.5,
    max_levels: int = 5,
) -> list[PriceLevel]:
    """Real swing-pivot support/resistance detection: find fractal swing highs/lows,
    cluster the ones that sit close together into levels, then keep the strongest
    levels nearest to the current price on each side."""
    if close.empty:
        return []
    current_price = float(close.iloc[-1])

    resistance_pivots = [p for p in _fractal_pivots(high, left, right, "high") if p > current_price]
    support_pivots = [p for p in _fractal_pivots(low, left, right, "low") if p < current_price]

    levels: list[PriceLevel] = []
    for price, strength in _cluster_levels(resistance_pivots, tolerance_pct):
        levels.append(
            PriceLevel(price=price, kind="resistance", strength=strength, distance_pct=price / current_price - 1)
        )
    for price, strength in _cluster_levels(support_pivots, tolerance_pct):
        levels.append(
            PriceLevel(price=price, kind="support", strength=strength, distance_pct=price / current_price - 1)
        )

    resistances = sorted((lv for lv in levels if lv.kind == "resistance"), key=lambda lv: lv.price)[:max_levels]
    supports = sorted((lv for lv in levels if lv.kind == "support"), key=lambda lv: lv.price, reverse=True)[
        :max_levels
    ]
    return sorted(supports + resistances, key=lambda lv: lv.price)
