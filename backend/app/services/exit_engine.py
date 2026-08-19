"""Answers a different question than `recommendation_engine.py`, with
different evidence, on purpose: not "is this a good buy today" but "should
this specific, already-open position be closed, trimmed, protected, or left
alone". Entering and exiting are asymmetric decisions - a stock's RS Rating,
fundamentals, or 52-week range position are reasons someone *bought* it, and
none of them are a reason to keep *holding* it once the technical structure
that justified the entry has actually broken. A good fundamental doesn't
compensate for a broken chart in a portfolio managed on a scale of weeks.

**This module deliberately never imports `recommendation_engine.py` and never
receives RS Rating, revenue growth, profit margin, leverage, or the Minervini
checklist as inputs** - `evaluate_exit`'s signature is proof of that, not
just this docstring (see docs/quant_methodology.md's audit for why this
matters: before this module existed, a held position's sell/hold signal was
derived from `build_recommendation`'s buy-side score, so a death cross with
strong fundamentals could still net out positive and read as "hold" - exactly
the bug report that started this work).

Verdicts come in five urgency levels (`ExitUrgency`), not a binary
sell/hold - a broken trend and a broken stop are not the same emergency, and
collapsing them loses the "trim, don't panic-sell" and "tighten the stop,
don't act yet" middle ground a discretionary trader already uses. Every
triggered rule is recorded in `ExitAssessment.reasons` regardless of which
tier it belongs to (not just the reasons behind the winning tier) - same
transparency principle `portfolio_risk_service.py` already applies: the
propietario needs to see everything that's wrong, not just the headline.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta

# Same 3% "close enough to matter" bar portfolio_risk_service.py already uses
# for support/resistance proximity - kept as this module's own constant
# rather than importing it, so this file has zero dependency on anything
# outside technical_analysis.py/multi_timeframe.py (see module docstring).
RESISTANCE_PROXIMITY = 0.03

# First-pass thresholds, explicitly not ablation-calibrated yet (same status
# BUY_THRESHOLD/AVOID_THRESHOLD started at in recommendation_engine.py
# before *their* audit) - these encode the rules the propietario specified
# directly, not a fitted model.
IMMINENT_CROSS_50_200_MIN_R2 = 0.7
IMMINENT_CROSS_50_200_MAX_BARS = 5
IMMINENT_CROSS_20_50_MIN_R2 = 0.6
ADX_FALLING_ENTRY = 25.0  # ADX had to have been at least this high recently...
ADX_FALLING_EXIT = 20.0  # ...to count as "falling" once it drops below this
EXTENDED_ATR_MULTIPLE = 4.0
PROFIT_TIGHTEN_R = 1.5
# A position that's gone this many closed daily bars without reaching +1R
# (and hasn't already stopped out) is tying up capital that isn't working -
# a real, if easy-to-miss, cost (Fase 3's "stop temporal"). 20 sessions is
# ~1 trading month, matched to a portfolio managed on a scale of weeks, not
# months.
MAX_BARS_WITHOUT_PROGRESS = 20
STALLED_PROGRESS_R = 1.0


class ExitUrgency(str, Enum):
    EXIT_NOW = "exit_now"  # salir en la próxima sesión
    REDUCE = "reduce"  # recortar la posición (50%), no cerrarla
    TIGHTEN_STOP = "tighten_stop"  # subir el stop, mantener
    WATCH = "watch"  # vigilar de cerca, sin acción
    HOLD = "hold"  # tesis intacta


# Order matters: index = severity, lower is more urgent. Used to pick the
# single most severe tier that actually fired, while still keeping every
# triggered reason (see module docstring) regardless of its own tier.
_URGENCY_SEVERITY = [ExitUrgency.EXIT_NOW, ExitUrgency.REDUCE, ExitUrgency.TIGHTEN_STOP, ExitUrgency.WATCH]


@dataclass(frozen=True, slots=True)
class PositionContext:
    """What the exit engine needs to know about the position itself - not
    just the ticker's chart. Before this existed, nothing about an open
    position (entry price, entry date, the stop that was actually proposed at
    purchase) reached any signal-computing code at all (see D4 in
    docs/quant_methodology.md); this is that gap closed."""

    ticker: str
    average_cost: float
    quantity: float
    opened_at: date  # from the first BUY transaction of this ticker in this portfolio
    initial_stop: float | None  # proposed when the position was opened - see trade_plan_service.py
    current_stop: float | None  # trailing stop currently in force (starts equal to initial_stop)
    initial_target: float | None  # take-profit proposed when the position was opened
    highest_close_since_entry: float
    unrealized_pnl_pct: float | None
    r_multiple: float | None  # (price - entry) / (entry - initial_stop) - None if initial_stop is unknown
    bars_held: int


@dataclass(frozen=True, slots=True)
class ExitAssessment:
    urgency: ExitUrgency
    reasons: list[str]  # every triggered rule, in Spanish, across all tiers - see module docstring


def evaluate_exit(
    price: float,
    position: PositionContext,
    multi_timeframe: mtf.MultiTimeframeRead,
    consecutive_closes_below_daily_sma50: int,
    nearest_support: ta.PriceLevel | None,
    nearest_resistance: ta.PriceLevel | None,
    obv_divergence: str | None,
    relative_volume: float | None,
    rsi14: float | None,
    rsi_recent_max: float | None,
    adx14: float | None,
    adx_recent_max: float | None,
    atr_multiple: float | None,
    candlestick_pattern: str | None,
) -> ExitAssessment:
    """Runs every disparador in `docs/quant_methodology.md`'s exit-engine
    section and returns the single most severe urgency that fired, with every
    triggered reason recorded regardless of tier.

    Args (beyond `price`/`position`/`multi_timeframe`, which are self-
    explanatory):
    - `consecutive_closes_below_daily_sma50`: how many of the most recent
      *closed* daily bars have closed below the daily SMA50, counting back
      from the latest - 0 if the latest close is at or above it.
    - `nearest_support`/`nearest_resistance`: from
      `technical_analysis.support_resistance_levels` on the daily frame.
    - `obv_divergence`, `rsi14`, `adx14`, `atr_multiple`, `candlestick_pattern`:
      the same daily-bar reads `compute_core_signals` already produces -
      passed in rather than recomputed, so this stays a pure function of
      already-computed inputs (no OHLCV frame here at all).
    - `rsi_recent_max`/`adx_recent_max`: the max RSI/ADX over a short recent
      lookback (e.g. the same window `compute_core_signals` uses for the
      chart) - needed to tell "RSI is falling from an overbought peak" or
      "ADX is falling from a real trend" apart from "RSI/ADX is just low".
    """
    reasons_by_tier: dict[ExitUrgency, list[str]] = {tier: [] for tier in _URGENCY_SEVERITY}
    daily = multi_timeframe.daily
    weekly = multi_timeframe.weekly
    weekly_not_bullish = weekly is None or weekly.trend != ta.TrendState.UPTREND

    # --- EXIT_NOW: hard, non-negotiable triggers -----------------------

    if position.current_stop is not None and price < position.current_stop:
        reasons_by_tier[ExitUrgency.EXIT_NOW].append(
            f"Precio ({price:.2f}) ha perforado el stop de protección vigente ({position.current_stop:.2f})."
        )

    strong_break_volume = relative_volume is not None and relative_volume > 1.5
    if consecutive_closes_below_daily_sma50 >= 2 and weekly_not_bullish:
        reasons_by_tier[ExitUrgency.EXIT_NOW].append(
            "Cierre confirmado bajo la SMA50 diaria durante 2 sesiones consecutivas, "
            "con la tendencia semanal ya no alcista."
        )
    elif consecutive_closes_below_daily_sma50 >= 1 and strong_break_volume and weekly_not_bullish:
        reasons_by_tier[ExitUrgency.EXIT_NOW].append(
            "Cierre bajo la SMA50 diaria con volumen relativo elevado "
            f"({relative_volume:.1f}x), tendencia semanal ya no alcista."
        )

    if multi_timeframe.alignment == "bearish_aligned":
        reasons_by_tier[ExitUrgency.EXIT_NOW].append(
            "Alineación bajista total: la temporalidad semanal y la diaria coinciden en tendencia bajista."
        )

    if (
        daily.cross_quality_20_50 is not None
        and daily.cross_quality_20_50.direction == "death"
        and daily.cross_quality_20_50.quality != "noise"
        and daily.price_vs_sma20 == "below"
        and daily.price_vs_sma50 == "below"
    ):
        reasons_by_tier[ExitUrgency.EXIT_NOW].append(
            "Death cross SMA20/SMA50 confirmado (calidad "
            f"'{daily.cross_quality_20_50.quality}'), precio por debajo de ambas medias."
        )

    if nearest_support is not None and price < nearest_support.price:
        reasons_by_tier[ExitUrgency.EXIT_NOW].append(
            f"Rotura del último soporte relevante en {nearest_support.price:.2f} - estructura de mínimos perdida."
        )

    # --- REDUCE: recortar, no cerrar ------------------------------------

    imminent_50_200 = daily.imminent_cross_50_200
    if (
        imminent_50_200 is not None
        and imminent_50_200.direction == "death"
        and imminent_50_200.r_squared >= IMMINENT_CROSS_50_200_MIN_R2
        and imminent_50_200.bars_until <= IMMINENT_CROSS_50_200_MAX_BARS
        and daily.trend != ta.TrendState.UPTREND
    ):
        reasons_by_tier[ExitUrgency.REDUCE].append(
            "Cruce de medias bajista (SMA50/SMA200) proyectado en "
            f"~{imminent_50_200.bars_until} sesiones (R²={imminent_50_200.r_squared:.2f}), "
            "tendencia diaria ya no alcista."
        )

    rsi_falling_from_overbought = (
        rsi14 is not None and rsi_recent_max is not None and rsi_recent_max > 70 and rsi14 < rsi_recent_max
    )
    volume_growing = relative_volume is not None and relative_volume > 1.0
    if obv_divergence == "bearish" and rsi_falling_from_overbought and volume_growing:
        reasons_by_tier[ExitUrgency.REDUCE].append(
            "Divergencia bajista de OBV con RSI cayendo desde sobrecompra y volumen creciente en las caídas."
        )

    if position.initial_target is not None and price >= position.initial_target:
        reasons_by_tier[ExitUrgency.REDUCE].append(
            f"Objetivo original alcanzado ({position.initial_target:.2f}) - recoger parte y dejar correr el resto."
        )

    in_profit = position.r_multiple is not None and position.r_multiple > 0
    if atr_multiple is not None and atr_multiple > EXTENDED_ATR_MULTIPLE and in_profit:
        reasons_by_tier[ExitUrgency.REDUCE].append(
            f"Extensión parabólica ({atr_multiple:.1f} ATR sobre la SMA50) en una posición ya en beneficio."
        )

    # Stalled position ("stop temporal", Fase 3): capital sitting in a trade
    # that's neither worked out nor stopped out yet has a real, easy-to-miss
    # opportunity cost. Only fires once there's actually an R multiple to
    # judge progress by (no initial_stop means no R-multiple denominator -
    # nothing to call "stalled" against).
    stalled = (
        position.r_multiple is not None
        and position.r_multiple < STALLED_PROGRESS_R
        and position.bars_held >= MAX_BARS_WITHOUT_PROGRESS
        and (position.current_stop is None or price >= position.current_stop)
    )
    if stalled:
        reasons_by_tier[ExitUrgency.REDUCE].append(
            f"Posición sin progreso tras {position.bars_held} sesiones sin alcanzar +1R - capital inmovilizado."
        )

    # --- TIGHTEN_STOP: subir el stop, mantener --------------------------

    imminent_20_50 = daily.imminent_cross_20_50
    if (
        imminent_20_50 is not None
        and imminent_20_50.direction == "death"
        and imminent_20_50.r_squared >= IMMINENT_CROSS_20_50_MIN_R2
    ):
        reasons_by_tier[ExitUrgency.TIGHTEN_STOP].append(
            "Cruce de medias bajista de corto plazo (SMA20/SMA50) proyectado en "
            f"~{imminent_20_50.bars_until} sesiones (R²={imminent_20_50.r_squared:.2f})."
        )

    adx_falling = (
        adx14 is not None and adx14 < ADX_FALLING_EXIT and adx_recent_max is not None
        and adx_recent_max >= ADX_FALLING_ENTRY
    )
    if adx_falling:
        reasons_by_tier[ExitUrgency.TIGHTEN_STOP].append(
            f"ADX cayendo por debajo de {ADX_FALLING_EXIT:g} tras haber superado {ADX_FALLING_ENTRY:g} "
            "recientemente - la tendencia se está agotando."
        )

    near_resistance = (
        nearest_resistance is not None and abs(nearest_resistance.distance_pct) <= RESISTANCE_PROXIMITY
    )
    if candlestick_pattern == "bearish_engulfing" and near_resistance:
        reasons_by_tier[ExitUrgency.TIGHTEN_STOP].append(
            f"Vela envolvente bajista justo en un nivel de resistencia ({nearest_resistance.price:.2f})."
        )

    if position.r_multiple is not None and position.r_multiple > PROFIT_TIGHTEN_R:
        reasons_by_tier[ExitUrgency.TIGHTEN_STOP].append(
            f"Posición en beneficio de {position.r_multiple:.1f}R - proteger la ganancia acumulada."
        )

    # --- WATCH: lo que ya vigilaba portfolio_risk_service.py, menos lo que
    # ahora se ha promovido a un nivel superior arriba ---------------------

    near_support = nearest_support is not None and abs(nearest_support.distance_pct) <= RESISTANCE_PROXIMITY
    if near_support or (near_resistance and candlestick_pattern != "bearish_engulfing"):
        reasons_by_tier[ExitUrgency.WATCH].append(
            "Precio cerca de un nivel técnico relevante (soporte/resistencia)."
        )

    reduce_already_fired = bool(reasons_by_tier[ExitUrgency.REDUCE])
    if imminent_50_200 is not None and imminent_50_200.direction == "death" and not reduce_already_fired:
        reasons_by_tier[ExitUrgency.WATCH].append(
            "Posible cruce de medias bajista (SMA50/SMA200) proyectado, "
            "por debajo del umbral de confianza para actuar."
        )

    if candlestick_pattern == "bearish_engulfing" and not near_resistance:
        reasons_by_tier[ExitUrgency.WATCH].append("Vela envolvente bajista en la última sesión.")

    for tier in _URGENCY_SEVERITY:
        if reasons_by_tier[tier]:
            all_reasons = [r for t in _URGENCY_SEVERITY for r in reasons_by_tier[t]]
            return ExitAssessment(urgency=tier, reasons=all_reasons)

    return ExitAssessment(urgency=ExitUrgency.HOLD, reasons=["Tesis técnica intacta - sin señales de deterioro."])
