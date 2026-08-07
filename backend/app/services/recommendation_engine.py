"""Turns the technical snapshot of a single ticker into a transparent, rule-based
buy/wait/avoid verdict with a suggested stop-loss and price target.

This is deliberately a simple weighted checklist, not a black box: every point
added or subtracted is returned as a labeled factor, so the "why" behind the
verdict is always visible to the user making the final call. Consistent with
the app's price-action philosophy (protect capital first), a stop-loss is only
offered when the verdict is "comprar", and it's built from two ideas at once:
place it just below the nearest support (a real, technical level), but never
let the risk exceed a multiple of ATR (a volatility-aware ceiling) even if the
nearest support is unusually far away.
"""

from dataclasses import dataclass

from app.services.markov_chain_model import MarkovChainResult
from app.services.technical_analysis import PriceLevel, Stage, TrendState
from app.services.volatility_model import GarchResult

BUY_THRESHOLD = 5
AVOID_THRESHOLD = -3
ATR_STOP_MULTIPLE = 2.5
REWARD_RISK_RATIO = 2.0
MAX_RESISTANCE_TARGET_DISTANCE = 0.30
SUPPORT_PROXIMITY = 0.03
MARKOV_BULLISH_THRESHOLD = 0.55
MARKOV_BEARISH_THRESHOLD = 0.45


@dataclass(frozen=True, slots=True)
class RecommendationFactor:
    label: str
    points: int
    triggered: bool


@dataclass(frozen=True, slots=True)
class Recommendation:
    verdict: str  # "comprar" | "esperar" | "evitar"
    score: int
    factors: list[RecommendationFactor]
    stop_loss: float | None
    take_profit: float | None
    take_profit_method: str | None
    risk_reward: float | None


def build_recommendation(
    price: float,
    trend: TrendState,
    stage: Stage | None,
    ma_cross: str | None,
    rsi14: float | None,
    adx14: float | None,
    plus_di: float | None,
    minus_di: float | None,
    atr14: float | None,
    atr_multiple: float | None,
    rs_rating: int | None,
    minervini_pass: bool,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    markov: MarkovChainResult | None = None,
    garch: GarchResult | None = None,
) -> Recommendation:
    factors: list[RecommendationFactor] = []

    def add(label: str, points: int, triggered: bool) -> None:
        factors.append(RecommendationFactor(label=label, points=points if triggered else 0, triggered=triggered))

    strong_trend = (
        adx14 is not None and adx14 >= 25 and plus_di is not None and minus_di is not None and plus_di > minus_di
    )
    near_support = nearest_support is not None and abs(nearest_support.distance_pct) <= SUPPORT_PROXIMITY

    add("Tendencia alcista (MA20 > MA50 > MA200)", 2, trend == TrendState.UPTREND)
    add("Tendencia bajista - evitar entradas largas", -3, trend == TrendState.DOWNTREND)
    add("Fase 2 de Weinstein (avance)", 2, stage == Stage.STAGE_2)
    add("Fase 4 de Weinstein (declive)", -3, stage == Stage.STAGE_4)
    add("Cumple el Trend Template de Minervini (8/8)", 2, minervini_pass)
    add("Golden cross reciente (MA50/MA200)", 1, ma_cross == "golden")
    add("Death cross reciente (MA50/MA200)", -2, ma_cross == "death")
    add("RS Rating alto (≥ 80): líder de mercado", 2, rs_rating is not None and rs_rating >= 80)
    add("RS Rating bajo (< 30): rezagado", -1, rs_rating is not None and rs_rating < 30)
    add("Tendencia fuerte y confirmada (ADX ≥ 25, +DI > -DI)", 1, strong_trend)
    add("Rebotando en un soporte cercano", 1, near_support and trend == TrendState.UPTREND)
    add("Sobrecompra extrema (RSI ≥ 80)", -1, rsi14 is not None and rsi14 >= 80)
    oversold_bounce = rsi14 is not None and rsi14 <= 30 and trend != TrendState.DOWNTREND
    add("Sobreventa (RSI ≤ 30): posible rebote técnico", 1, oversold_bounce)
    parabolic = atr_multiple is not None and atr_multiple > 4
    add("Extensión parabólica (riesgo de reversión a corto plazo)", -2, parabolic)

    # Only credited when the runs test says the ticker's own up/down sequence is
    # NOT statistically indistinguishable from iid noise - a Markov forecast on a
    # sequence that looks random carries no real edge, so it's excluded rather
    # than presented as a signal.
    markov_sequence_has_structure = markov is not None and not markov.sequence_looks_random
    markov_bullish = markov_sequence_has_structure and markov.prob_bullish_21d >= MARKOV_BULLISH_THRESHOLD
    markov_bearish = markov_sequence_has_structure and markov.prob_bullish_21d <= MARKOV_BEARISH_THRESHOLD
    add("Cadena de Markov: continuidad alcista probable (secuencia no aleatoria)", 2, markov_bullish)
    add("Cadena de Markov: continuidad bajista probable (secuencia no aleatoria)", -2, markov_bearish)

    high_vol_regime = garch is not None and garch.regime == "alta"
    add("Volatilidad condicional elevada (GARCH, percentil ≥75 de su propio historial)", -1, high_vol_regime)

    score = sum(f.points for f in factors)

    if score >= BUY_THRESHOLD:
        verdict = "comprar"
    elif score <= AVOID_THRESHOLD:
        verdict = "evitar"
    else:
        verdict = "esperar"

    stop_loss = None
    take_profit = None
    take_profit_method = None
    risk_reward = None

    if verdict == "comprar" and atr14:
        candidate_stops = [price - ATR_STOP_MULTIPLE * atr14]
        if nearest_support is not None:
            candidate_stops.append(nearest_support.price * 0.99)
        stop_loss = max(candidate_stops)  # the tighter of the two - never risk more than the ATR ceiling

        risk = price - stop_loss
        if risk > 0:
            resistance_target = None
            if (
                nearest_resistance is not None
                and 0 < nearest_resistance.distance_pct <= MAX_RESISTANCE_TARGET_DISTANCE
            ):
                resistance_target = nearest_resistance.price

            # Only use the nearby resistance as the target if it still clears a
            # minimum reward:risk - a resistance sitting right on top of the entry
            # makes for a bad trade even when the technical setup itself is strong,
            # so fall back to the fixed 2:1 objective instead of proposing a buy
            # with unfavorable asymmetry.
            if resistance_target is not None and (resistance_target - price) / risk >= 1.0:
                take_profit = resistance_target
                take_profit_method = "resistencia más cercana"
            else:
                take_profit = price + REWARD_RISK_RATIO * risk
                take_profit_method = f"objetivo {REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"
            risk_reward = (take_profit - price) / risk

    return Recommendation(
        verdict=verdict,
        score=score,
        factors=factors,
        stop_loss=stop_loss,
        take_profit=take_profit,
        take_profit_method=take_profit_method,
        risk_reward=risk_reward,
    )
