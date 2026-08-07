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

**Ensemble design - what each factor actually contributes, so they reinforce
rather than duplicate each other** (audited 2026-08, see
`docs/quant_methodology.md` for the full writeup):

- Trend/stage/RS/Minervini all touch the same underlying "is this in a
  confirmed uptrend" fact from different angles. Trend (MA ordering), Stage
  (Weinstein's slower-moving stage-of-the-cycle read) and RS Rating
  (cross-sectional strength vs the universe) are kept as independently scored
  factors because each is genuinely a different *lens* on trend quality.
  Minervini's 8-point Trend Template is mostly a re-statement of those three
  (6 of its 8 criteria) - only its 52-week-range-position criteria add new
  information (a confirmed move, not a dead-cat bounce off the low or one
  already fully extended off the low). Its point value is deliberately kept
  small (a confirmation bonus, not another full vote) precisely to avoid one
  underlying "clean uptrend" fact getting counted four times.
- ADX/DMI, RSI extremes and the ATR-extension gauge each measure a genuinely
  different axis (trend *strength/conviction*, momentum *exhaustion*, and
  *overextension* vs the moving average) that plain trend/stage classification
  can miss entirely (e.g. a technically-uptrending but choppy, low-ADX stock).
- OBV divergence is the only volume-based factor - a second, independent data
  source (participation, not just price) that price-only indicators cannot
  see by construction (Wyckoff's "effort vs result").
- Markov/GARCH are statistically gated (a chain forecast is only credited when
  a runs test says the ticker's own history isn't indistinguishable from
  noise) - independent of, and structurally different from, the rule-based
  checklist above.
- Fundamentals (revenue growth, profit margin, leverage) are the only factors
  that don't come from price/volume at all - a CANSLIM/quality-factor-style
  check that a technical setup is backed by a business that's actually
  growing and solvent, not just a chart pattern.

BUY_THRESHOLD/AVOID_THRESHOLD are first-pass values inherited from before this
audit; `scripts/factor_ablation_study.py` measures each factor's actual
marginal forward-return contribution across the full ~217-ticker universe and
should be re-run (and these thresholds revisited) whenever a factor is added,
removed, or reweighted - see that script's own docstring for methodology.
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
REVENUE_GROWTH_STRONG = 0.15
PROFIT_MARGIN_HEALTHY = 0.15
DEBT_TO_EQUITY_HIGH = 200.0


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
    minervini_range_confirmed: bool = False,
    markov: MarkovChainResult | None = None,
    garch: GarchResult | None = None,
    obv_divergence: str | None = None,
    revenue_growth: float | None = None,
    profit_margins: float | None = None,
    debt_to_equity: float | None = None,
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
    # Kept at +1, not +2: 6 of its 8 criteria already re-state trend/stage/RS,
    # already scored above - this is a small confirmation bonus for passing
    # ALL 8 gates simultaneously (multi-confirmation has some ensemble value
    # even when individual pieces overlap), not a second full vote for the
    # same underlying "clean uptrend" fact. See module docstring.
    add("Cumple el Trend Template de Minervini (8/8)", 1, minervini_pass)
    # Scored *independently* of the 8/8 AND-gate above, and weighted higher:
    # scripts/factor_ablation_study.py, run cross-sectionally over ~217
    # tickers x 10 years (2026-08), found this specific criterion - price 25%+
    # above its 52-week low AND within 25% of its 52-week high, i.e. a
    # confirmed move, neither a dead-cat bounce off the low nor one already
    # fully extended off it - to be the single most robustly validated factor
    # in the entire checklist: significant (p<0.01, permutation test) and
    # correctly-signed at BOTH a 3-month (+1.04pp) and 6-month (+2.67pp)
    # forward-return horizon, the only factor to clear that bar at either
    # horizon. Gating it behind Minervini's other 7 criteria (as the 8/8
    # bonus above does) would silently zero out this validated edge for any
    # stock that fails just one unrelated criterion (e.g. RS Rating 65 vs the
    # required 70) - seeing this file? recompute this study before trusting
    # this weight blindly on a materially different universe/period.
    add(
        "Movimiento confirmado: precio 25%+ sobre su mínimo anual y dentro del 25% de su máximo anual",
        2,
        minervini_range_confirmed,
    )
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

    # Wyckoff "effort vs result": price near a range high/low without real
    # volume behind it - the only participation-based (not price-derived)
    # factor in the checklist. See `technical_analysis.obv_divergence`.
    add(
        "Divergencia bajista de volumen (OBV): el avance no está respaldado por compras reales",
        -2,
        obv_divergence == "bearish",
    )
    add(
        "Divergencia alcista de volumen (OBV): la presión vendedora se agota pese a la caída de precio",
        1,
        obv_divergence == "bullish",
    )

    # Fundamentals: the only non-price/volume factors. Kept optional (None by
    # default) and deliberately NOT wired into the portfolio-risk/premium-
    # watchlist hot paths (see ticker_analysis_service.py) - an extra
    # per-ticker network call there re-introduces exactly the N-ticker
    # latency problem already found and fixed once in production (see
    # PortfolioRiskService's docstring). Only the single-ticker "Analizar
    # activo" deep dive, which already pays for a fundamentals fetch, feeds
    # these in.
    add(
        "Crecimiento de ingresos sólido (≥15% interanual)",
        1,
        revenue_growth is not None and revenue_growth >= REVENUE_GROWTH_STRONG,
    )
    add(
        "Ingresos en contracción (crecimiento interanual negativo)",
        -1,
        revenue_growth is not None and revenue_growth < 0,
    )
    add(
        "Margen neto saludable (≥15%)",
        1,
        profit_margins is not None and profit_margins >= PROFIT_MARGIN_HEALTHY,
    )
    add("Empresa no rentable (margen neto negativo)", -1, profit_margins is not None and profit_margins < 0)
    add(
        "Apalancamiento elevado (deuda/patrimonio > 200%)",
        -1,
        debt_to_equity is not None and debt_to_equity > DEBT_TO_EQUITY_HIGH,
    )

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
