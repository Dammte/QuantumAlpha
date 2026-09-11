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
- Market regime (benchmark below its own 200-day SMA, VIX in panic/crisis) was
  tried and deliberately walked back, not shipped: an external audit correctly
  pointed out this was computed for the "Contexto" dashboard but never reached
  individual-ticker verdicts, and Meb Faber's tactical SMA-200 filter plus a
  VIX stress gate looked like an obvious, well-cited fix. `scripts/
  factor_ablation_study.py` tested it anyway rather than trusting the
  citation - and at 21/126-day horizons, on this ~217-ticker universe, buying
  *into* a benchmark-below-SMA200 or VIX-panic reading was followed by
  **better** forward returns, not worse (p<0.01 after Benjamini-Hochberg
  correction, both horizons) - a "buy the fear" pattern in an already-curated
  quality universe, not the multi-year bear-market avoidance Faber's own
  research measures (a structurally different, longer-horizon claim his
  filter was never tested against here). Confident enough that the assumed
  direction was wrong, not confident enough in the opposite direction to
  score it either (one 10-year sample dominated by a couple of sharp V-shaped
  recoveries is thin evidence for a contrarian bet) - so it isn't scored at
  all. `technical_analysis.market_regime_inputs()` and `vix_regime()` still
  exist and are still used by `MarketContextService` and the ablation study;
  they're just not wired into this function anymore. See
  `docs/quant_methodology.md` for the full writeup.

2026-09: Markov chain continuity (+-2), GARCH high-volatility regime (-1) and
the Hurst mean-reversion caution (-1) were removed from this checklist -
`markov_chain_model.py`'s bullish threshold turned out to be mathematically
unreachable and its bearish threshold effectively always true in practice
(the chain's stationary distribution is reached in one step regardless of
input, verified with synthetic AR(1) series), and `volatility_model.py`/
`statistical_structure.py` never had cross-sectional evidence behind their
weights to begin with (Nivel 3 in docs/quant_methodology.md - "plausible, sin
muestra suficiente"). See docs/quant_methodology.md for the measured writeup.
GARCH's only surviving role is picking the Chandelier Exit's volatility
bucket (`technical_analysis.volatility_regime_from_atr_percentile`, fed by an
ATR percentile instead of a per-ticker model fit) - it no longer touches this
score.

2026-09 (reconstruction, Fase 1): the fundamentals factor (revenue growth,
profit margin, leverage) is retired from this checklist too, for the same
reason as the market-regime attempt above - CLAUDE.md's rule that no weight
enters or stays without `scripts/factor_ablation_study.py` evidence behind it
was never actually applied to `REVENUE_GROWTH_STRONG`/`PROFIT_MARGIN_HEALTHY`/
`DEBT_TO_EQUITY_HIGH`; they were first-pass thresholds that were never
measured. The raw numbers themselves aren't gone - revenue growth, margin and
leverage are still shown in the ticker deep-dive's Fundamentals tab as plain
informational context - they just no longer move this score.

BUY_THRESHOLD/AVOID_THRESHOLD are first-pass values inherited from before this
audit; `scripts/factor_ablation_study.py` measures each factor's actual
marginal forward-return contribution across the full ~217-ticker universe and
should be re-run (and these thresholds revisited) whenever a factor is added,
removed, or reweighted - see that script's own docstring for methodology.
"""

from dataclasses import dataclass

from app.services.technical_analysis import PriceLevel, Stage, TrendState

# Bumped whenever the factor list or a weight changes materially - stamped
# onto every persisted RecommendationSnapshotORM row (see models.py) so a
# past verdict can always be traced back to the exact scoring logic that
# produced it, not just "some earlier version of the app". Also stamped onto
# TradePlan/PositionSignalSnapshot (see trade_plan_service.py), so this
# tracks the exit engine's own trigger set too, not only this file's score -
# bumped again for v4 even though no buy-side factor/weight changed, because
# exit_engine.py's trigger set changed materially again (see
# docs/quant_methodology.md §13: stalled-position ceiling, unified weekly
# bias with an explicit "unknown" state). v5: the fast-pair (EMA21/55) veto
# below - no existing factor/weight touched, but a "comprar" verdict can now
# come back "esperar" for a reason the score itself never carried before.
# v6: Markov/GARCH/Hurst removed from the checklist entirely (see module
# docstring) - the first step of the reconstruction toward a levels/triggers
# gate (docs/quant_methodology.md). The gate itself, when it replaces this
# checklist wholesale, gets its own version string ("...-v6-levels") rather
# than reusing this intermediate one. v7: the fundamentals factor (revenue
# growth/profit margin/leverage) removed from the checklist too - same
# never-measured-weight reasoning, see module docstring.
ENGINE_VERSION = "2026-09-audit-v7"

BUY_THRESHOLD = 5
AVOID_THRESHOLD = -3
ATR_STOP_MULTIPLE = 2.5
REWARD_RISK_RATIO = 2.0
MAX_RESISTANCE_TARGET_DISTANCE = 0.30
SUPPORT_PROXIMITY = 0.03


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
    # Cuarta auditoría, Bloque B (B-1.3): non-`None` only when a bearish
    # EMA21/55 signal downgraded what would otherwise have been "comprar" to
    # "esperar" - see `technical_analysis.detect_fast_pair_bearish_veto`.
    # Never touches `score` (the checklist's own number stays honest about
    # what it actually found) - shown separately in the UI, exactly like
    # `entry_timing`/`imminent_cross` already are, never folded into `factors`.
    veto_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StopAndTarget:
    stop_loss: float | None
    take_profit: float | None
    take_profit_method: str | None
    risk_reward: float | None


def compute_stop_and_target(
    price: float,
    atr14: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
) -> StopAndTarget:
    """The mechanical stop/target math, isolated from the buy/wait/avoid
    checklist below so it's independently reusable: place the stop just below
    the nearest support (a real technical level), but never let the risk
    exceed `ATR_STOP_MULTIPLE` x ATR (a volatility-aware ceiling) even if
    support is unusually far away; target the nearest resistance only if it
    still clears a minimum reward:risk, otherwise fall back to a fixed
    `REWARD_RISK_RATIO`:1 objective.

    Used by `build_recommendation` below (only when the verdict is
    "comprar" - a stop/target is only ever offered for a fresh buy signal),
    and independently by `trade_plan_service.py`, which runs this exact same
    math against a ticker's point-in-time history to reconstruct what a
    position's stop would have been at its actual entry date - regardless of
    what today's checklist verdict happens to say, since a stop already in
    force doesn't retroactively stop existing just because the setup no
    longer scores as a fresh "comprar" today."""
    if not atr14:
        return StopAndTarget(None, None, None, None)

    candidate_stops = [price - ATR_STOP_MULTIPLE * atr14]
    if nearest_support is not None:
        candidate_stops.append(nearest_support.price * 0.99)
    stop_loss = max(candidate_stops)  # the tighter of the two - never risk more than the ATR ceiling

    risk = price - stop_loss
    if risk <= 0:
        return StopAndTarget(stop_loss, None, None, None)

    resistance_target = None
    if nearest_resistance is not None and 0 < nearest_resistance.distance_pct <= MAX_RESISTANCE_TARGET_DISTANCE:
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

    return StopAndTarget(stop_loss, take_profit, take_profit_method, risk_reward)


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
    obv_divergence: str | None = None,
    fast_pair_bearish_signal: str | None = None,
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
    # Scored *independently* of the 8/8 AND-gate above: gating it behind
    # Minervini's other 7 criteria would zero this out for any stock that
    # fails just one unrelated criterion (e.g. RS Rating 65 vs the required
    # 70).
    #
    # CORRECTION (Segunda auditoría, Bloque 4, 2026-08) to what this comment
    # used to claim: the ORIGINAL factor_ablation_study.py run (pre-Fase 5)
    # found this "the single most robustly validated factor in the entire
    # checklist... correctly-signed at BOTH 3-month (+1.04pp) and 6-month
    # (+2.67pp)". The rewritten v2 study (triple-barrier labeling, demeaned,
    # Fase 5) - the one docs/quant_methodology.md §17 now actually surfaces
    # in the UI - measures the OPPOSITE sign for `minervini_range_position`
    # at both of those same horizons (-0.11pp @63d, not BH-significant;
    # -0.51pp @126d, significant at raw p<0.01 but not after BH correction).
    # This weight (+2) has NOT been changed - that needs the owner's explicit
    # sign-off (CLAUDE.md), not a silent fix here - but the claim this
    # comment used to make is stale and was citing a study Fase 5 already
    # superseded. See docs/quant_methodology.md §17 for the full comparison
    # across all four measured horizons before trusting this weight.
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
    # Only a caution flag outside a strong, ADX-confirmed uptrend: in a
    # genuinely strong trend RSI can (and often should) stay pinned above 80
    # for weeks while the trend keeps running - penalizing that unconditionally
    # fights the trend factors above rather than complementing them. An
    # external audit correctly flagged this exact conflict (trend-followers vs
    # oscillators disagreeing by construction) as the kind of thing a naive
    # multi-indicator checklist gets wrong.
    overbought_outside_strong_trend = (
        rsi14 is not None and rsi14 >= 80 and not (trend == TrendState.UPTREND and strong_trend)
    )
    add(
        "Sobrecompra extrema (RSI ≥ 80) fuera de una tendencia fuerte confirmada",
        -1,
        overbought_outside_strong_trend,
    )
    oversold_bounce = rsi14 is not None and rsi14 <= 30 and trend != TrendState.DOWNTREND
    add("Sobreventa (RSI ≤ 30): posible rebote técnico", 1, oversold_bounce)
    parabolic = atr_multiple is not None and atr_multiple > 4
    add("Extensión parabólica (riesgo de reversión a corto plazo)", -2, parabolic)

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

    score = sum(f.points for f in factors)

    if score >= BUY_THRESHOLD:
        verdict = "comprar"
    elif score <= AVOID_THRESHOLD:
        verdict = "evitar"
    else:
        verdict = "esperar"

    # Cuarta auditoría, Bloque B (B-1.3): a genuine bearish signal on the fast
    # EMA21/55 pair downgrades "comprar" to "esperar" - never to "evitar"
    # (the checklist itself may still show real bullish evidence; "esperar"
    # is the honest state, "the setup looks good, but not to enter right this
    # moment", not a claim that the setup itself is bad). Only ever fires
    # against a verdict that was actually "comprar" - it has nothing to add
    # to "esperar"/"evitar", which already aren't buy signals.
    veto_reason = None
    if verdict == "comprar" and fast_pair_bearish_signal is not None:
        verdict = "esperar"
        veto_reason = fast_pair_bearish_signal

    stop_loss = take_profit = take_profit_method = risk_reward = None
    if verdict == "comprar":
        stop_target = compute_stop_and_target(price, atr14, nearest_support, nearest_resistance)
        stop_loss = stop_target.stop_loss
        take_profit = stop_target.take_profit
        take_profit_method = stop_target.take_profit_method
        risk_reward = stop_target.risk_reward

    return Recommendation(
        verdict=verdict,
        score=score,
        factors=factors,
        stop_loss=stop_loss,
        take_profit=take_profit,
        take_profit_method=take_profit_method,
        risk_reward=risk_reward,
        veto_reason=veto_reason,
    )
