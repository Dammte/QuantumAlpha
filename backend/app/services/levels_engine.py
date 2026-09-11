"""Reconstruction (2026-09), Fase 3: the boolean gate that replaces
`recommendation_engine.py`'s weighted checklist for the question "is this
specific entry good" (Parte 0, pregunta 3 of the reconstruction brief). See
`trade_geometry.py`'s docstring for the "at what price" half of the same
Fase, and for the provenance note (author's best-effort reconstruction,
2026-09-11 owner sign-off) that applies equally to this file.

Why a gate instead of a score: a weighted sum can pass with a strong RS
Rating compensating for an ugly, overextended entry, or fail by one point on
an otherwise clean setup - exactly the "no veo por qué" opacity the checklist
approach was retired for. Every condition below is a hard AND instead: all
must hold or the gate simply doesn't pass, and which one(s) failed is always
visible (never collapsed to a bare yes/no) - the same transparency principle
`recommendation_engine.RecommendationFactor` and
`exit_engine.ExitAssessment.reasons` already established, just without a
point value attached to any single condition.

**Which conditions made the cut, and why - deliberately narrow, not a
re-scoring of the retired checklist's every factor as a gate:**
- Trend/stage (uptrend or Stage 2) - the same underlying "is this actually in
  an advance" fact `recommendation_engine.py` used to score from three
  separate angles (trend, stage, RS Rating); here it's one structural gate,
  not three separate votes for the same thing.
- Not parabolic (ATR multiple) and not overbought outside a confirmed strong
  trend - both were *risk* factors in the old checklist (protect capital
  first - CLAUDE.md). Kept as hard gates rather than points on purpose: a
  genuinely overextended entry shouldn't be rescued by an unrelated strength
  reading elsewhere.
- No bearish OBV divergence - was already the checklist's only
  volume-based (participation, not price) factor; promoted to a hard gate
  here for the same "don't buy a move real buying isn't backing" reasoning,
  now applied at entry instead of only as a scored penalty.
- No bearish fast-pair (EMA21/55) veto - already a hard override in the old
  checklist (never just a point value), unchanged here.
- A minimum reward:risk on the computed stop/target - new: the old checklist
  gated on a buy/wait/avoid verdict but never separately on trade *quality*
  once a stop/target existed. A technically clean setup with a poor
  reward:risk is still a bad trade to actually take.

RS Rating and Minervini's 8/8 - the old checklist's two most heavily-weighted
"is this a leader" factors - are deliberately NOT hard gates here: a
genuinely good setup on a name that hasn't yet earned a high RS Rating (a
fresh breakout, an early Stage 2) is exactly the kind of entry this rebuild
is supposed to still catch, not exclude by construction. Both stay visible
as context on the ticker's own precomputed state (Fase 2's
`TickerDailyState`) without gating the trigger itself.

This split is a first-pass judgment call, not a measured one -
`GATE_VERSION` follows the same versioning discipline
`recommendation_engine.ENGINE_VERSION` established (stamped wherever a gate
result is persisted, so a past verdict is always traceable to the exact
logic that produced it), and Fase 8 retargets `scripts/factor_ablation_study.py`
at trigger outcomes precisely so this split can be revisited with evidence
instead of intuition.
"""

from dataclasses import dataclass

from app.services.technical_analysis import PriceLevel, Stage, TrendState
from app.services.trade_geometry import EntryTrigger, StopAndTarget, compute_entry_trigger, compute_stop_and_target

# Bumped whenever a gate condition or threshold changes materially - same
# discipline as recommendation_engine.ENGINE_VERSION, its own separate
# version string (see that module's docstring for why this isn't the same
# constant).
GATE_VERSION = "2026-09-levels-v1"

# Same bar recommendation_engine.py's own "parabolic" risk factor used.
EXTENDED_ATR_MULTIPLE = 4.0
MIN_REWARD_RISK = 1.5


@dataclass(frozen=True, slots=True)
class GateCondition:
    label: str
    passed: bool


@dataclass(frozen=True, slots=True)
class GateResult:
    """`passes` is the single boolean the Radar/Screener (Fase 6) filters on;
    `conditions` is what makes that answer auditable instead of a black box -
    every condition evaluated, not just the one(s) that failed. `entry_trigger`
    can be `None` even when `passes` is `True`: a clean, tradeable setup with
    no support/resistance level close enough to define an imminent watch
    price yet is a real, if less actionable-today, state - see
    `trade_geometry.compute_entry_trigger`."""

    passes: bool
    conditions: list[GateCondition]
    entry_trigger: EntryTrigger | None
    stop_and_target: StopAndTarget | None


def evaluate_gate(
    price: float,
    trend: TrendState,
    stage: Stage | None,
    rsi14: float | None,
    adx14: float | None,
    plus_di: float | None,
    minus_di: float | None,
    atr14: float | None,
    atr_multiple: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    obv_divergence: str | None = None,
    fast_pair_bearish_signal: str | None = None,
) -> GateResult:
    conditions: list[GateCondition] = []

    def add(label: str, passed: bool) -> None:
        conditions.append(GateCondition(label=label, passed=passed))

    strong_trend = (
        adx14 is not None and adx14 >= 25 and plus_di is not None and minus_di is not None and plus_di > minus_di
    )

    add("Tendencia alcista o Fase 2 de Weinstein", trend == TrendState.UPTREND or stage == Stage.STAGE_2)

    parabolic = atr_multiple is not None and atr_multiple > EXTENDED_ATR_MULTIPLE
    add("Sin extensión parabólica (ATR múltiplo <= 4)", not parabolic)

    overbought_outside_strong_trend = (
        rsi14 is not None and rsi14 >= 80 and not (trend == TrendState.UPTREND and strong_trend)
    )
    add(
        "Sin sobrecompra extrema (RSI >= 80) fuera de tendencia fuerte confirmada",
        not overbought_outside_strong_trend,
    )

    add("Sin divergencia bajista de volumen (OBV)", obv_divergence != "bearish")

    add("Sin veto bajista del par rápido (EMA21/55)", fast_pair_bearish_signal is None)

    stop_and_target = compute_stop_and_target(price, atr14, nearest_support, nearest_resistance)
    reward_risk_ok = stop_and_target.risk_reward is not None and stop_and_target.risk_reward >= MIN_REWARD_RISK
    add(f"Relación beneficio:riesgo >= {MIN_REWARD_RISK:.1f}", reward_risk_ok)

    entry_trigger = compute_entry_trigger(price, nearest_support, nearest_resistance)

    return GateResult(
        passes=all(c.passed for c in conditions),
        conditions=conditions,
        entry_trigger=entry_trigger,
        stop_and_target=stop_and_target,
    )
