"""Reconstruction (2026-09), Fase 3: the "where" half of the levels/triggers
system the propietario commissioned to replace `recommendation_engine.py`'s
weighted checklist (see docs/quant_methodology.md and that module's own
docstring for the full history of what led here). Pure geometry, no verdict:
given a ticker's already-computed support/resistance levels and volatility,
at what exact price would a valid entry actually confirm, and what would it
cost to protect it once taken.

This module answers "what's about to trigger entry" (Parte 0, pregunta 2 of
the reconstruction brief) at the price level; `levels_engine.py` (same Fase)
adds the pass/fail gate on top of it (pregunta 3 - "is this specific entry
good"). `watchlist_service.py`'s own setup detectors - which *pattern*
(breakout, pullback, oversold bounce, trend continuation) a ticker matches -
were a separate concern this module never touched; that module retired
2026-09 (docs/quant_methodology.md §25), `GET /market/radar` now owns "which
pattern"/"is it good" together. This module still only answers "at what
price", never either of those.

Provenance note (2026-09-11): this file, `levels_engine.py`, and the tables/
jobs Fase 2 builds around them are the author's best-effort reconstruction of
Parte 3+ of the original brief, written after the literal spec text had
already scrolled out of context - the propietario explicitly signed off on
proceeding this way (see the session's own record) rather than re-pasting it.
Every threshold below is a first-pass value in the same spirit
`recommendation_engine.BUY_THRESHOLD`/`exit_engine.IMMINENT_CROSS_*` already
shipped as - not yet run through `scripts/factor_ablation_study.py` (Fase 8
retargets that script at trigger outcomes instead of the old score) - so
treat the exact numbers as adjustable, not as measured fact.

2026-09 (later pass, once the literal brief text was back in hand): the
above approximation shipped `compute_stop_and_target` - one fixed ATR
multiple, one fixed 2:1 target - never the real Parte 7 design (a stop
*cascade* by entry type, an ATR-percentile-adaptive risk ceiling, fixed-risk
sizing with three limits, cost-net reward:risk). `compute_trade_geometry`/
`TradeGeometry`/`EntryType` below are that real design, added *alongside*
the original function - not a replacement. `compute_stop_and_target` keeps
every existing caller (`levels_engine.evaluate_gate`, `trade_plan_service.py`,
`scripts/factor_ablation_study.py`'s re-export) working unchanged; wiring
the richer function into those call sites is deliberately left for its own
follow-up pass (`app.core.trading_params` already carries every constant
this function needs, so that migration is data plumbing, not new design).
"""

from dataclasses import dataclass
from enum import Enum

from app.core.trading_params import (
    MAX_POSITION_PCT,
    MIN_POSITION_USD,
    MIN_RISK_REWARD_NET,
    RISK_CEILING_ATR_MULTIPLE,
    RISK_CEILING_MAX_PCT,
    RISK_CEILING_MIN_PCT,
    RISK_PER_TRADE_PCT,
    STOP_ATR_CEILING,
    TRANSACTION_COST_PCT,
)
from app.services.technical_analysis import PriceLevel, TrendState

# A raw touch of the exact pivot price is noise, not confirmation - the
# breakout trigger sits a small buffer above the resistance itself. Smaller
# than support_resistance_levels' own 1.5% clustering tolerance on purpose:
# this buffer is about confirming a *break*, not about whether two pivots are
# "the same level".
BREAKOUT_BUFFER_PCT = 0.003  # 0.3% above the resistance pivot

# A resistance further than this isn't "about to" trigger - it's background
# context for the deep-dive view, not a Radar candidate. First-pass value,
# see module docstring.
BREAKOUT_TRIGGER_MAX_DISTANCE = 0.08  # 8%

# Same "close enough to matter" bar recommendation_engine.py's own
# near_support factor used before this reconstruction.
PULLBACK_PROXIMITY_PCT = 0.03

ATR_STOP_MULTIPLE = 2.5
REWARD_RISK_RATIO = 2.0
MAX_RESISTANCE_TARGET_DISTANCE = 0.30


@dataclass(frozen=True, slots=True)
class EntryTrigger:
    """One concrete, watchable price level - not a verdict. `already_triggered`
    tells a caller whether *today's* price has already crossed it (a breakout
    already confirmed, a pullback already at support) versus still being
    something to wait and watch for."""

    trigger_type: str  # "breakout" | "pullback_bounce"
    trigger_price: float
    already_triggered: bool


@dataclass(frozen=True, slots=True)
class StopAndTarget:
    stop_loss: float | None
    take_profit: float | None
    take_profit_method: str | None
    risk_reward: float | None


def compute_entry_trigger(
    price: float,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
) -> EntryTrigger | None:
    """The single most relevant "what would make this a buy" price level for
    a ticker right now, or `None` when neither a nearby support nor a nearby
    resistance makes for a plausible near-term trigger.

    Pullback takes priority over breakout when both are technically present
    (rare, but possible on a wide multi-level chart): a price already sitting
    at support is an actionable trigger *today*, while a distant breakout is
    still just something to watch for - the more immediate answer wins.
    """
    if nearest_support is not None and abs(nearest_support.distance_pct) <= PULLBACK_PROXIMITY_PCT:
        return EntryTrigger(
            trigger_type="pullback_bounce",
            trigger_price=nearest_support.price,
            already_triggered=True,
        )
    if (
        nearest_resistance is not None
        and 0 < nearest_resistance.distance_pct <= BREAKOUT_TRIGGER_MAX_DISTANCE
    ):
        trigger_price = nearest_resistance.price * (1 + BREAKOUT_BUFFER_PCT)
        return EntryTrigger(
            trigger_type="breakout",
            trigger_price=trigger_price,
            already_triggered=price >= trigger_price,
        )
    return None


def compute_stop_and_target(
    price: float,
    atr14: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
) -> StopAndTarget:
    """The mechanical stop/target math - moved here unchanged from
    `recommendation_engine.compute_stop_and_target` (same function, same
    behavior, new home alongside the rest of the levels/triggers geometry).
    Place the stop just below the nearest support (a real technical level),
    but never let the risk exceed `ATR_STOP_MULTIPLE` x ATR (a
    volatility-aware ceiling) even if support is unusually far away; target
    the nearest resistance only if it still clears a minimum reward:risk,
    otherwise fall back to a fixed `REWARD_RISK_RATIO`:1 objective.

    Used by `levels_engine.evaluate_gate` below, and independently by
    `trade_plan_service.py`, which runs this exact same math against a
    ticker's point-in-time history to reconstruct what a position's stop
    would have been at its actual entry date - regardless of what today's
    gate says, since a stop already in force doesn't retroactively stop
    existing just because the setup no longer passes the gate today."""
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

    if resistance_target is not None and (resistance_target - price) / risk >= 1.0:
        take_profit = resistance_target
        take_profit_method = "resistencia más cercana"
    else:
        take_profit = price + REWARD_RISK_RATIO * risk
        take_profit_method = f"objetivo {REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"
    risk_reward = (take_profit - price) / risk

    return StopAndTarget(stop_loss, take_profit, take_profit_method, risk_reward)


# --- Parte 7 (real design): stop cascade + adaptive risk ceiling + sizing ----

# Cushion below the broken/tested level, in ATR - smaller for the two
# "at a real level" rungs (breakout/bounce), larger for the two MA-based
# rungs (pullback/continuation), matching Parte 7's own per-rung typical
# ranges (a raw MA is a fuzzier reference than a pivot, so it earns a wider
# cushion before calling the stop "under" it).
LEVEL_STOP_CUSHION_ATR = 0.3
MA_STOP_CUSHION_ATR = 0.4


class EntryType(str, Enum):
    """Which of Parte 7's four stop-cascade rungs actually produced this
    geometry's stop - kept on the result, not just used internally, so a
    caller/UI can show *why* the stop sits where it does (the same
    transparency principle `stop_basis`'s own prose text already gives, just
    as a stable enum instead of a string to match on)."""

    BREAKOUT = "breakout"
    PULLBACK_SUPPORT = "pullback_support"
    PULLBACK_EMA21 = "pullback_ema21"
    CONTINUATION_EMA55 = "continuation_ema55"


@dataclass(frozen=True, slots=True)
class TradeGeometry:
    """The full Parte 7 read for one potential entry: not just a stop/target
    (`StopAndTarget` above), but the complete chain from "where's the stop"
    to "how many shares" to "is this even worth taking" - every step visible,
    never collapsed to a bare yes/no. `viable=False` is a normal, frequent
    result (Parte 0's "empty list is valid"), never an error - `rejection_reason`
    says which specific check failed; every other field before that check is
    still populated where computable, so a caller can show *how close* a
    rejected setup came, not just that it was rejected."""

    entry_price: float
    stop_price: float | None
    stop_basis: str | None  # Spanish prose: which rung, and whether the hard ceiling capped it
    entry_type: EntryType | None
    risk_pct: float | None  # (entry - stop) / entry
    risk_atr: float | None  # (entry - stop) / atr14
    risk_ceiling_pct: float | None  # the adaptive ceiling this trade was measured against
    target_price: float | None
    target_basis: str | None  # "resistencia más cercana" | "objetivo N:1 sobre el riesgo"
    reward_pct: float | None  # (target - entry) / entry
    risk_reward_gross: float | None
    risk_reward_net: float | None  # net of round-trip TRANSACTION_COST_PCT - what's actually shown
    shares_for_risk_budget: float | None
    position_value: float | None
    pct_of_portfolio: float | None  # position_value / capital_total
    viable: bool
    rejection_reason: str | None  # Spanish, only when viable is False


def _not_viable(entry_price: float, reason: str, **known: float | None) -> TradeGeometry:
    """Builds a rejected `TradeGeometry`, filling in whatever the caller
    already knows (Parte 0: showing *why* and *how close* beats a bare
    rejection) and `None`/`False` for everything downstream of the failure."""
    fields = dict(
        entry_price=entry_price, stop_price=None, stop_basis=None, entry_type=None,
        risk_pct=None, risk_atr=None, risk_ceiling_pct=None,
        target_price=None, target_basis=None, reward_pct=None,
        risk_reward_gross=None, risk_reward_net=None,
        shares_for_risk_budget=None, position_value=None, pct_of_portfolio=None,
    )
    fields.update(known)
    return TradeGeometry(**fields, viable=False, rejection_reason=reason)


def _stop_cascade(
    price: float,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    ema21: float | None,
    ema55: float | None,
    trend: TrendState,
) -> tuple[float, EntryType, str] | None:
    """The first applicable rung, in the exact order Parte 7 specifies -
    `(level_price, entry_type, basis_label)`, or `None` when nothing in the
    cascade applies (no nearby level, no EMA reads, not in an uptrend for the
    two MA-based rungs). The hard 2.0 ATR ceiling is applied by the caller,
    uniformly across every rung, not duplicated here.

    1. Breakout: `nearest_resistance` already cleared (mirrors
       `compute_entry_trigger`'s own breakout confirmation) - the broken
       level now acts as support.
    2. Bounce: price sitting at `nearest_support` (same `PULLBACK_PROXIMITY_PCT`
       proximity `compute_entry_trigger` uses for a pullback trigger).
    3. Pullback to EMA21: only in an uptrend, price at or just above EMA21.
    4. Continuation over EMA55: only in an uptrend, price above EMA55 - the
       last-resort rung once nothing more specific applies (e.g. a trending
       stock with no nearby pivot and already past its EMA21)."""
    if nearest_resistance is not None:
        breakout_trigger = nearest_resistance.price * (1 + BREAKOUT_BUFFER_PCT)
        if price >= breakout_trigger:
            return (
                nearest_resistance.price,
                EntryType.BREAKOUT,
                f"bajo el nivel de resistencia roto en {nearest_resistance.price:.2f}",
            )
    if nearest_support is not None and abs(nearest_support.distance_pct) <= PULLBACK_PROXIMITY_PCT:
        return (
            nearest_support.price,
            EntryType.PULLBACK_SUPPORT,
            f"bajo el soporte en {nearest_support.price:.2f}",
        )
    if trend == TrendState.UPTREND and ema21 is not None and price >= ema21:
        if (price - ema21) / price <= PULLBACK_PROXIMITY_PCT:
            return (ema21, EntryType.PULLBACK_EMA21, f"bajo la EMA21 ({ema21:.2f})")
    if trend == TrendState.UPTREND and ema55 is not None and price > ema55:
        return (ema55, EntryType.CONTINUATION_EMA55, f"bajo la EMA55 ({ema55:.2f})")
    return None


def compute_trade_geometry(
    price: float,
    atr14: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    ema21: float | None,
    ema55: float | None,
    trend: TrendState,
    capital_total: float,
    atr_percentile_252: float | None = None,
) -> TradeGeometry:
    """The real Parte 7 pipeline, in order - each step can end the chain with
    `viable=False`, never by silently widening the stop or stretching the
    target to force a "yes" (Parte 18's own anti-pattern list names this
    explicitly): (1) pick the *natural* (uncapped) stop off the cascade
    above; (2) reject if the resulting risk exceeds the ATR-percentile-
    adaptive ceiling (`RISK_CEILING_ATR_MULTIPLE * atr_pct`, clamped to
    `[RISK_CEILING_MIN_PCT, RISK_CEILING_MAX_PCT]`) - checked against the
    *natural* distance, deliberately before the hard ATR cap below, so a
    stop that's merely far (but survives the risk-ceiling check) isn't
    rejected for a reason it didn't actually trigger, and one that's simply
    too risky is rejected with the real, uncapped number in the reason, not
    an already-shrunk one (Parte 15/20's own worked examples - a calm 1.2%-
    ATR name's natural stop rejected at its true, uncapped percentage - only
    come out right in this order); (3) only once that passes, cap the stop
    at `STOP_ATR_CEILING` if the natural one asked for more (this can only
    ever *shrink* the risk further, so it never re-triggers step 2); (4)
    target the nearest resistance if its *net*-of-cost reward:risk clears
    `MIN_RISK_REWARD_NET`, else the fixed `REWARD_RISK_RATIO`:1 objective,
    else reject as "too close"; (5) size for `RISK_PER_TRADE_PCT` of
    `capital_total`, halved once `atr_percentile_252 >= 0.85` (an unusually
    volatile stretch for this specific ticker), capped at `MAX_POSITION_PCT`
    of capital, rejected outright under `MIN_POSITION_USD` (a position that
    small lets transaction costs eat the trade). The 6% aggregate-risk-
    across-all-positions cap from the brief is deliberately NOT enforced
    here - that needs every other open position's own risk, which is
    `portfolio_construction_service.final_position_size`'s job, one layer up
    from a single ticker's own geometry."""
    if not atr14 or atr14 <= 0:
        return _not_viable(price, "ATR no disponible - no se puede definir un stop con base de volatilidad")

    cascade = _stop_cascade(price, nearest_support, nearest_resistance, ema21, ema55, trend)
    if cascade is None:
        return _not_viable(price, "sin nivel de referencia (soporte/resistencia/EMA21/EMA55) para anclar el stop")

    level_price, entry_type, basis_label = cascade
    at_a_real_level = entry_type in (EntryType.BREAKOUT, EntryType.PULLBACK_SUPPORT)
    cushion = LEVEL_STOP_CUSHION_ATR if at_a_real_level else MA_STOP_CUSHION_ATR
    raw_stop = level_price - cushion * atr14
    raw_risk_per_share = price - raw_stop
    if raw_risk_per_share <= 0:
        return _not_viable(
            price, "el nivel de referencia queda por encima del precio - stop no válido",
            entry_type=entry_type,
        )

    raw_risk_pct = raw_risk_per_share / price
    atr_pct = atr14 / price
    risk_ceiling_pct = min(max(RISK_CEILING_ATR_MULTIPLE * atr_pct, RISK_CEILING_MIN_PCT), RISK_CEILING_MAX_PCT)

    if raw_risk_pct > risk_ceiling_pct:
        reason = (
            f"riesgo natural ({raw_risk_pct:.1%}) excede el techo adaptativo "
            f"({risk_ceiling_pct:.0%}) para esta volatilidad"
        )
        return _not_viable(
            price, reason,
            entry_type=entry_type, risk_pct=raw_risk_pct, risk_atr=raw_risk_per_share / atr14,
            risk_ceiling_pct=risk_ceiling_pct,
        )

    if raw_risk_per_share / atr14 > STOP_ATR_CEILING:
        stop_price = price - STOP_ATR_CEILING * atr14
        stop_basis = f"{basis_label} - excede el techo de {STOP_ATR_CEILING:.1f} ATR, stop no bajo el nivel"
    else:
        stop_price = raw_stop
        stop_basis = basis_label

    risk_per_share = price - stop_price
    risk_pct = risk_per_share / price
    risk_atr = risk_per_share / atr14

    round_trip_cost_pct = 2 * TRANSACTION_COST_PCT

    def _net_reward_risk(target_price: float) -> tuple[float, float]:
        gross_reward = target_price - price
        gross = gross_reward / risk_per_share
        net_reward_pct = (gross_reward / price) - round_trip_cost_pct
        net = (net_reward_pct * price) / risk_per_share
        return gross, net

    target_price = target_basis = None
    risk_reward_gross = risk_reward_net = None
    if nearest_resistance is not None and nearest_resistance.price > price:
        gross, net = _net_reward_risk(nearest_resistance.price)
        if net >= MIN_RISK_REWARD_NET:
            target_price, target_basis = nearest_resistance.price, "resistencia más cercana"
            risk_reward_gross, risk_reward_net = gross, net

    if target_price is None:
        fixed_target = price + REWARD_RISK_RATIO * risk_per_share
        gross, net = _net_reward_risk(fixed_target)
        if net >= MIN_RISK_REWARD_NET:
            target_price, target_basis = fixed_target, f"objetivo {REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"
            risk_reward_gross, risk_reward_net = gross, net
        else:
            reason = (
                "resistencia demasiado cerca - ni ella ni el objetivo fijo alcanzan un "
                "beneficio:riesgo mínimo"
                if nearest_resistance is not None
                else "riesgo por acción demasiado pequeño - el objetivo fijo no compensa los costes"
            )
            return _not_viable(
                price, reason,
                stop_price=stop_price, stop_basis=stop_basis, entry_type=entry_type,
                risk_pct=risk_pct, risk_atr=risk_atr, risk_ceiling_pct=risk_ceiling_pct,
            )

    reward_pct = (target_price - price) / price

    shares_for_risk_budget = (capital_total * RISK_PER_TRADE_PCT) / risk_per_share
    if atr_percentile_252 is not None and atr_percentile_252 >= 0.85:
        shares_for_risk_budget /= 2

    position_value = shares_for_risk_budget * price
    max_position_value = capital_total * MAX_POSITION_PCT
    if position_value > max_position_value:
        position_value = max_position_value
        shares_for_risk_budget = max_position_value / price

    pct_of_portfolio = (position_value / capital_total) if capital_total > 0 else None

    common_fields = dict(
        stop_price=stop_price, stop_basis=stop_basis, entry_type=entry_type,
        risk_pct=risk_pct, risk_atr=risk_atr, risk_ceiling_pct=risk_ceiling_pct,
        target_price=target_price, target_basis=target_basis, reward_pct=reward_pct,
        risk_reward_gross=risk_reward_gross, risk_reward_net=risk_reward_net,
        shares_for_risk_budget=shares_for_risk_budget, position_value=position_value,
        pct_of_portfolio=pct_of_portfolio,
    )

    if position_value < MIN_POSITION_USD:
        reason = "posición demasiado pequeña - el coste de transacción se comería el resultado"
        return _not_viable(price, reason, **common_fields)

    return TradeGeometry(entry_price=price, **common_fields, viable=True, rejection_reason=None)
