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
"""

from dataclasses import dataclass

from app.services.technical_analysis import PriceLevel

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
