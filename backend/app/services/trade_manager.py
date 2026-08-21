"""Turns a trade plan into active risk management: how far the trailing
stop should have moved by now, and what a scaled exit should look like at
each R milestone. `exit_engine.py` decides *whether* something is wrong right
now; this module decides how the mechanics of an already-healthy trade
should evolve over time - the two are complementary, not overlapping (this
module never computes an urgency, and never will: a trailing stop moving up
is not itself a signal, it's bookkeeping the position's own progress creates).

Pure functions only, same discipline as `technical_analysis.py` - no I/O, no
DB. `portfolio_risk_service.py` is the one caller that persists what this
computes (`TradePlanRepositoryPort.update_trailing`).
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

# Chuck LeBeau's canonical Chandelier Exit parameters - the standard,
# widely-cited starting point for this indicator (highest high over N bars,
# minus a multiple of ATR(N)).
CHANDELIER_WINDOW = 22

# Base multiplier per GARCH volatility regime (volatility_model.GarchResult.regime),
# not a single fixed constant: the literature is consistent that an ATR
# trailing stop should give a trending, genuinely volatile tape more room
# (avoid a noise-driven stopout) and a calm tape less (a calm tape's own
# smaller moves are already meaningful, so protect them more precisely).
# First-pass values, not ablation-calibrated yet - same status
# BUY_THRESHOLD/AVOID_THRESHOLD started at before their own audit.
CHANDELIER_MULTIPLIER_BY_REGIME: dict[str, float] = {
    "baja": 2.5,
    "normal": 3.0,
    "elevada": 3.25,
    "alta": 3.5,
}
CHANDELIER_MULTIPLIER_DEFAULT = 3.0  # unknown/missing regime
# Once a position is already up >2R, protecting the locked-in gain outranks
# giving the trade room to breathe - a tighter multiplier regardless of
# volatility regime.
CHANDELIER_MULTIPLIER_PROFIT_LOCK = 2.0
CHANDELIER_PROFIT_LOCK_R = 2.0

# Per-position risk cap: no single stop should be allowed to risk more than
# this fraction of the portfolio's own capital. The aggregate cap across all
# open positions at once (6% in the brief) needs cross-position awareness
# this module doesn't have - that's portfolio_construction_service.py's job
# in a later phase; this is only ever the per-position guard.
MAX_POSITION_RISK_PCT = 0.01

# Scaled-exit milestones: sell a third of the *original* position at +1R
# (and move the stop to break-even - a suggestion, not something this module
# does automatically), another third at +2R, and let the remainder ride the
# Chandelier trail. A small tolerance (not an exact 1/3, 2/3 split) absorbs
# rounding from whole-share quantities.
SCALE_OUT_FRACTION = 1 / 3
SCALE_OUT_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class ChandelierResult:
    stop: float | None
    multiplier: float


def chandelier_stop(
    high: pd.Series, atr14: pd.Series, multiplier: float, window: int = CHANDELIER_WINDOW
) -> float | None:
    """The raw Chandelier Exit level for a long position: highest high over
    the trailing `window` bars, minus `multiplier` x the latest ATR(`window`
    is *not* necessarily the ATR window - `atr14` is whatever ATR series the
    caller already computed, typically the standard 14-bar one).

    `high` is expected to already be bounded to whatever history is actually
    relevant (see `portfolio_risk_service.py`, which slices it to bars on or
    after the position's own entry date) - this function itself no longer
    demands a full `window`-bar history before answering: a position younger
    than `window` bars uses everything it has instead of returning `None`,
    same idiom as `detect_recent_cross`'s own lookback-window slicing. A
    fixed `window`-bar requirement, applied to a `high` series that reached
    back before the position even opened, is exactly what let the trailing
    stop use a pre-entry high on a young position after a pullback - `None`
    is still correct when there's no data at all."""
    if high.empty or atr14.empty:
        return None
    highest = high.iloc[-window:].max() if len(high) > window else high.max()
    latest_atr = atr14.iloc[-1]
    if pd.isna(highest) or pd.isna(latest_atr):
        return None
    return float(highest - multiplier * latest_atr)


def chandelier_multiplier(vol_regime: str | None, r_multiple: float | None) -> float:
    """Which multiplier applies right now: the tighter profit-lock one once
    a position is up >2R, otherwise the volatility-regime-adjusted one."""
    if r_multiple is not None and r_multiple >= CHANDELIER_PROFIT_LOCK_R:
        return CHANDELIER_MULTIPLIER_PROFIT_LOCK
    return CHANDELIER_MULTIPLIER_BY_REGIME.get(vol_regime or "", CHANDELIER_MULTIPLIER_DEFAULT)


def update_trailing_stop(current_stop: float | None, candidate: float | None) -> float | None:
    """The trailing stop only ever moves up for a long position, never down
    - once risk is locked in, it stays locked in. `None` inputs pass through
    gracefully: no candidate yet keeps the current stop, no current stop yet
    adopts the candidate outright."""
    if candidate is None:
        return current_stop
    if current_stop is None:
        return candidate
    return max(current_stop, candidate)


def compute_trailing_stop(
    high: pd.Series,
    atr14: pd.Series,
    current_stop: float | None,
    r_multiple: float | None,
    vol_regime: str | None,
    price: float | None = None,
    window: int = CHANDELIER_WINDOW,
) -> ChandelierResult:
    """The full trailing-stop update for one evaluation: picks the right
    multiplier for where the trade stands, computes the Chandelier candidate,
    and folds it into the current stop (never lowering it). This is what
    `portfolio_risk_service.py` persists via `TradePlanRepositoryPort.update_trailing`
    each fresh evaluation.

    `price` (the latest closed price) is an extra sanity guard: a stop at or
    above the current price is never valid for a long position - discard
    that candidate outright (same as if none had been computed) rather than
    ever raising the stop past where the position actually sits. `None`
    (the default) skips the guard, matching every existing caller that
    hasn't been updated to pass it yet."""
    multiplier = chandelier_multiplier(vol_regime, r_multiple)
    candidate = chandelier_stop(high, atr14, multiplier, window)
    if candidate is not None and price is not None and candidate >= price:
        candidate = None
    return ChandelierResult(stop=update_trailing_stop(current_stop, candidate), multiplier=multiplier)


def max_shares_for_position_risk(
    portfolio_capital: float, entry_price: float, stop_price: float, max_risk_pct: float = MAX_POSITION_RISK_PCT
) -> float | None:
    """How many shares keep this position's risk (distance to stop x size)
    within `max_risk_pct` of `portfolio_capital` - the answer to "if the
    stop is this wide, how big can the position actually be", never the
    other way around (widening the stop to fit a desired size is exactly
    the anti-pattern this guards against). `None` when there's no real risk
    to size against (stop at or above entry, non-positive capital)."""
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0 or portfolio_capital <= 0:
        return None
    return (portfolio_capital * max_risk_pct) / risk_per_share


class ScaleOutAction(str, Enum):
    NONE = "none"  # no milestone reached yet, or every milestone already handled
    SELL_AT_1R = "sell_at_1r"
    SELL_AT_2R = "sell_at_2r"


@dataclass(frozen=True, slots=True)
class ScaledExitPlan:
    action: ScaleOutAction
    shares_to_sell: float
    shares_remaining_after: float
    suggested_new_stop: float | None  # break-even at +1R; None otherwise (Chandelier already governs +2R+)
    description: str  # human-readable Spanish, with concrete quantities - see module docstring


def compute_scaled_exit_plan(
    r_multiple: float | None, quantity_held: float, initial_quantity: float, entry_price: float
) -> ScaledExitPlan:
    """Whether a +1R or +2R scaled exit is due right now, read directly off
    how much of the *original* position is still held (`quantity_held` vs
    `initial_quantity`) rather than a separately-persisted "already
    suggested" flag - the position's own size is the ground truth of what's
    actually been sold, so this can't drift out of sync with reality the way
    a flag could (e.g. if a sell happened outside this app's suggestion)."""
    no_action = ScaledExitPlan(
        action=ScaleOutAction.NONE,
        shares_to_sell=0.0,
        shares_remaining_after=quantity_held,
        suggested_new_stop=None,
        description="Sin acción de escalado pendiente.",
    )
    if r_multiple is None or initial_quantity <= 0 or quantity_held <= 0:
        return no_action

    fraction_remaining = quantity_held / initial_quantity
    already_scaled_once = fraction_remaining <= (1 - SCALE_OUT_FRACTION) + SCALE_OUT_TOLERANCE
    already_scaled_twice = fraction_remaining <= (1 - 2 * SCALE_OUT_FRACTION) + SCALE_OUT_TOLERANCE

    if r_multiple >= 2.0 and already_scaled_once and not already_scaled_twice:
        shares_to_sell = min(initial_quantity * SCALE_OUT_FRACTION, quantity_held)
        remaining = quantity_held - shares_to_sell
        return ScaledExitPlan(
            action=ScaleOutAction.SELL_AT_2R,
            shares_to_sell=shares_to_sell,
            shares_remaining_after=remaining,
            suggested_new_stop=None,  # the Chandelier trail already governs from here
            description=(
                f"Objetivo de +2R alcanzado: vender {shares_to_sell:g} de {quantity_held:g} acciones a "
                f"mercado y dejar correr el resto ({remaining:g}) con el stop dinámico."
            ),
        )

    if r_multiple >= 1.0 and not already_scaled_once:
        shares_to_sell = min(initial_quantity * SCALE_OUT_FRACTION, quantity_held)
        remaining = quantity_held - shares_to_sell
        return ScaledExitPlan(
            action=ScaleOutAction.SELL_AT_1R,
            shares_to_sell=shares_to_sell,
            shares_remaining_after=remaining,
            suggested_new_stop=entry_price,  # break-even
            description=(
                f"Objetivo de +1R alcanzado: vender {shares_to_sell:g} de {quantity_held:g} acciones a "
                f"mercado y subir el stop a {entry_price:.2f} (break-even) en el resto."
            ),
        )

    return no_action
