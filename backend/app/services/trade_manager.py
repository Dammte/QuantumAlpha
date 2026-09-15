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

from app.core.trading_params import (
    CHANDELIER_MULT_BY_VOL,
    CHANDELIER_PROFIT_LOCK_MULT,
    CHANDELIER_PROFIT_LOCK_R,
    CHANDELIER_WINDOW,
    LAST_TRANCHE_TIME_STOP_BARS,
    MIN_POSITION_FOR_SCALING,
    RISK_PER_TRADE_PCT,
    SCALE_OUT_1R_FRACTION,
    SCALE_OUT_2R_FRACTION,
    TRANSACTION_COST_PCT,
)

# Re-exported so existing call sites/tests (`tm.CHANDELIER_WINDOW`, etc.) keep
# resolving unchanged - `app.core.trading_params` is the single source of
# truth for the values themselves (Parte 19 of the reconstruction brief).
# Chuck LeBeau's canonical Chandelier Exit shape (highest high over N bars,
# minus a multiple of ATR(N)) still applies; only the window/multipliers were
# recalibrated (Parte 3.2/20 - tighter across the board for a 2-10 session
# holding period, not the multi-week swing the original 22-bar/2.5-3.5 ATR
# values were tuned for) - not yet run through
# `scripts/chandelier_calibration_study.py` against a real trigger-based
# sample (Fase 8), same "coherent but unmeasured" status Parte 20 flags.
CHANDELIER_MULTIPLIER_BY_REGIME = CHANDELIER_MULT_BY_VOL
CHANDELIER_MULTIPLIER_DEFAULT = CHANDELIER_MULT_BY_VOL["normal"]  # unknown/missing regime
# Once a position is already up beyond CHANDELIER_PROFIT_LOCK_R, protecting
# the locked-in gain outranks giving the trade room to breathe - a tighter
# multiplier regardless of volatility regime.
CHANDELIER_MULTIPLIER_PROFIT_LOCK = CHANDELIER_PROFIT_LOCK_MULT

# Per-position risk cap: no single stop should be allowed to risk more than
# this fraction of the portfolio's own capital. The aggregate cap across all
# open positions at once (6% in the brief) needs cross-position awareness
# this module doesn't have - that's portfolio_construction_service.py's job;
# this is only ever the per-position guard.
MAX_POSITION_RISK_PCT = RISK_PER_TRADE_PCT

# Scaled-exit milestones (Parte 8): sell ~a third of the *original* position
# at +1R (and move the stop to break-even, cost-inclusive - a suggestion,
# never something this module executes automatically), another ~third at
# +2R, and let the remainder ride the Chandelier trail. A small tolerance
# (not an exact 1/3, 2/3 split) absorbs rounding from whole-share quantities.
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


def update_trailing_stop(
    current_stop: float | None, candidate: float | None, price: float | None = None
) -> float | None:
    """The trailing stop only ever moves up for a long position, never down
    - once risk is locked in, it stays locked in. `None` inputs pass through
    gracefully: no candidate yet keeps the current stop, no current stop yet
    adopts the candidate outright.

    Tercera auditoría, Bloque A-1: a `current_stop` at or above `price` is
    never a legitimate state for an open long position - `compute_trailing_stop`'s
    own `price` guard only ever *prevents* a new invalid candidate from being
    adopted, it can't repair a stop that was already persisted as invalid
    before that guard existed (the exact Chandelier pre-entry-high bug this
    same audit's predecessor fixed). Treating that impossible value as
    "already locked in" made `max(current_stop, candidate)` keep it forever,
    even once fresh, valid (lower) candidates started arriving - reproduced
    with a pre-fix plan (`current_stop=263.0`, `price=95.0`): three
    consecutive evaluations all stayed at `EXIT_NOW` with "el precio ha
    perforado el stop de protección vigente (263.00)". Once `price` is
    passed, a corrupt `current_stop` is discarded (never trusted as a floor)
    so a valid candidate can replace it - self-healing on the very next
    evaluation, no data migration needed, since `price` is already threaded
    through from `portfolio_risk_service.py`."""
    if price is not None and current_stop is not None and current_stop >= price:
        current_stop = None
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
    return ChandelierResult(stop=update_trailing_stop(current_stop, candidate, price=price), multiplier=multiplier)


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
    # Parte 8: the last tranche (after both scale-outs) hasn't reached +3R
    # within LAST_TRANCHE_TIME_STOP_BARS - close what's left outright rather
    # than let the Chandelier trail run on it indefinitely.
    CLOSE_LAST_TRANCHE = "close_last_tranche"


@dataclass(frozen=True, slots=True)
class ScaledExitPlan:
    action: ScaleOutAction
    shares_to_sell: float
    shares_remaining_after: float
    suggested_new_stop: float | None  # break-even at +1R; None otherwise (Chandelier already governs +2R+)
    description: str  # human-readable Spanish, with concrete quantities - see module docstring


def _break_even_with_costs(entry_price: float) -> float:
    """Parte 8: break-even INCLUDES the round-trip transaction cost, not the
    bare entry price - a stop at exactly `entry_price` still nets a small
    loss once both legs' costs are counted."""
    return entry_price * (1 + 2 * TRANSACTION_COST_PCT)


def _price_at_r_multiple(entry_price: float, initial_stop: float | None, r: float) -> float | None:
    """The price `r` R-multiples above entry, using the position's own
    *initial* risk-per-share (entry - initial_stop) as the R unit - the same
    denominator `r_multiple` itself is computed against elsewhere, so a
    stop suggested here is directly comparable to it. `None` when
    `initial_stop` is unknown or invalid (at/above entry) - the caller
    degrades gracefully rather than fabricating a stop from nothing."""
    if initial_stop is None or initial_stop >= entry_price:
        return None
    risk_per_share = entry_price - initial_stop
    return entry_price + r * risk_per_share


def compute_scaled_exit_plan(
    r_multiple: float | None,
    quantity_held: float,
    initial_quantity: float,
    entry_price: float,
    initial_stop: float | None = None,
    bars_held: int | None = None,
) -> ScaledExitPlan:
    """Whether a +1R or +2R scaled exit is due right now, read directly off
    how much of the *original* position is still held (`quantity_held` vs
    `initial_quantity`) rather than a separately-persisted "already
    suggested" flag - the position's own size is the ground truth of what's
    actually been sold, so this can't drift out of sync with reality the way
    a flag could (e.g. if a sell happened outside this app's suggestion).

    Parte 8 in full: +1R sells `SCALE_OUT_1R_FRACTION` and raises the stop to
    break-even *including* round-trip costs (`_break_even_with_costs`); +2R
    sells `SCALE_OUT_2R_FRACTION` more and raises the stop to the +1R price
    level (not just "the Chandelier trail governs from here" - a concrete
    floor beneath it); a position under `MIN_POSITION_FOR_SCALING` at open
    never scales at all (a third of a $120 position is $40, commission eats
    the profit) - it exits whole at +2R instead; and once fully scaled down
    to the last tranche, `bars_held` beyond `LAST_TRANCHE_TIME_STOP_BARS`
    without reaching +3R closes what's left outright rather than letting the
    trail run on it indefinitely. `initial_stop`/`bars_held` are optional
    (`None` skips the behavior that needs them) so a caller that doesn't
    have one yet still gets the rest of this function's behavior."""
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
    already_scaled_once = fraction_remaining <= (1 - SCALE_OUT_1R_FRACTION) + SCALE_OUT_TOLERANCE
    already_scaled_twice = (
        fraction_remaining <= (1 - SCALE_OUT_1R_FRACTION - SCALE_OUT_2R_FRACTION) + SCALE_OUT_TOLERANCE
    )

    if (
        already_scaled_twice
        and bars_held is not None
        and bars_held > LAST_TRANCHE_TIME_STOP_BARS
        and r_multiple < 3.0
    ):
        return ScaledExitPlan(
            action=ScaleOutAction.CLOSE_LAST_TRANCHE,
            shares_to_sell=quantity_held,
            shares_remaining_after=0.0,
            suggested_new_stop=None,
            description=(
                f"Último tercio sin alcanzar +3R tras {bars_held} sesiones: cerrar las {quantity_held:g} "
                "acciones restantes a mercado (stop temporal del último tercio, Parte 8)."
            ),
        )

    small_position = (initial_quantity * entry_price) < MIN_POSITION_FOR_SCALING
    if small_position:
        if r_multiple >= 2.0 and not already_scaled_once:
            return ScaledExitPlan(
                action=ScaleOutAction.SELL_AT_2R,
                shares_to_sell=quantity_held,
                shares_remaining_after=0.0,
                suggested_new_stop=None,
                description=(
                    f"Posición pequeña (menos de {MIN_POSITION_FOR_SCALING:g}$ al abrir): sin escalado - "
                    f"objetivo de +2R alcanzado, vender las {quantity_held:g} acciones a mercado."
                ),
            )
        return no_action

    if r_multiple >= 2.0 and already_scaled_once and not already_scaled_twice:
        shares_to_sell = min(initial_quantity * SCALE_OUT_2R_FRACTION, quantity_held)
        remaining = quantity_held - shares_to_sell
        one_r_price = _price_at_r_multiple(entry_price, initial_stop, 1.0)
        stop_clause = f" y subir el stop a {one_r_price:.2f} (+1R)" if one_r_price is not None else ""
        return ScaledExitPlan(
            action=ScaleOutAction.SELL_AT_2R,
            shares_to_sell=shares_to_sell,
            shares_remaining_after=remaining,
            suggested_new_stop=one_r_price,
            description=(
                f"Objetivo de +2R alcanzado: vender {shares_to_sell:g} de {quantity_held:g} acciones a mercado"
                f"{stop_clause} en el resto ({remaining:g})."
            ),
        )

    if r_multiple >= 1.0 and not already_scaled_once:
        shares_to_sell = min(initial_quantity * SCALE_OUT_1R_FRACTION, quantity_held)
        remaining = quantity_held - shares_to_sell
        break_even = _break_even_with_costs(entry_price)
        return ScaledExitPlan(
            action=ScaleOutAction.SELL_AT_1R,
            shares_to_sell=shares_to_sell,
            shares_remaining_after=remaining,
            suggested_new_stop=break_even,
            description=(
                f"Objetivo de +1R alcanzado: vender {shares_to_sell:g} de {quantity_held:g} acciones a "
                f"mercado y subir el stop a {break_even:.2f} (break-even + costes) en el resto."
            ),
        )

    return no_action
