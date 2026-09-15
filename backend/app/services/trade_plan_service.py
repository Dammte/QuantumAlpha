"""Lifecycle of a position's trade plan: reconstructing what a stop/target
would have been at entry, and translating the persisted plan into the
`PositionContext` `exit_engine.py` actually consumes.

Deliberately lazy, not captured synchronously the moment a BUY transaction is
registered: the transactions endpoint stays a fast, DB-only write (no
network/quant-suite cost added to it - see `PortfolioRiskService`'s docstring
for the per-ticker latency incident this is careful not to repeat), and
reconstructing from point-in-time history produces the *same* stop/target
number a live capture would have (identical formula -
`trade_geometry.compute_stop_and_target` - identical historical window). The
one thing this genuinely can't reconstruct is the propietario's own
subjective reasoning for buying - nobody but the person doing the buying can
supply that. `thesis` (Parte 5.4) is deliberately not that: it's
`generate_thesis`'s auto-generated, factual description of the setup at
entry (trend, stop distance, target basis) built from the same already-
computed `StopAndTarget` this function needed anyway, with
`RECONSTRUCTED_THESIS` appended honestly - every plan through this path is,
and will remain, built after the fact (see "deliberately lazy" above), so
that disclaimer is never stale or misleading to drop.

Trailing-stop updates (Chandelier Exit) and scaled exits are `trade_manager.py`'s
job - this module only ever sets `current_stop` once, equal to `initial_stop`,
at creation; `trade_manager.py` computes what it should trail to afterward
and persists it via `TradePlanRepositoryPort.update_trailing`.

2026-09 (reconstruction, Fase 4): the plan's own `engine_version` now stamps
`levels_engine.GATE_VERSION`, not `recommendation_engine.ENGINE_VERSION` -
the same "which live engine generation produced this" marker, now pointed at
whichever module is actually live (see `portfolio_risk_service.py`'s own
Fase 4 note).
"""

from datetime import date

import pandas as pd

from app.domain.interfaces.trade_plan_repository import TradePlanRepositoryPort
from app.domain.models.trade_plan import TradePlan
from app.domain.models.transaction import Transaction, TransactionType
from app.services import exit_engine as ee
from app.services import technical_analysis as ta
from app.services.levels_engine import GATE_VERSION
from app.services.trade_geometry import StopAndTarget, compute_stop_and_target

RECONSTRUCTED_THESIS = (
    "Plan reconstruido retroactivamente a partir del histórico de precio en la fecha de entrada - "
    "no es el stop/objetivo que se habría mostrado en el momento real de la compra."
)

_TREND_LABEL = {
    ta.TrendState.UPTREND: "una tendencia alcista confirmada",
    ta.TrendState.DOWNTREND: "una tendencia bajista",
    ta.TrendState.SIDEWAYS: "una tendencia lateral, sin una dirección clara",
}


def generate_thesis(ticker: str, entry_price: float, trend: ta.TrendState, stop_and_target: StopAndTarget) -> str:
    """Parte 5.4: an auto-generated, factual description of the setup at
    entry - never the propietario's own subjective reasoning (nobody but
    the person buying can supply that, see this module's own docstring),
    just the technical facts a fresh gate evaluation would have shown at
    the time: trend, stop distance, target and its basis. Built entirely
    from values `ensure_trade_plan` already computes (`trend` from the same
    `as_of_entry` frame `reconstruct_stop_and_target` uses, that function's
    own `StopAndTarget`) - no extra computation, no network call, safe to
    call from the same lazy reconstruction path every plan already goes
    through. `None` stop/target fields (no ATR yet, e.g.) are simply
    omitted rather than guessed at."""
    parts = [f"Entrada en {ticker} a {entry_price:.2f}, con {_TREND_LABEL[trend]}."]
    if stop_and_target.stop_loss is not None:
        risk_pct = (entry_price - stop_and_target.stop_loss) / entry_price
        parts.append(f"Stop en {stop_and_target.stop_loss:.2f} ({risk_pct:.1%} de riesgo).")
    if stop_and_target.take_profit is not None and stop_and_target.take_profit_method is not None:
        target_sentence = f"Objetivo en {stop_and_target.take_profit:.2f} ({stop_and_target.take_profit_method})"
        if stop_and_target.risk_reward is not None:
            target_sentence += f", relación beneficio:riesgo {stop_and_target.risk_reward:.1f}:1."
        else:
            target_sentence += "."
        parts.append(target_sentence)
    return " ".join(parts)


def find_current_lot_entry(transactions: list[Transaction], ticker: str) -> Transaction | None:
    """The BUY that opened the position currently held - the most recent
    point running quantity went from 0 (or never having started) to positive
    and has stayed above 0 ever since. Deliberately *not* "the first BUY
    ever": a full close-and-reopen cycle starts a new lot with its own entry
    price/date, and a stop/target reconstruction only makes sense against the
    *current* lot, not a fully-closed prior one. `transactions` must already
    be ordered by `executed_at` (as `PortfolioRepository.get_transactions`
    returns them) - unordered input would silently produce a wrong answer."""
    quantity = 0.0
    entry: Transaction | None = None
    for tx in transactions:
        if tx.ticker != ticker or tx.transaction_type not in (TransactionType.BUY, TransactionType.SELL):
            continue
        if tx.transaction_type == TransactionType.BUY:
            if quantity <= 1e-9:
                entry = tx  # (re)opening the position - this BUY starts the current lot
            quantity += tx.quantity
        else:
            quantity -= tx.quantity
    return entry if quantity > 1e-9 else None


def current_held_quantity(transactions: list[Transaction], ticker: str) -> float:
    """Net shares currently held for `ticker` - BUY quantity minus SELL
    quantity across every transaction ever recorded (not just the current
    lot), so it agrees with what the portfolio summary shows as held. Same
    algorithm as `PortfolioRepository._currently_held_quantity`, duplicated
    here rather than imported: that one lives in the DB-coupled
    infrastructure layer, this is a pure function of the same domain
    `Transaction` list `services/` is allowed to depend on directly."""
    quantity = 0.0
    for tx in transactions:
        if tx.ticker != ticker:
            continue
        if tx.transaction_type == TransactionType.BUY:
            quantity += tx.quantity
        elif tx.transaction_type == TransactionType.SELL:
            quantity -= tx.quantity
    return quantity


def reconstruct_stop_and_target(entry_price: float, ohlcv_as_of_entry: pd.DataFrame) -> StopAndTarget:
    """Runs the exact math a fresh passing gate uses (`compute_stop_and_target`)
    against the ticker's OWN history *as of the entry date* -
    `ohlcv_as_of_entry` must already be sliced to end there, so this never
    looks at a bar that hadn't happened yet."""
    close, high, low = ohlcv_as_of_entry["close"], ohlcv_as_of_entry["high"], ohlcv_as_of_entry["low"]
    raw_atr = ta.atr(high, low, close).iloc[-1] if len(close) else None
    atr14 = None if raw_atr is None or pd.isna(raw_atr) else float(raw_atr)
    levels = ta.support_resistance_levels(high, low, close)
    nearest_support = min(
        (lv for lv in levels if lv.kind == "support"), key=lambda lv: abs(lv.distance_pct), default=None
    )
    nearest_resistance = min(
        (lv for lv in levels if lv.kind == "resistance"), key=lambda lv: abs(lv.distance_pct), default=None
    )
    return compute_stop_and_target(entry_price, atr14, nearest_support, nearest_resistance)


def ensure_trade_plan(
    repo: TradePlanRepositoryPort,
    portfolio_id: int,
    ticker: str,
    transactions: list[Transaction],
    ohlcv: pd.DataFrame,
) -> TradePlan | None:
    """Returns the open trade plan for this ticker, reconstructing it on
    first use if one doesn't exist yet - covers both a position opened before
    this table existed and one opened after but never separately captured.
    Returns `None` only when there's no currently-open lot at all (nothing to
    plan for) or the entry date falls outside the `ohlcv` history provided."""
    entry_tx = find_current_lot_entry(transactions, ticker)
    if entry_tx is None:
        return None
    entry_date = entry_tx.executed_at.date()

    existing = repo.get_open(portfolio_id, ticker)
    # Defensive staleness guard, independent of whether a SELL-to-zero ever
    # called `repo.close()` on the way here (see the transactions endpoint,
    # which now does): if the open plan's own entry_date doesn't match the
    # *current* lot's real entry, it belongs to a fully-closed prior lot -
    # a full sell-and-rebuy cycle must never inherit a dead lot's already-
    # trailed current_stop, which could easily sit above the new lot's own
    # entry price. Closed explicitly (not just ignored) so it doesn't sit
    # around as a second, orphaned "open" row.
    if existing is not None and existing.entry_date == entry_date:
        return existing

    as_of_entry = ohlcv[ohlcv.index.date <= entry_date]
    if as_of_entry.empty:
        # Tercera auditoría, Bloque A-3: this used to close() the stale
        # `existing` plan (if any) *before* this check - if the OHLCV history
        # provided doesn't reach back to entry_date (a DCA'd position whose
        # earlier lot predates HISTORY_YEARS, or any entry_date/history
        # mismatch), the stale plan was destroyed and nothing replaced it.
        # And it never recovered: the next call finds no open plan at all
        # (already closed), still can't build a replacement (same history
        # gap), and returns None again - forever. With `plan is None`,
        # portfolio_risk_service.py skips the entire exit-engine block for
        # that position - no stop, no trailing, no exit_urgency, invisible to
        # the exit system. Closing a plan we cannot actually replace is worse
        # than leaving the stale one in place, so confirm the replacement is
        # buildable *first*.
        return None

    if existing is not None:
        repo.close(portfolio_id, ticker)

    stop_target = reconstruct_stop_and_target(entry_tx.price, as_of_entry)
    # Same classify_trend basis (SMA20/50/200) evaluate_gate itself judges
    # entries against - cheap, already-in-memory, no extra network cost,
    # just for the auto-generated thesis below (Parte 5.4).
    entry_close = as_of_entry["close"]
    sma20 = ta.sma(entry_close, 20).iloc[-1] if len(entry_close) >= 20 else None
    sma50 = ta.sma(entry_close, 50).iloc[-1] if len(entry_close) >= 50 else None
    sma200 = ta.sma(entry_close, 200).iloc[-1] if len(entry_close) >= 200 else None
    trend = ta.classify_trend(
        entry_tx.price,
        None if sma20 is None or pd.isna(sma20) else float(sma20),
        None if sma50 is None or pd.isna(sma50) else float(sma50),
        None if sma200 is None or pd.isna(sma200) else float(sma200),
    )
    thesis = f"{generate_thesis(ticker, entry_tx.price, trend, stop_target)} {RECONSTRUCTED_THESIS}"
    # The quantity held *right now*, not just entry_tx's own quantity - a
    # DCA'd position (bought more after the initial entry, before this plan
    # was ever created) should start scaled-exit tracking from what's
    # actually held today, not just the first buy's size.
    initial_quantity = current_held_quantity(transactions, ticker)
    return repo.create(
        portfolio_id=portfolio_id,
        ticker=ticker,
        entry_price=entry_tx.price,
        entry_date=entry_date,
        initial_stop=stop_target.stop_loss,
        initial_target=stop_target.take_profit,
        initial_quantity=initial_quantity,
        thesis=thesis,
        engine_version=GATE_VERSION,
    )


def build_position_context(
    plan: TradePlan, price: float, quantity: float, average_cost: float, bars_held: int
) -> ee.PositionContext:
    """Translates a persisted `TradePlan` plus the position's live facts into
    what `exit_engine.evaluate_exit` actually consumes."""
    r_multiple = None
    if plan.initial_stop is not None and plan.entry_price > plan.initial_stop:
        r_multiple = (price - plan.entry_price) / (plan.entry_price - plan.initial_stop)
    unrealized_pnl_pct = (price - average_cost) / average_cost if average_cost else None
    return ee.PositionContext(
        ticker=plan.ticker,
        average_cost=average_cost,
        quantity=quantity,
        opened_at=plan.entry_date,
        initial_stop=plan.initial_stop,
        current_stop=plan.current_stop,
        initial_target=plan.initial_target,
        highest_close_since_entry=max(plan.highest_close_since_entry, price),
        unrealized_pnl_pct=unrealized_pnl_pct,
        r_multiple=r_multiple,
        bars_held=bars_held,
    )


def bars_held_since(ohlcv: pd.DataFrame, entry_date: date) -> int:
    """How many bars of `ohlcv` (typically the closed daily frame) fall on or
    after `entry_date` - used for `PositionContext.bars_held`, e.g. for the
    "no progress in N sessions" stop (`trade_manager.py`, a later phase)."""
    if ohlcv.empty:
        return 0
    return int((ohlcv.index.date >= entry_date).sum())
