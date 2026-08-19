"""Lifecycle of a position's trade plan: reconstructing what a stop/target
would have been at entry, and translating the persisted plan into the
`PositionContext` `exit_engine.py` actually consumes.

Deliberately lazy, not captured synchronously the moment a BUY transaction is
registered: the transactions endpoint stays a fast, DB-only write (no
network/quant-suite cost added to it - see `PortfolioRiskService`'s docstring
for the per-ticker latency incident this is careful not to repeat), and
reconstructing from point-in-time history produces the *same* stop/target
number a live capture would have (identical formula -
`recommendation_engine.compute_stop_and_target` - identical historical
window). The one thing this genuinely can't reconstruct is *why* the position
was opened - nobody but the person doing the buying can fill in a thesis, so
a reconstructed plan says so honestly (`RECONSTRUCTED_THESIS`) rather than
inventing one.

Trailing-stop updates (Chandelier Exit) and scaled exits are `trade_manager.py`'s
job (a later phase) - this module only ever sets `current_stop` once, equal
to `initial_stop`, at creation.
"""

from datetime import date

import pandas as pd

from app.domain.interfaces.trade_plan_repository import TradePlanRepositoryPort
from app.domain.models.trade_plan import TradePlan
from app.domain.models.transaction import Transaction, TransactionType
from app.services import exit_engine as ee
from app.services import technical_analysis as ta
from app.services.recommendation_engine import ENGINE_VERSION, StopAndTarget, compute_stop_and_target

RECONSTRUCTED_THESIS = (
    "Plan reconstruido retroactivamente a partir del histórico de precio en la fecha de entrada - "
    "no es el stop/objetivo que se habría mostrado en el momento real de la compra."
)


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


def reconstruct_stop_and_target(entry_price: float, ohlcv_as_of_entry: pd.DataFrame) -> StopAndTarget:
    """Runs the exact math `build_recommendation` uses for a fresh "comprar"
    signal (`compute_stop_and_target`) against the ticker's OWN history *as
    of the entry date* - `ohlcv_as_of_entry` must already be sliced to end
    there, so this never looks at a bar that hadn't happened yet."""
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
    existing = repo.get_open(portfolio_id, ticker)
    if existing is not None:
        return existing

    entry_tx = find_current_lot_entry(transactions, ticker)
    if entry_tx is None:
        return None

    entry_date = entry_tx.executed_at.date()
    as_of_entry = ohlcv[ohlcv.index.date <= entry_date]
    if as_of_entry.empty:
        return None

    stop_target = reconstruct_stop_and_target(entry_tx.price, as_of_entry)
    return repo.create(
        portfolio_id=portfolio_id,
        ticker=ticker,
        entry_price=entry_tx.price,
        entry_date=entry_date,
        initial_stop=stop_target.stop_loss,
        initial_target=stop_target.take_profit,
        thesis=RECONSTRUCTED_THESIS,
        engine_version=ENGINE_VERSION,
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
