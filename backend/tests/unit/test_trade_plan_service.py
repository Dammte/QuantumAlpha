from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from app.domain.models.trade_plan import TradePlan
from app.domain.models.transaction import Transaction, TransactionType
from app.services import trade_plan_service as tps


def _tx(
    ticker: str | None,
    transaction_type: TransactionType,
    quantity: float,
    price: float = 100.0,
    executed_at: datetime = datetime(2024, 1, 1),
    tx_id: int = 1,
) -> Transaction:
    return Transaction(
        id=tx_id,
        portfolio_id=1,
        ticker=ticker,
        transaction_type=transaction_type,
        quantity=quantity,
        price=price,
        fees=0.0,
        executed_at=executed_at,
    )


# --- find_current_lot_entry --------------------------------------------------


def test_find_current_lot_entry_is_the_only_buy():
    txs = [_tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5))]
    entry = tps.find_current_lot_entry(txs, "AAPL")
    assert entry is not None
    assert entry.executed_at == datetime(2024, 1, 5)


def test_find_current_lot_entry_none_when_fully_sold():
    txs = [
        _tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5)),
        _tx("AAPL", TransactionType.SELL, 10, executed_at=datetime(2024, 2, 1)),
    ]
    assert tps.find_current_lot_entry(txs, "AAPL") is None


def test_find_current_lot_entry_is_the_reopening_buy_not_the_original_one():
    txs = [
        _tx("AAPL", TransactionType.BUY, 10, price=90.0, executed_at=datetime(2024, 1, 5)),
        _tx("AAPL", TransactionType.SELL, 10, executed_at=datetime(2024, 2, 1)),
        _tx("AAPL", TransactionType.BUY, 5, price=120.0, executed_at=datetime(2024, 3, 1)),
    ]
    entry = tps.find_current_lot_entry(txs, "AAPL")
    assert entry is not None
    assert entry.executed_at == datetime(2024, 3, 1)
    assert entry.price == pytest.approx(120.0)


def test_find_current_lot_entry_survives_a_partial_sell():
    # A partial sell doesn't close the lot - the entry is still the original
    # BUY, not the (nonexistent) later one.
    txs = [
        _tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5)),
        _tx("AAPL", TransactionType.SELL, 4, executed_at=datetime(2024, 2, 1)),
    ]
    entry = tps.find_current_lot_entry(txs, "AAPL")
    assert entry is not None
    assert entry.executed_at == datetime(2024, 1, 5)


def test_find_current_lot_entry_averages_up_still_anchors_to_the_first_buy():
    # Two BUYs without a full close in between - opened_at stays the first one.
    txs = [
        _tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5)),
        _tx("AAPL", TransactionType.BUY, 5, executed_at=datetime(2024, 2, 1)),
    ]
    entry = tps.find_current_lot_entry(txs, "AAPL")
    assert entry is not None
    assert entry.executed_at == datetime(2024, 1, 5)


def test_find_current_lot_entry_ignores_other_tickers_and_cash_movements():
    txs = [
        _tx("MSFT", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 1)),
        _tx(None, TransactionType.DEPOSIT, 1000, executed_at=datetime(2024, 1, 2)),
        _tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5)),
    ]
    entry = tps.find_current_lot_entry(txs, "AAPL")
    assert entry is not None
    assert entry.executed_at == datetime(2024, 1, 5)


# --- current_held_quantity ----------------------------------------------------


def test_current_held_quantity_nets_buys_and_sells():
    txs = [
        _tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5)),
        _tx("AAPL", TransactionType.BUY, 5, executed_at=datetime(2024, 2, 1)),
        _tx("AAPL", TransactionType.SELL, 3, executed_at=datetime(2024, 3, 1)),
    ]
    assert tps.current_held_quantity(txs, "AAPL") == pytest.approx(12.0)


def test_current_held_quantity_ignores_other_tickers():
    txs = [
        _tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5)),
        _tx("MSFT", TransactionType.BUY, 100, executed_at=datetime(2024, 1, 6)),
    ]
    assert tps.current_held_quantity(txs, "AAPL") == pytest.approx(10.0)


def test_current_held_quantity_zero_when_fully_sold():
    txs = [
        _tx("AAPL", TransactionType.BUY, 10, executed_at=datetime(2024, 1, 5)),
        _tx("AAPL", TransactionType.SELL, 10, executed_at=datetime(2024, 2, 1)),
    ]
    assert tps.current_held_quantity(txs, "AAPL") == pytest.approx(0.0)


# --- reconstruct_stop_and_target ---------------------------------------------


def _ohlcv_df(n: int, closes) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(closes, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([1_000_000.0] * n, index=dates),
        }
    )


def test_reconstruct_stop_and_target_produces_a_stop_with_enough_history():
    n = 60
    df = _ohlcv_df(n, 100 + np.sin(np.arange(n) / 3) * 5)
    result = tps.reconstruct_stop_and_target(entry_price=float(df["close"].iloc[-1]), ohlcv_as_of_entry=df)
    assert result.stop_loss is not None
    assert result.stop_loss < df["close"].iloc[-1]


def test_reconstruct_stop_and_target_none_with_too_little_history_for_atr():
    df = _ohlcv_df(5, [100.0, 101.0, 99.0, 102.0, 100.0])
    result = tps.reconstruct_stop_and_target(entry_price=100.0, ohlcv_as_of_entry=df)
    assert result.stop_loss is None


# --- build_position_context ---------------------------------------------------


def _plan(
    initial_stop: float | None = 90.0,
    current_stop: float | None = 92.0,
    initial_target: float | None = 120.0,
    entry_price: float = 100.0,
    highest_close_since_entry: float = 105.0,
    initial_quantity: float = 10.0,
) -> TradePlan:
    return TradePlan(
        id=1,
        portfolio_id=1,
        ticker="AAPL",
        entry_price=entry_price,
        entry_date=date(2024, 1, 5),
        initial_stop=initial_stop,
        initial_target=initial_target,
        current_stop=current_stop,
        highest_close_since_entry=highest_close_since_entry,
        initial_quantity=initial_quantity,
        thesis="",
        engine_version="v1",
        updated_at=datetime(2024, 1, 5),
        closed_at=None,
    )


def test_build_position_context_computes_r_multiple():
    # entry=100, stop=90 -> risk=10. price=115 -> (115-100)/10 = 1.5R.
    ctx = tps.build_position_context(_plan(), price=115.0, quantity=10.0, average_cost=100.0, bars_held=20)
    assert ctx.r_multiple == pytest.approx(1.5)
    assert ctx.opened_at == date(2024, 1, 5)
    assert ctx.current_stop == pytest.approx(92.0)


def test_build_position_context_r_multiple_none_without_an_initial_stop():
    ctx = tps.build_position_context(
        _plan(initial_stop=None), price=115.0, quantity=10.0, average_cost=100.0, bars_held=20
    )
    assert ctx.r_multiple is None


def test_build_position_context_highest_close_tracks_the_live_price_too():
    # Even if the persisted highest_close_since_entry hasn't been trailed up
    # yet by trade_manager.py, a new all-time-high live price is reflected.
    ctx = tps.build_position_context(
        _plan(highest_close_since_entry=105.0), price=130.0, quantity=10.0, average_cost=100.0, bars_held=20
    )
    assert ctx.highest_close_since_entry == pytest.approx(130.0)


def test_build_position_context_unrealized_pnl_pct():
    ctx = tps.build_position_context(_plan(), price=110.0, quantity=10.0, average_cost=100.0, bars_held=20)
    assert ctx.unrealized_pnl_pct == pytest.approx(0.10)


# --- bars_held_since -----------------------------------------------------------


def test_bars_held_since_counts_bars_on_or_after_entry():
    df = _ohlcv_df(10, [100.0] * 10)
    entry_date = df.index[3].date()
    assert tps.bars_held_since(df, entry_date) == 7


def test_bars_held_since_empty_df_is_zero():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert tps.bars_held_since(empty, date(2024, 1, 1)) == 0
