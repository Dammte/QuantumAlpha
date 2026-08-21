import numpy as np
import pandas as pd
import pytest

from app.services import backtest_engine as be
from app.services import technical_analysis as ta

# --- label_triple_barrier -----------------------------------------------------


def _ohlc(closes: list[float], wiggle: float = 0.5) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = pd.Series(closes, dtype=float)
    return close, close + wiggle, close - wiggle


def test_label_triple_barrier_target_hit_first():
    close = pd.Series([100.0, 102.0, 105.0, 108.0, 110.0])
    high = pd.Series([100.5, 102.5, 106.0, 108.5, 110.5])
    low = pd.Series([99.5, 101.5, 104.0, 107.5, 109.5])
    label = be.label_triple_barrier(close, high, low, entry_index=0, stop_loss=95.0, take_profit=105.0,
                                     vertical_barrier_bars=4)
    assert label is not None
    assert label.exit_reason == "target"
    assert label.exit_index == 2
    assert label.exit_price == pytest.approx(105.0)
    assert label.return_pct == pytest.approx(0.05)
    assert label.bars_held == 2
    assert label.mfe_pct == pytest.approx(0.06)
    assert label.mae_pct == pytest.approx(0.0)


def test_label_triple_barrier_stop_hit_first():
    close = pd.Series([100.0, 98.0, 95.0, 90.0, 85.0])
    high = pd.Series([100.5, 98.5, 95.5, 90.5, 85.5])
    low = pd.Series([99.5, 97.5, 90.0, 89.5, 84.5])
    label = be.label_triple_barrier(close, high, low, entry_index=0, stop_loss=92.0, take_profit=115.0,
                                     vertical_barrier_bars=4)
    assert label is not None
    assert label.exit_reason == "stop"
    assert label.exit_price == pytest.approx(92.0)
    assert label.return_pct == pytest.approx(-0.08)
    assert label.bars_held == 2
    assert label.mae_pct == pytest.approx(-0.10)
    assert label.mfe_pct == pytest.approx(0.0)


def test_label_triple_barrier_same_bar_tie_stop_wins():
    close = pd.Series([100.0, 100.0, 100.0])
    high = pd.Series([100.5, 120.0, 100.5])  # target (110) touched at bar 1
    low = pd.Series([99.5, 80.0, 99.5])  # stop (90) also touched at bar 1
    label = be.label_triple_barrier(close, high, low, entry_index=0, stop_loss=90.0, take_profit=110.0,
                                     vertical_barrier_bars=2)
    assert label is not None
    assert label.exit_reason == "stop"  # conservative convention - stop always wins a tie
    assert label.exit_price == pytest.approx(90.0)
    assert label.bars_held == 1


def test_label_triple_barrier_vertical_barrier_when_neither_touched():
    close = pd.Series([100.0, 101.0, 102.0, 103.0])
    high = pd.Series([100.5, 101.5, 102.5, 103.5])
    low = pd.Series([99.5, 100.5, 101.5, 102.5])
    label = be.label_triple_barrier(close, high, low, entry_index=0, stop_loss=90.0, take_profit=150.0,
                                     vertical_barrier_bars=3)
    assert label is not None
    assert label.exit_reason == "vertical"
    assert label.exit_index == 3
    assert label.exit_price == pytest.approx(103.0)
    assert label.return_pct == pytest.approx(0.03)
    assert label.bars_held == 3


def test_label_triple_barrier_none_when_entry_too_close_to_the_end():
    close = pd.Series([100.0, 101.0])
    high = close + 0.5
    low = close - 0.5
    assert be.label_triple_barrier(close, high, low, entry_index=1, stop_loss=90.0, take_profit=110.0,
                                    vertical_barrier_bars=5) is None


def test_label_triple_barrier_none_with_an_invalid_stop():
    close = pd.Series([100.0, 101.0, 102.0])
    high, low = close + 0.5, close - 0.5
    assert be.label_triple_barrier(close, high, low, 0, stop_loss=100.0, take_profit=110.0,
                                    vertical_barrier_bars=2) is None  # stop >= entry


def test_label_triple_barrier_none_with_an_invalid_target():
    close = pd.Series([100.0, 101.0, 102.0])
    high, low = close + 0.5, close - 0.5
    assert be.label_triple_barrier(close, high, low, 0, stop_loss=90.0, take_profit=100.0,
                                    vertical_barrier_bars=2) is None  # target <= entry


def test_label_triple_barrier_trailing_stop_locks_in_profit_a_fixed_stop_would_miss():
    # A rally that fully round-trips back to a loss: with a wide fixed stop
    # and unreachable target, the trade rides all the way down to the
    # vertical barrier at a loss. With trade_manager's real Chandelier
    # trailing, the stop ratchets up during the rally and locks in a gain
    # well before the round trip completes.
    close = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 130.0, 120.0, 110.0, 100.0, 90.0])
    high, low = close + 1.0, close - 1.0
    atr14 = pd.Series([2.0] * len(close))

    fixed = be.label_triple_barrier(close, high, low, 0, stop_loss=50.0, take_profit=1000.0,
                                     vertical_barrier_bars=9)
    trailing = be.label_triple_barrier(close, high, low, 0, stop_loss=50.0, take_profit=1000.0,
                                        vertical_barrier_bars=9, trailing=True, atr14=atr14, vol_regime="normal")

    assert fixed is not None and fixed.exit_reason == "vertical"
    assert fixed.return_pct == pytest.approx(-0.10)  # rode the full round trip down

    assert trailing is not None and trailing.exit_reason == "stop"
    assert trailing.exit_index == 5
    assert trailing.exit_price == pytest.approx(135.0)
    assert trailing.return_pct == pytest.approx(0.35)  # locked in well before the round trip


def test_label_triple_barrier_trailing_uses_the_profit_lock_multiplier_once_up_2r():
    # Entry at 100, stop at 90 -> risk (R) = 10. The rally clears +2R at the
    # bar where close=120 (r_multiple recomputed from the *original* stop_loss,
    # never the already-trailing current_stop) - trade_manager's profit-lock
    # multiplier (2.0, tighter than "normal"'s 3.0) should apply from there on.
    # Before this fix, chandelier_multiplier was always called with
    # r_multiple=None, so the trail only ever used the looser 3.0 multiplier.
    close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 120.0, 115.0, 110.0])
    high, low = close + 1.0, close - 1.0
    atr14 = pd.Series([2.0] * len(close))

    label = be.label_triple_barrier(
        close, high, low, entry_index=0, stop_loss=90.0, take_profit=1000.0,
        vertical_barrier_bars=8, trailing=True, atr14=atr14, vol_regime="normal",
    )

    assert label is not None
    assert label.exit_reason == "stop"
    assert label.exit_index == 6
    # Trail: 106-3*2=100 -> 111-3*2=105 -> 116-3*2=110 -> (r=2.0>=2R, mult=2.0)
    # 121-2*2=117 -> (r=2.5) 126-2*2=122 -> bar 6's low (119) touches 122.
    assert label.exit_price == pytest.approx(122.0)
    assert label.return_pct == pytest.approx(0.22)


def test_label_triple_barrier_gap_through_stop_fills_at_the_worse_open_price():
    # A stop at 95 that the market gaps straight through overnight, opening
    # at 62 - the same numbers the brief's own example uses. Without gap
    # awareness this books as -5% (filled at the stop, which was never
    # actually available to sell at); the real fill is -38% (the open).
    close = pd.Series([100.0, 62.0])
    high = pd.Series([100.5, 63.0])
    low = pd.Series([99.5, 60.0])
    open_ = pd.Series([100.0, 62.0])

    gap_aware = be.label_triple_barrier(close, high, low, entry_index=0, stop_loss=95.0, take_profit=200.0,
                                         vertical_barrier_bars=1, open_=open_)
    assert gap_aware is not None and gap_aware.exit_reason == "stop"
    assert gap_aware.exit_price == pytest.approx(62.0)
    assert gap_aware.return_pct == pytest.approx(-0.38)

    # No open series given -> no gap information to act on, the old,
    # honest-about-its-limits fill (at the stop itself) still holds.
    gap_blind = be.label_triple_barrier(close, high, low, entry_index=0, stop_loss=95.0, take_profit=200.0,
                                         vertical_barrier_bars=1)
    assert gap_blind is not None and gap_blind.exit_reason == "stop"
    assert gap_blind.exit_price == pytest.approx(95.0)
    assert gap_blind.return_pct == pytest.approx(-0.05)


def test_label_triple_barrier_gap_through_target_fills_at_the_better_open_price():
    close = pd.Series([100.0, 130.0])
    high = pd.Series([100.5, 131.0])
    low = pd.Series([99.5, 129.0])
    open_ = pd.Series([100.0, 130.0])

    label = be.label_triple_barrier(close, high, low, entry_index=0, stop_loss=90.0, take_profit=110.0,
                                     vertical_barrier_bars=1, open_=open_)
    assert label is not None
    assert label.exit_reason == "target"
    assert label.exit_price == pytest.approx(130.0)
    assert label.return_pct == pytest.approx(0.30)


# --- compute_trading_metrics ---------------------------------------------------


def _label(return_pct: float, bars_held: int = 5, mae: float = -0.01, mfe: float = 0.02) -> be.TripleBarrierLabel:
    entry_price = 100.0
    return be.TripleBarrierLabel(
        entry_index=0, exit_index=bars_held, exit_reason="target" if return_pct > 0 else "stop",
        entry_price=entry_price, exit_price=entry_price * (1 + return_pct), return_pct=return_pct,
        bars_held=bars_held, mae_pct=mae, mfe_pct=mfe,
    )


def test_trading_metrics_empty_labels():
    metrics = be.compute_trading_metrics([])
    assert metrics.n_trades == 0
    assert metrics.win_rate is None
    assert metrics.profit_factor is None


def test_trading_metrics_win_rate_and_expectancy():
    # 2 wins of +10%, 2 losses of -5%, net of the round-trip cost:
    # net_win = 0.10 - cost, net_loss = -0.05 - cost -> win_rate=0.5,
    # expectancy = 0.5*net_win + 0.5*net_loss.
    labels = [_label(0.10), _label(0.10), _label(-0.05), _label(-0.05)]
    metrics = be.compute_trading_metrics(labels)
    net_win = 0.10 - be.ROUND_TRIP_COST_PCT
    net_loss = -0.05 - be.ROUND_TRIP_COST_PCT
    assert metrics.n_trades == 4
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.avg_win_pct == pytest.approx(net_win)
    assert metrics.avg_loss_pct == pytest.approx(net_loss)
    assert metrics.expectancy_pct == pytest.approx(0.5 * net_win + 0.5 * net_loss)


def test_trading_metrics_profit_factor():
    # sum(net wins) / sum(|net losses|), not the gross 0.20/0.10=2.0 - costs
    # shrink every win and deepen every loss before the ratio is taken.
    labels = [_label(0.10), _label(0.10), _label(-0.05), _label(-0.05)]
    metrics = be.compute_trading_metrics(labels)
    net_win = 0.10 - be.ROUND_TRIP_COST_PCT
    net_loss = -0.05 - be.ROUND_TRIP_COST_PCT
    assert metrics.profit_factor == pytest.approx((2 * net_win) / abs(2 * net_loss))


def test_trading_metrics_profit_factor_none_without_losses():
    labels = [_label(0.10), _label(0.05)]
    metrics = be.compute_trading_metrics(labels)
    assert metrics.profit_factor is None


def test_trading_metrics_a_trade_that_doesnt_clear_costs_counts_as_a_net_loss():
    # +0.10% gross is smaller than the round-trip cost (0.2%) - after costs
    # it's a real loss, and must count as one for win_rate/profit_factor, not
    # get counted as a "win" just because the raw price move was positive.
    labels = [_label(0.001), _label(-0.05)]
    metrics = be.compute_trading_metrics(labels)
    assert metrics.win_rate == pytest.approx(0.0)
    assert metrics.profit_factor == pytest.approx(0.0)  # losses exist, but zero wins to offset them


def test_trading_metrics_mae_mfe_averages():
    labels = [_label(0.10, mae=-0.02, mfe=0.12), _label(-0.05, mae=-0.06, mfe=0.01)]
    metrics = be.compute_trading_metrics(labels)
    assert metrics.avg_mae_pct == pytest.approx((-0.02 + -0.06) / 2)
    assert metrics.avg_mfe_pct == pytest.approx((0.12 + 0.01) / 2)


def test_trading_metrics_max_drawdown_on_a_losing_streak():
    # Three consecutive -10% trades, net of the round-trip cost. Segunda
    # auditoría, Bloque 2: the equity curve must start from a baseline of 1.0
    # *before* the first trade, not from the first trade's own multiplier -
    # my own bug from the previous round (this exact test used to assert the
    # understated (0.729-0.9)/0.9, which makes trade 1's own loss invisible
    # to the drawdown). The real drawdown is peak(1.0)-to-trough, the full
    # compounded loss from the pre-trade baseline.
    labels = [_label(-0.10), _label(-0.10), _label(-0.10)]
    metrics = be.compute_trading_metrics(labels)
    net_r = -0.10 - be.ROUND_TRIP_COST_PCT
    expected_curve = np.cumprod([1.0, 1 + net_r, 1 + net_r, 1 + net_r])
    expected_drawdown = (expected_curve[-1] - expected_curve[0]) / expected_curve[0]
    assert metrics.max_drawdown_pct == pytest.approx(expected_drawdown, abs=1e-9)


def test_trading_metrics_avg_bars_held_split_by_outcome():
    labels = [_label(0.10, bars_held=3), _label(0.10, bars_held=5), _label(-0.05, bars_held=10)]
    metrics = be.compute_trading_metrics(labels)
    assert metrics.avg_bars_held_winners == pytest.approx(4.0)
    assert metrics.avg_bars_held_losers == pytest.approx(10.0)


def test_trading_metrics_net_return_subtracts_round_trip_cost():
    labels = [_label(0.10)]
    metrics = be.compute_trading_metrics(labels)
    assert metrics.gross_return_pct == pytest.approx(0.10)
    assert metrics.net_return_pct == pytest.approx(0.10 - be.ROUND_TRIP_COST_PCT)


# --- split_by_date --------------------------------------------------------------


def test_split_by_date_partitions_train_and_test():
    index = pd.bdate_range("2020-01-01", periods=10)
    train_mask, test_mask = be.split_by_date(index, cutoff=pd.Timestamp(index[5]))
    assert train_mask.sum() == 5
    assert test_mask.sum() == 5
    assert not (train_mask & test_mask).any()


# --- purged_kfold_splits ---------------------------------------------------------


def test_purged_kfold_splits_train_and_test_never_overlap():
    splits = be.purged_kfold_splits(n_samples=100, n_splits=5, embargo_fraction=0.02)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))


def test_purged_kfold_splits_embargoes_around_each_test_fold():
    splits = be.purged_kfold_splits(n_samples=100, n_splits=5, embargo_fraction=0.05)
    # embargo = int(100*0.05) = 5 samples purged from train on each side of a test fold
    _, test_idx = splits[2]  # a middle fold, embargoed on both sides
    train_idx = splits[2][0]
    test_start, test_end = test_idx[0], test_idx[-1]
    for offset in range(1, 5):
        assert (test_start - offset) not in train_idx
        assert (test_end + offset) not in train_idx


def test_purged_kfold_splits_empty_with_too_few_samples():
    assert be.purged_kfold_splits(n_samples=3, n_splits=5) == []


# --- buy_and_hold_labels ---------------------------------------------------------


def test_buy_and_hold_labels_basic_return():
    close = pd.Series([100.0, 105.0, 110.0, 108.0])
    labels = be.buy_and_hold_labels(close, entry_indices=[0], horizon_bars=3)
    assert len(labels) == 1
    assert labels[0].exit_reason == "vertical"
    assert labels[0].return_pct == pytest.approx(0.08)
    assert labels[0].mfe_pct == pytest.approx(0.10)  # peak at 110
    assert labels[0].mae_pct == pytest.approx(0.0)  # never dipped below entry


def test_buy_and_hold_labels_skips_entries_too_close_to_the_end():
    close = pd.Series([100.0, 105.0])
    assert be.buy_and_hold_labels(close, entry_indices=[1], horizon_bars=5) == []


# --- random_entry_labels ---------------------------------------------------------


def test_random_entry_labels_produces_valid_labels():
    n = 100
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, n)))
    high, low = close + 1.0, close - 1.0
    atr14 = pd.Series([2.0] * n)
    labels = be.random_entry_labels(close, high, low, n_trades=10, horizon_bars=10, atr14=atr14,
                                     atr_stop_multiple=2.5, reward_risk_ratio=2.0, seed=42)
    assert 0 < len(labels) <= 10
    for label in labels:
        assert label.exit_reason in ("stop", "target", "vertical")


def test_random_entry_labels_deterministic_given_a_seed():
    n = 100
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, n)))
    high, low = close + 1.0, close - 1.0
    atr14 = pd.Series([2.0] * n)
    first = be.random_entry_labels(close, high, low, 10, 10, atr14, 2.5, 2.0, seed=7)
    second = be.random_entry_labels(close, high, low, 10, 10, atr14, 2.5, 2.0, seed=7)
    assert [lbl.entry_index for lbl in first] == [lbl.entry_index for lbl in second]


# --- run_triple_barrier_backtest (orchestration) --------------------------------


def _synthetic_regime_series(n: int, block: int, seed: int = 123) -> pd.Series:
    """Same construction as test_walk_forward_backtest.py's helper -
    alternating long uptrend/downtrend regimes, long enough (relative to the
    200-day SMA's memory) for classify_trend to cleanly read the regime it's
    actually in."""
    rng = np.random.default_rng(seed)
    returns = []
    n_blocks = n // block + 1
    for k in range(n_blocks):
        drift = 0.0025 if k % 2 == 0 else -0.0025
        returns.extend(rng.normal(drift, 0.008, block))
    prices = 100 * np.cumprod(1 + np.array(returns[:n]))
    return pd.Series(prices)


def _full_bundle(close: pd.Series) -> dict:
    high = close * 1.01
    low = close * 0.99
    # Each bar's open ~= the prior bar's close (a small, realistic gap), not
    # close itself - a degenerate zero-range "open==close" bar would never
    # exercise the gap-fill path at all.
    open_ = close.shift(1).bfill()
    return {
        "close": close,
        "high": high,
        "low": low,
        "open_": open_,
        "sma20": ta.sma(close, 20),
        "sma50": ta.sma(close, 50),
        "sma150": ta.sma(close, 150),
        "sma200": ta.sma(close, 200),
        "rsi14": ta.rsi(close),
        "adx14": ta.adx(high, low, close),
        "plus_di": ta.dmi(high, low, close)[0],
        "minus_di": ta.dmi(high, low, close)[1],
        "atr14": ta.atr(high, low, close),
    }


def test_run_triple_barrier_backtest_on_a_long_uptrend_produces_a_result():
    close = _synthetic_regime_series(n=1500, block=1500)  # one long uptrend block
    bundle = _full_bundle(close)
    result = be.run_triple_barrier_backtest(horizon_days=21, **bundle)
    assert result is not None
    assert result.horizon_days == 21
    assert result.n_signals_evaluated >= 0
    # Every "comprar" signal replayed also has a benchmark counterpart at the
    # exact same entry points.
    assert result.buy_and_hold.n_trades <= result.n_signals_evaluated


def test_run_triple_barrier_backtest_none_with_too_little_history():
    close = pd.Series(np.linspace(100, 110, 50))
    bundle = _full_bundle(close)
    assert be.run_triple_barrier_backtest(horizon_days=21, **bundle) is None


def test_run_triple_barrier_backtest_empty_metrics_when_no_comprar_signals_fire():
    # An unbroken, steep downtrend should never replay a "comprar" verdict.
    n = 1500
    close = pd.Series(500 - np.arange(n) * 0.2)
    bundle = _full_bundle(close)
    result = be.run_triple_barrier_backtest(horizon_days=21, **bundle)
    assert result is not None
    assert result.n_signals_evaluated == 0
    assert result.strategy_fixed.n_trades == 0
    assert result.strategy_trailing.n_trades == 0


def test_run_triple_barrier_backtest_samples_on_a_non_overlapping_grid():
    # Regression guard: the number of "comprar" signals actually evaluated
    # can never exceed the non-overlapping grid's own size (stride ==
    # horizon_days, same discipline walk_forward_backtest.py uses to avoid
    # overlapping-window pseudo-replication) - a bug that samples every bar
    # instead would let n_signals_evaluated blow past this ceiling.
    close = _synthetic_regime_series(n=2000, block=2000)
    bundle = _full_bundle(close)
    horizon = 10
    result = be.run_triple_barrier_backtest(horizon_days=horizon, **bundle)
    assert result is not None
    max_possible_grid_points = len(range(be.WARMUP_BARS, len(close) - horizon - 1, horizon))
    assert max_possible_grid_points > 1  # sanity: the fixture is big enough to matter
    assert result.n_signals_evaluated <= max_possible_grid_points
