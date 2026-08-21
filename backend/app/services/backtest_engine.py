"""Triple-barrier backtesting (López de Prado, *Advances in Financial
Machine Learning*, cap. 3) - the D7 fix. `walk_forward_backtest.py` measures
`close[i+h]/close[i] - 1`: buy-and-hold to a fixed horizon, completely
ignoring the very `stop_loss`/`take_profit` `build_recommendation` proposes
at that same bar. It validates a strategy nobody actually executes. This
module labels each signal with whichever of three barriers is hit first -
the proposed stop, the proposed target, or a maximum holding period - using
intrabar high/low (not just the close, which would systematically understate
stop-outs), and reports the trading metrics that actually matter for a
system managed at this scale (expectancy, profit factor, MAE/MFE, drawdown,
net of costs), not just a mean return.

`walk_forward_backtest.py` is left as-is (legacy, per the brief's own
suggestion) - still used by `compute_core_signals()` for the per-ticker
backtest read shown in "Analizar activo" at its own 1m/3m/6m horizons. This
is a separate, additive engine, validated at 10/21 sessions (this
portfolio's actual holding horizon, not 63/126), for `scripts/factor_ablation_study.py`'s
rewrite (Fase 5) and a future "Rendimiento del sistema" view (Fase 7) - not
yet wired into the live single-ticker analysis path, which is a follow-up
integration, not a correctness gap in this module itself.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services import trade_manager as tm
from app.services.walk_forward_backtest import replay_recommendation_at

# Same warmup this replays against in walk_forward_backtest.py - enough
# history for SMA200 + its 25-bar slope lookback to be valid.
WARMUP_BARS = 260

# Same conservative, explicitly-not-measured assumption opportunity_cost.py
# already documents (MIN_EXPECTED_EDGE_AFTER_COSTS) - no calibrated cost
# model exists yet, so this is a round, documented estimate, not a fact.
ROUND_TRIP_COST_PCT = 0.002

VERTICAL_BARRIER_HORIZONS = (10, 21)  # trading sessions - this portfolio's actual holding horizon, not 63/126


@dataclass(frozen=True, slots=True)
class TripleBarrierLabel:
    entry_index: int
    exit_index: int
    exit_reason: str  # "stop" | "target" | "vertical"
    entry_price: float
    exit_price: float
    return_pct: float
    bars_held: int
    mae_pct: float  # Maximum Adverse Excursion - most negative intrabar move seen before exit (<=0)
    mfe_pct: float  # Maximum Favorable Excursion - most positive intrabar move seen before exit (>=0)


def label_triple_barrier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    entry_index: int,
    stop_loss: float,
    take_profit: float,
    vertical_barrier_bars: int,
    trailing: bool = False,
    atr14: pd.Series | None = None,
    vol_regime: str | None = None,
    open_: pd.Series | None = None,
) -> TripleBarrierLabel | None:
    """Simulates one trade opened at `entry_index`'s close, walking forward
    bar by bar until the stop, the target, or the vertical (time) barrier is
    hit - whichever comes first. Uses each bar's high/low, not just its
    close, to decide if a barrier was touched intrabar. If both the stop and
    target are touched within the *same* bar, the stop wins - the
    conservative convention (the brief's own instruction), since there is no
    way to know from daily bars alone which was actually touched first.

    `open_`, when given, makes the fill gap-aware: a bar that *opens* beyond
    a barrier (an overnight gap through the stop, or a gap-up through the
    target) never actually traded at the barrier price - the barrier level
    was skipped over, not touched and filled. The fill is the worse of
    (open, stop) for a stop-out and the better of (open, target) for a
    target hit. Segunda auditoría, Bloque 2: without this, a gap to 62 with
    a stop at 95 booked as -5% (filled at the unreachable stop) instead of
    the real -38% (filled at the open), inflating expectancy and profit
    factor on exactly the bars that hurt the most. `None` (the default) keeps
    the old, gap-blind fill - honest about not inventing an open price that
    wasn't actually passed in.

    `trailing=True` runs `trade_manager.py`'s actual Chandelier Exit logic
    bar by bar instead of a fixed stop - tests the strategy this system
    would really run (Fase 3), not a simplified stand-in. Requires `atr14`
    (an ATR series aligned with `close`/`high`/`low`).
    """
    if entry_index >= len(close) - 1:
        return None
    entry_price = float(close.iloc[entry_index])
    if stop_loss >= entry_price or take_profit <= entry_price:
        return None

    current_stop = stop_loss
    highest_high_since_entry = entry_price
    mae_pct = 0.0
    mfe_pct = 0.0

    max_index = min(entry_index + vertical_barrier_bars, len(close) - 1)
    if max_index <= entry_index:
        return None

    for i in range(entry_index + 1, max_index + 1):
        bar_high = float(high.iloc[i])
        bar_low = float(low.iloc[i])
        bar_open = float(open_.iloc[i]) if open_ is not None else None

        mae_pct = min(mae_pct, (bar_low - entry_price) / entry_price)
        mfe_pct = max(mfe_pct, (bar_high - entry_price) / entry_price)

        # Checked against the stop as it stood *coming into* this bar, before
        # today's own high can raise it - the same causal ordering
        # portfolio_risk_service.py uses live (evaluate_exit judges today's
        # close against the stop already in force, only *then* recomputes
        # the trailing level for the next evaluation). Updating first and
        # checking second would let a single bar's own high "predict" and
        # raise the stop that its own low is then tested against - a subtle
        # same-bar look-ahead.
        stop_touched = bar_low <= current_stop
        target_touched = bar_high >= take_profit

        if stop_touched:  # stop wins on a same-bar tie - see docstring
            # A gap-down open trades through the stop before it can ever be
            # filled there - `bar_open` is always <= `bar_high` and >=
            # `bar_low` by construction, so this is a no-op (fill == stop)
            # on any bar that didn't actually gap.
            fill_price = min(bar_open, current_stop) if bar_open is not None else current_stop
            return TripleBarrierLabel(
                entry_index, i, "stop", entry_price, fill_price, fill_price / entry_price - 1,
                i - entry_index, mae_pct, mfe_pct,
            )
        if target_touched:
            fill_price = max(bar_open, take_profit) if bar_open is not None else take_profit
            return TripleBarrierLabel(
                entry_index, i, "target", entry_price, fill_price, fill_price / entry_price - 1,
                i - entry_index, mae_pct, mfe_pct,
            )

        if trailing and atr14 is not None:
            highest_high_since_entry = max(highest_high_since_entry, bar_high)
            latest_atr = atr14.iloc[i]
            if not pd.isna(latest_atr):
                # Segunda auditoría, Bloque 2: recomputed at each bar from
                # THIS bar's close against the ORIGINAL (never-mutated)
                # stop_loss parameter - never the currently-trailing
                # `current_stop` - as the risk denominator, so the >=2R
                # profit-lock multiplier (chandelier_multiplier) can engage
                # mid-simulation exactly like the live system, instead of
                # being permanently `None` and never tightening.
                risk_per_share = entry_price - stop_loss
                bar_close = float(close.iloc[i])
                r_multiple = (bar_close - entry_price) / risk_per_share if risk_per_share > 0 else None
                multiplier = tm.chandelier_multiplier(vol_regime, r_multiple)
                candidate = highest_high_since_entry - multiplier * float(latest_atr)
                current_stop = tm.update_trailing_stop(current_stop, candidate)

    exit_price = float(close.iloc[max_index])
    return TripleBarrierLabel(
        entry_index, max_index, "vertical", entry_price, exit_price, exit_price / entry_price - 1,
        max_index - entry_index, mae_pct, mfe_pct,
    )


@dataclass(frozen=True, slots=True)
class TradingMetrics:
    n_trades: int
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy_pct: float | None  # win_rate * avg_win + loss_rate * avg_loss (avg_loss already negative)
    profit_factor: float | None  # sum(wins) / |sum(losses)| - None if there are no losses to divide by
    avg_mae_pct: float | None
    avg_mfe_pct: float | None
    max_drawdown_pct: float | None  # of the equity curve built by chaining trade returns in order
    avg_bars_held_winners: float | None
    avg_bars_held_losers: float | None
    gross_return_pct: float | None  # mean trade return, before costs
    net_return_pct: float | None  # after ROUND_TRIP_COST_PCT - docs/quant_methodology.md §7 flagged this gap


_EMPTY_METRICS = TradingMetrics(0, None, None, None, None, None, None, None, None, None, None, None, None)


def compute_trading_metrics(labels: list[TripleBarrierLabel]) -> TradingMetrics:
    """Segunda auditoría, Bloque 2: every P&L-derived stat here - win_rate,
    avg_win/avg_loss, expectancy, profit_factor, the equity curve and its
    drawdown - is computed net of `ROUND_TRIP_COST_PCT`, not just the summary
    `net_return_pct` scalar. A trade whose gross move is smaller than the
    round-trip cost is a real loss, not a "win" with an asterisk, and must
    count as one everywhere a win/loss split happens. `avg_mae_pct`/
    `avg_mfe_pct` stay gross on purpose - they describe intrabar price
    action (how far the trade wandered), not P&L, and costs don't apply to a
    price level that was never actually realized as an entry/exit."""
    if not labels:
        return _EMPTY_METRICS

    gross_returns = [label.return_pct for label in labels]
    net_returns = [r - ROUND_TRIP_COST_PCT for r in gross_returns]

    wins = [r for r in net_returns if r > 0]
    losses = [r for r in net_returns if r <= 0]
    n = len(labels)
    win_rate = len(wins) / n
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    sum_wins = float(sum(wins))
    sum_losses = abs(float(sum(losses)))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else None

    # Baseline of 1.0 prepended *before* the first trade - without it, the
    # first trade's own loss is invisible to the drawdown calculation (my own
    # bug from the previous round: see quant_methodology.md §13).
    equity_curve = np.cumprod([1.0] + [1 + r for r in net_returns])
    running_max = np.maximum.accumulate(equity_curve)
    max_drawdown = float(((equity_curve - running_max) / running_max).min())

    winners_bars = [label.bars_held for label, r in zip(labels, net_returns, strict=True) if r > 0]
    losers_bars = [label.bars_held for label, r in zip(labels, net_returns, strict=True) if r <= 0]

    gross_mean = float(np.mean(gross_returns))
    net_mean = float(np.mean(net_returns))

    return TradingMetrics(
        n_trades=n,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        expectancy_pct=expectancy,
        profit_factor=profit_factor,
        avg_mae_pct=float(np.mean([label.mae_pct for label in labels])),
        avg_mfe_pct=float(np.mean([label.mfe_pct for label in labels])),
        max_drawdown_pct=max_drawdown,
        avg_bars_held_winners=float(np.mean(winners_bars)) if winners_bars else None,
        avg_bars_held_losers=float(np.mean(losers_bars)) if losers_bars else None,
        gross_return_pct=gross_mean,
        net_return_pct=net_mean,
    )


def split_by_date(index: pd.DatetimeIndex, cutoff: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for a simple temporal train/test split (the brief's
    2016-2022 / 2023-2026 partition) - everything strictly before `cutoff` is
    train, everything on/after is test. Never mixed: a weight calibrated on
    the train mask is only ever judged against the test mask."""
    train_mask = np.asarray(index < cutoff)
    test_mask = ~train_mask
    return train_mask, test_mask


def purged_kfold_splits(
    n_samples: int, n_splits: int, embargo_fraction: float = 0.01
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold with embargo (López de Prado, cap. 7): plain K-Fold on
    time-series data leaks information between adjacent folds whenever a
    label's own forward-looking window (the vertical barrier) overlaps the
    fold boundary - a sample near the end of a train fold can have "seen"
    price action that its neighbor in the test fold's label was computed
    from. This purges an `embargo_fraction`-sized band of samples immediately
    after each test fold from the corresponding train fold, so no train
    sample's label window can leak into the adjacent test fold.

    Returns a list of (train_indices, test_indices) index arrays into a
    samples array of length `n_samples`, ordered chronologically (index 0 =
    earliest sample)."""
    if n_splits < 2 or n_samples < n_splits:
        return []
    indices = np.arange(n_samples)
    fold_bounds = np.array_split(indices, n_splits)
    embargo = max(1, int(n_samples * embargo_fraction))

    splits = []
    for test_idx in fold_bounds:
        test_start, test_end = test_idx[0], test_idx[-1]
        purge_start = max(0, test_start - embargo)
        purge_end = min(n_samples, test_end + embargo + 1)
        train_idx = np.array([i for i in indices if i < purge_start or i >= purge_end])
        splits.append((train_idx, test_idx))
    return splits


def buy_and_hold_labels(close: pd.Series, entry_indices: list[int], horizon_bars: int) -> list[TripleBarrierLabel]:
    """The honest benchmark the brief asks for: what buy-and-hold to the same
    vertical horizon would have returned from the exact same entry points,
    with no stop/target at all (a "barrier" so wide it's never touched -
    modeled directly rather than by disabling the barrier logic, so this
    reuses the same label shape/metrics as the real strategy)."""
    labels = []
    for entry_index in entry_indices:
        if entry_index >= len(close) - 1:
            continue
        max_index = min(entry_index + horizon_bars, len(close) - 1)
        if max_index <= entry_index:
            continue
        entry_price = float(close.iloc[entry_index])
        exit_price = float(close.iloc[max_index])
        window = close.iloc[entry_index : max_index + 1]
        mae_pct = float((window.min() - entry_price) / entry_price)
        mfe_pct = float((window.max() - entry_price) / entry_price)
        labels.append(
            TripleBarrierLabel(
                entry_index, max_index, "vertical", entry_price, exit_price, exit_price / entry_price - 1,
                max_index - entry_index, min(mae_pct, 0.0), max(mfe_pct, 0.0),
            )
        )
    return labels


def random_entry_labels(
    close: pd.Series, high: pd.Series, low: pd.Series, n_trades: int, horizon_bars: int, atr14: pd.Series,
    atr_stop_multiple: float, reward_risk_ratio: float, seed: int = 0, open_: pd.Series | None = None,
) -> list[TripleBarrierLabel]:
    """The "no skill" comparator the brief asks for: `n_trades` entries at
    uniformly random bars, sized with the exact same ATR-based stop/target
    formula `compute_stop_and_target` uses for a real signal (so this isn't
    a strawman with worse risk management, only a strategy with no actual
    edge in *when* it enters) - if the real strategy's expectancy doesn't
    clear this, timing carries no edge over picking days at random."""
    rng = np.random.default_rng(seed)
    atr_valid = atr14.notna().to_numpy()
    candidates = [i for i in range(len(close) - 1) if atr_valid[i]]
    if not candidates:
        return []
    chosen = rng.choice(candidates, size=min(n_trades, len(candidates)), replace=False)

    labels = []
    for entry_index in sorted(int(i) for i in chosen):
        entry_price = float(close.iloc[entry_index])
        atr_t = float(atr14.iloc[entry_index])
        if atr_t <= 0:
            continue
        stop = entry_price - atr_stop_multiple * atr_t
        risk = entry_price - stop
        target = entry_price + reward_risk_ratio * risk
        label = label_triple_barrier(close, high, low, entry_index, stop, target, horizon_bars, open_=open_)
        if label is not None:
            labels.append(label)
    return labels


@dataclass(frozen=True, slots=True)
class TripleBarrierBacktestResult:
    horizon_days: int
    n_signals_evaluated: int  # "comprar" signals with a resolvable stop/target - the tradeable subset
    strategy_fixed: TradingMetrics  # the proposed stop/target, held fixed
    strategy_trailing: TradingMetrics  # the same entries, run through trade_manager's real Chandelier trail
    buy_and_hold: TradingMetrics  # honest benchmark: same entries, no stop/target at all
    random_entries: TradingMetrics  # honest benchmark: random entries, same ATR-based sizing


def run_triple_barrier_backtest(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    sma20: pd.Series,
    sma50: pd.Series,
    sma150: pd.Series,
    sma200: pd.Series,
    rsi14: pd.Series,
    adx14: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    atr14: pd.Series,
    horizon_days: int,
    volume: pd.Series | None = None,
    vol_regime: str | None = None,
    warmup_bars: int = WARMUP_BARS,
) -> TripleBarrierBacktestResult | None:
    """Walks the same non-overlapping sampling grid `walk_forward_backtest.py`
    uses (stride == horizon, avoiding the classic overlapping-window pseudo-
    replication trap), replaying the full recommendation - verdict *and*
    stop/target - at each point via `walk_forward_backtest.replay_recommendation_at`,
    the single source of truth for "what would the system have proposed
    here". Only "comprar" signals with a resolvable stop/target are actually
    tradeable and get labeled - this validates the strategy the system
    actually proposes, not every bar indiscriminately.

    A thin per-ticker sample is a known, already-documented limitation this
    inherits from `walk_forward_backtest.py` (see docs/quant_methodology.md
    §3 on why the cross-sectional ablation study exists at 217-ticker scale
    instead) - `n_signals_evaluated` is reported plainly rather than hidden
    behind a silent `None` below some arbitrary sample-size floor, so a
    thin-sample result reads as thin, not as a confident answer.

    `open_` is required, not optional: every real OHLCV source already has
    it, and skipping gap-aware fills silently here would understate exactly
    the tail-risk trades this backtest exists to be honest about (Segunda
    auditoría, Bloque 2).
    """
    n = len(close)
    last_valid_start = n - horizon_days - 1
    if last_valid_start <= warmup_bars:
        return None

    entries: list[tuple[int, float, float]] = []
    for i in range(warmup_bars, last_valid_start, horizon_days):
        rec = replay_recommendation_at(
            i, close, sma20, sma50, sma150, sma200, rsi14, adx14, plus_di, minus_di, atr14, volume
        )
        if rec is None or rec.verdict != "comprar" or rec.stop_loss is None or rec.take_profit is None:
            continue
        entries.append((i, rec.stop_loss, rec.take_profit))

    if not entries:
        return TripleBarrierBacktestResult(
            horizon_days=horizon_days, n_signals_evaluated=0, strategy_fixed=_EMPTY_METRICS,
            strategy_trailing=_EMPTY_METRICS, buy_and_hold=_EMPTY_METRICS, random_entries=_EMPTY_METRICS,
        )

    fixed_labels = []
    trailing_labels = []
    for i, stop, target in entries:
        fixed = label_triple_barrier(close, high, low, i, stop, target, horizon_days, open_=open_)
        if fixed is not None:
            fixed_labels.append(fixed)
        trailing = label_triple_barrier(
            close, high, low, i, stop, target, horizon_days, trailing=True, atr14=atr14, vol_regime=vol_regime,
            open_=open_,
        )
        if trailing is not None:
            trailing_labels.append(trailing)

    entry_indices = [i for i, _, _ in entries]
    bnh_labels = buy_and_hold_labels(close, entry_indices, horizon_days)
    random_labels = random_entry_labels(
        close, high, low, len(entries), horizon_days, atr14, atr_stop_multiple=2.5, reward_risk_ratio=2.0,
        open_=open_,
    )

    return TripleBarrierBacktestResult(
        horizon_days=horizon_days,
        n_signals_evaluated=len(entries),
        strategy_fixed=compute_trading_metrics(fixed_labels),
        strategy_trailing=compute_trading_metrics(trailing_labels),
        buy_and_hold=compute_trading_metrics(bnh_labels),
        random_entries=compute_trading_metrics(random_labels),
    )
