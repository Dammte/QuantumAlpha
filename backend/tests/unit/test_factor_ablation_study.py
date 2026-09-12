"""Segunda auditoría, Bloque 5: unit tests for the new pure functions added to
scripts/factor_ablation_study.py (setup-type triggers/segmentation, temporal
train/validate split, point-in-time universe resolution). Not exercised
before this file existed - the script only had `docs/factor_ablation_report_v2_*.csv`
outputs from a real run as evidence, no synthetic-input regression tests.

Imported as `scripts.factor_ablation_study` - pytest's `pythonpath = ["."]`
(pyproject.toml) puts `backend/` on sys.path, and `scripts/` has no
`__init__.py`, so this relies on Python's implicit namespace packages (works
the same way the script's own `sys.path.insert` + `from app...` imports do)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import scripts.factor_ablation_study as fas
from app.services import technical_analysis as ta
from app.services.market_universe import universe_tickers

# --- compute_triggers_at: the four setup-type triggers (Bloque 3's setups,
# originally watchlist_service.py's own, that module retired 2026-09 - see
# PULLBACK_MAX_DISTANCE_ABOVE_SMA50's own comment in the script) ---


def _indicators(close: pd.Series, high: pd.Series, low: pd.Series) -> dict[str, pd.Series]:
    return dict(
        sma20=ta.sma(close, 20), sma50=ta.sma(close, 50),
        sma150=ta.sma(close, 150), sma200=ta.sma(close, 200),
        rsi14=ta.rsi(close), adx14=ta.adx(high, low, close), atr14=ta.atr(high, low, close),
    )


def _triggers_for_last_bar(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series
) -> dict[str, bool]:
    ind = _indicators(close, high, low)
    plus_di, minus_di = ta.dmi(high, low, close)
    result = fas.compute_triggers_at(
        len(close) - 1, close, high, low, volume,
        ind["sma20"], ind["sma50"], ind["sma150"], ind["sma200"],
        ind["rsi14"], ind["adx14"], plus_di, minus_di, ind["atr14"],
    )
    assert result is not None
    return result


def test_compute_triggers_at_setup_oversold_bounce_on_a_steep_decline_then_a_green_day():
    # 270 flat bars (warmup) -> 25 days of a steep 3%/day decline (pushes RSI
    # near 0, Wilder's ewm smoothing means the single bounce day barely moves
    # it) -> one +1% bounce day, which is both "RSI deeply oversold" and
    # "today is green" - exactly what the setup checks for.
    flat = [100.0] * 270
    decline = [100.0 * (0.97**k) for k in range(1, 26)]
    bounce = decline[-1] * 1.01
    close = pd.Series(flat + decline + [bounce])
    high, low = close + 1, close - 1
    volume = pd.Series([1_000_000.0] * len(close))
    triggers = _triggers_for_last_bar(close, high, low, volume)
    assert triggers["setup_oversold_bounce"]


def test_compute_triggers_at_setup_breakout_volume_on_a_new_high_with_a_volume_spike():
    # Monotonic uptrend -> today's close is the highest close in the trailing
    # 252 bars by construction (dist_52w_high == 0.0), and today's volume is
    # 3x the prior 20-day average (relative_volume == 3.0) - both comfortably
    # clear the setup's >= -0.02 / >= 1.3 thresholds.
    close = pd.Series(100 + np.arange(280) * 0.3)
    high, low = close + 1, close - 1
    volume = pd.Series([1_000_000.0] * 279 + [3_000_000.0])
    triggers = _triggers_for_last_bar(close, high, low, volume)
    assert triggers["setup_breakout_volume"]


def test_compute_triggers_at_setup_trend_continuation_in_a_clean_uptrend():
    # A steady, unbroken uptrend: +DI dominates -DI every single bar (low
    # never falls), so ADX climbs well past the 25 "trending" threshold, and
    # the last 5 sessions are (like every other stretch) positive.
    close = pd.Series(100 + np.arange(280) * 0.5)
    high, low = close + 1, close - 1
    volume = pd.Series([1_000_000.0] * 280)
    triggers = _triggers_for_last_bar(close, high, low, volume)
    assert triggers["setup_trend_continuation"]


def test_compute_triggers_at_setup_pullback_to_support_flattening_at_sma50():
    # A long uptrend (sma50 pulls well above sma200) followed by 50 flat bars
    # right at today's price - sma50 catches up to exactly today's close
    # (0% distance, comfortably under the 4% ceiling), sma200 is still well
    # below it, and RSI stays elevated (> 40) since there's never a down day.
    up = 100 + np.arange(230) * 0.3
    flat = [up[-1]] * 50
    close = pd.Series(np.concatenate([up, flat]))
    high, low = close + 1, close - 1
    volume = pd.Series([1_000_000.0] * len(close))
    triggers = _triggers_for_last_bar(close, high, low, volume)
    assert triggers["setup_pullback_to_support"]


def test_compute_triggers_at_no_setup_matches_a_flat_boring_series():
    close = pd.Series([100.0] * 280)
    high, low = close + 0.5, close - 0.5
    volume = pd.Series([1_000_000.0] * 280)
    triggers = _triggers_for_last_bar(close, high, low, volume)
    for key in fas.SETUP_TRIGGER_KEYS:
        assert not triggers[key], key


# --- compute_triggers_at: Fase 8 reorientation (gate factors) ---

_GATE_FACTOR_KEYS = (
    "gate_passes", "gate_trend_or_stage2", "gate_not_parabolic",
    "gate_not_overbought_outside_strong_trend", "gate_no_obv_bearish_divergence", "gate_no_fast_pair_veto",
)

# The three old checklist factors each gate_* factor above replaced (exact
# logical negations - see the module docstring's Fase 8 section for why
# keeping both would be a real collinearity defect).
_REMOVED_CHECKLIST_KEYS = ("atr_parabolic", "rsi_overbought_outside_strong_trend", "obv_bearish")


def test_compute_triggers_at_exposes_every_gate_factor_and_drops_its_superseded_checklist_twin():
    close = pd.Series(100 + np.arange(280) * 0.3)
    high, low = close + 1, close - 1
    volume = pd.Series([1_000_000.0] * 280)
    triggers = _triggers_for_last_bar(close, high, low, volume)

    for key in _GATE_FACTOR_KEYS:
        assert isinstance(triggers[key], bool), key
    for key in _REMOVED_CHECKLIST_KEYS:
        assert key not in triggers, key
    # The reward:risk gate condition is deliberately never exposed as its own
    # factor here (constant True without point-in-time support/resistance -
    # see the module docstring).
    assert not any("reward_risk" in key for key in triggers)


def test_compute_triggers_at_gate_passes_is_the_and_of_its_own_sub_conditions():
    # A clean, unbroken uptrend with no OBV divergence and no fast-pair
    # veto - every gate condition this replay can actually vary should read
    # True, and gate_passes should agree with their conjunction.
    close = pd.Series(100 + np.arange(280) * 0.3)
    high, low = close + 1, close - 1
    volume = pd.Series([1_000_000.0] * 280)
    triggers = _triggers_for_last_bar(close, high, low, volume)

    sub_conditions = (
        triggers["gate_trend_or_stage2"],
        triggers["gate_not_parabolic"],
        triggers["gate_not_overbought_outside_strong_trend"],
        triggers["gate_no_obv_bearish_divergence"],
        triggers["gate_no_fast_pair_veto"],
    )
    # gate_passes also requires reward:risk >= 1.5, which this point-in-time
    # replay always satisfies (see the module docstring) - so it must equal
    # the AND of the five conditions this test can actually vary.
    assert triggers["gate_passes"] == all(sub_conditions)


# --- segment_by_setup_type ---


_ALL_TRIGGER_KEYS = (
    "trend_up", "trend_down", "stage2", "stage4", "golden_cross", "death_cross", "adx_strong_trend",
    "rsi_oversold_bounce", "obv_bullish", "minervini_range_position", "market_below_sma200", "vix_stress",
    "gate_passes", "gate_trend_or_stage2", "gate_not_parabolic", "gate_not_overbought_outside_strong_trend",
    "gate_no_obv_bearish_divergence", "gate_no_fast_pair_veto", *fas.SETUP_TRIGGER_KEYS,
)


def _sample(
    ticker: str, date_str: str, fwd_return: float = 0.0, exit_reason: str = "vertical", bars_held: int = 21,
    mae_pct: float = -0.01, mfe_pct: float = 0.01, risk_pct: float = 0.02, **trigger_overrides: bool
) -> fas.FactorSample:
    triggers = dict.fromkeys(_ALL_TRIGGER_KEYS, False)
    triggers.update(trigger_overrides)
    return fas.FactorSample(
        ticker=ticker, date=pd.Timestamp(date_str), fwd_return=fwd_return, demeaned_return=fwd_return,
        triggers=triggers, exit_reason=exit_reason, bars_held=bars_held, mae_pct=mae_pct, mfe_pct=mfe_pct,
        risk_pct=risk_pct,
    )


def test_segment_by_setup_type_groups_by_matching_setup_and_allows_overlap():
    samples = [
        _sample("A", "2024-01-01", setup_oversold_bounce=True, setup_breakout_volume=True),
        _sample("B", "2024-01-02", setup_oversold_bounce=True),
        _sample("C", "2024-01-03"),  # matches nothing
    ]
    segments = fas.segment_by_setup_type(samples)
    assert set(segments.keys()) == {
        "oversold_bounce", "breakout_volume", "trend_continuation", "pullback_to_support",
    }
    assert [s.ticker for s in segments["oversold_bounce"]] == ["A", "B"]
    assert [s.ticker for s in segments["breakout_volume"]] == ["A"]  # A appears in both - not mutually exclusive
    assert segments["trend_continuation"] == []
    assert segments["pullback_to_support"] == []


def test_segment_by_setup_type_empty_input_returns_empty_segments():
    segments = fas.segment_by_setup_type([])
    assert all(segment == [] for segment in segments.values())


# --- split_samples_by_date ---


def test_split_samples_by_date_calibrate_strictly_before_cutoff_validate_on_or_after():
    samples = [
        _sample("A", "2022-01-01"),
        _sample("B", "2022-12-31"),
        _sample("C", "2023-01-01"),  # exactly the cutoff - goes to validate
        _sample("D", "2024-06-01"),
    ]
    calibrate, validate = fas.split_samples_by_date(samples)
    assert [s.ticker for s in calibrate] == ["A", "B"]
    assert [s.ticker for s in validate] == ["C", "D"]


def test_split_samples_by_date_respects_a_custom_cutoff():
    samples = [_sample("A", "2020-01-01"), _sample("B", "2021-06-01")]
    calibrate, validate = fas.split_samples_by_date(samples, cutoff=pd.Timestamp("2021-01-01"))
    assert [s.ticker for s in calibrate] == ["A"]
    assert [s.ticker for s in validate] == ["B"]


def test_split_samples_by_date_empty_input_returns_two_empty_lists():
    assert fas.split_samples_by_date([]) == ([], [])


# --- compute_setup_outcome_stats (Tercera auditoría, Bloque F-6) -----------


def _outcome_sample(
    ticker: str, exit_reason: str, bars_held: int, mae_pct: float, fwd_return: float, risk_pct: float = 0.02,
    **trigger_overrides: bool,
) -> fas.FactorSample:
    return _sample(
        ticker, "2024-01-01", fwd_return=fwd_return, exit_reason=exit_reason, bars_held=bars_held,
        mae_pct=mae_pct, mfe_pct=abs(fwd_return), risk_pct=risk_pct, **trigger_overrides,
    )


def test_compute_setup_outcome_stats_win_rate_counts_target_exits_only():
    # 18 winners (exit_reason="target"), 12 losers (exit_reason="stop") - a
    # clean, hand-countable 60% win rate for oversold_bounce.
    winners = [
        _outcome_sample(f"W{i}", "target", bars_held=6, mae_pct=-0.01, fwd_return=0.08, setup_oversold_bounce=True)
        for i in range(18)
    ]
    losers = [
        _outcome_sample(f"L{i}", "stop", bars_held=4, mae_pct=-0.04, fwd_return=-0.04, setup_oversold_bounce=True)
        for i in range(12)
    ]
    stats = fas.compute_setup_outcome_stats(winners + losers)
    assert stats["oversold_bounce"].n == 30
    assert stats["oversold_bounce"].win_rate == pytest.approx(0.6)


def test_compute_setup_outcome_stats_expectancy_r_uses_each_samples_own_risk_pct():
    # fwd_return=0.04, risk_pct=0.02 -> exactly +2R for every sample.
    samples = [
        _outcome_sample(f"T{i}", "target", bars_held=5, mae_pct=-0.005, fwd_return=0.04, risk_pct=0.02,
                        setup_breakout_volume=True)
        for i in range(35)
    ]
    stats = fas.compute_setup_outcome_stats(samples)
    assert stats["breakout_volume"].expectancy_r == pytest.approx(2.0)


def test_compute_setup_outcome_stats_median_bars_held():
    samples = [
        _outcome_sample(f"T{i}", "vertical", bars_held=bars, mae_pct=-0.01, fwd_return=0.01,
                        setup_trend_continuation=True)
        for i, bars in enumerate([3] * 15 + [21] * 15)  # even split -> median exactly between 3 and 21
    ]
    stats = fas.compute_setup_outcome_stats(samples)
    assert stats["trend_continuation"].median_bars_held == pytest.approx(12.0)


def test_compute_setup_outcome_stats_mae_p80_reflects_the_tail_not_the_average():
    # 24 samples with a small MAE, 6 with a much larger one - the 80th
    # percentile index falls exactly at the boundary between the two groups
    # (interpolated, so strictly between them), not a plain average.
    small_mae = [
        _outcome_sample(f"S{i}", "target", bars_held=5, mae_pct=-0.01, fwd_return=0.02,
                        setup_pullback_to_support=True)
        for i in range(24)
    ]
    big_mae = [
        _outcome_sample(f"B{i}", "stop", bars_held=5, mae_pct=-0.15, fwd_return=-0.05,
                        setup_pullback_to_support=True)
        for i in range(6)
    ]
    stats = fas.compute_setup_outcome_stats(small_mae + big_mae)
    assert stats["pullback_to_support"].mae_p80_pct > 0.01  # pulled up by the tail
    assert stats["pullback_to_support"].mae_p80_pct < 0.15  # but not all the way to the extreme


def test_compute_setup_outcome_stats_omits_a_setup_with_too_few_samples():
    samples = [
        _outcome_sample(f"T{i}", "target", bars_held=5, mae_pct=-0.01, fwd_return=0.02, setup_oversold_bounce=True)
        for i in range(fas.MIN_GROUP_SIZE - 1)
    ]
    assert fas.compute_setup_outcome_stats(samples) == {}


def test_compute_setup_outcome_stats_empty_input():
    assert fas.compute_setup_outcome_stats([]) == {}


# --- resolve_universe_tickers ---


class _FakeSession:
    def close(self) -> None:
        pass


def test_resolve_universe_tickers_default_uses_the_curated_dict():
    # Tercera auditoría, Bloque F-1: returns ticker -> region now, not a flat
    # list - filter_samples_by_point_in_time_membership needs to know which
    # region's snapshot history to check each ticker against.
    result = fas.resolve_universe_tickers(["us"], use_dynamic_universe=False)
    assert set(result) == set(universe_tickers("us"))
    assert all(region == "us" for region in result.values())


def test_resolve_universe_tickers_dynamic_uses_the_point_in_time_snapshot_when_present(monkeypatch):
    monkeypatch.setattr(fas, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(fas, "UniverseMembershipRepository", lambda db: object())
    monkeypatch.setattr(fas.dus, "read_dynamic_universe", lambda repo, region: {"AAPL": "Technology"})
    result = fas.resolve_universe_tickers(["us"], use_dynamic_universe=True)
    assert result == {"AAPL": "us"}


def test_resolve_universe_tickers_dynamic_falls_back_to_curated_per_region_with_no_snapshot(monkeypatch):
    # "us" has a live snapshot, "europe" doesn't yet - each region falls back
    # independently, never all-or-nothing across the whole call.
    monkeypatch.setattr(fas, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(fas, "UniverseMembershipRepository", lambda db: object())
    monkeypatch.setattr(
        fas.dus, "read_dynamic_universe",
        lambda repo, region: {"AAPL": "Technology"} if region == "us" else None,
    )
    result = fas.resolve_universe_tickers(["us", "europe"], use_dynamic_universe=True)
    assert result["AAPL"] == "us"
    assert set(universe_tickers("europe")) <= set(result)
    assert all(result[t] == "europe" for t in universe_tickers("europe"))


# --- _resolve_as_of_snapshot_date / filter_samples_by_point_in_time_membership ---


def test_resolve_as_of_snapshot_date_picks_the_latest_on_or_before():
    dates = [date(2024, 1, 1), date(2024, 6, 1), date(2025, 1, 1)]
    assert fas._resolve_as_of_snapshot_date(dates, date(2024, 7, 1)) == date(2024, 6, 1)


def test_resolve_as_of_snapshot_date_exact_match():
    dates = [date(2024, 1, 1), date(2024, 6, 1)]
    assert fas._resolve_as_of_snapshot_date(dates, date(2024, 6, 1)) == date(2024, 6, 1)


def test_resolve_as_of_snapshot_date_falls_back_to_earliest_when_sample_predates_everything():
    # The honest limitation: a 2016 sample with the only snapshot on file
    # dated 2026 gets that 2026 snapshot - no worse than the pre-fix bug,
    # and correct once earlier snapshots eventually exist.
    dates = [date(2026, 8, 1)]
    assert fas._resolve_as_of_snapshot_date(dates, date(2016, 1, 1)) == date(2026, 8, 1)


def test_resolve_as_of_snapshot_date_none_when_no_snapshots_at_all():
    assert fas._resolve_as_of_snapshot_date([], date(2020, 1, 1)) is None


def test_filter_samples_by_point_in_time_membership_is_a_no_op_for_curated_universe():
    samples = [_sample("AAPL", "2020-01-01"), _sample("MSFT", "2020-01-01")]
    result = fas.filter_samples_by_point_in_time_membership(samples, {}, use_dynamic_universe=False)
    assert result == samples


def test_filter_samples_by_point_in_time_membership_empty_input():
    assert fas.filter_samples_by_point_in_time_membership([], {"AAPL": "us"}, use_dynamic_universe=True) == []


def test_filter_samples_by_point_in_time_membership_drops_a_non_member_ticker(monkeypatch):
    # AAPL was a member as of the snapshot on/before the sample date; MSFT
    # was not (e.g. dropped from the index by then, per that snapshot).
    monkeypatch.setattr(fas, "SessionLocal", lambda: _FakeSession())

    class _FakeRepo:
        def all_as_of_dates(self, region):
            return [date(2020, 1, 1)]

    monkeypatch.setattr(fas, "UniverseMembershipRepository", lambda db: _FakeRepo())
    monkeypatch.setattr(fas.dus, "read_dynamic_universe", lambda repo, region, as_of_date=None: {"AAPL": "Tech"})

    samples = [_sample("AAPL", "2020-06-01"), _sample("MSFT", "2020-06-01")]
    result = fas.filter_samples_by_point_in_time_membership(
        samples, {"AAPL": "us", "MSFT": "us"}, use_dynamic_universe=True
    )
    assert [s.ticker for s in result] == ["AAPL"]


# --- _permutation_test: moved here unchanged (2026-09, reconstruction Fase 4)
# from the retired walk_forward_backtest.py - this script was always its only
# real consumer (a generic statistics routine, no dependency on that
# module's own retired scoring replay). See docs/quant_methodology.md.


def test_permutation_test_high_p_value_for_identical_distributions():
    rng = np.random.default_rng(1)
    sample_a = rng.normal(0, 0.02, 40)
    sample_b = rng.normal(0, 0.02, 40)
    p = fas._permutation_test(sample_a, sample_b, n_permutations=2000, seed=1)
    assert p > 0.05


def test_permutation_test_low_p_value_for_clearly_different_means():
    rng = np.random.default_rng(2)
    sample_a = rng.normal(0.05, 0.01, 40)
    sample_b = rng.normal(-0.05, 0.01, 40)
    p = fas._permutation_test(sample_a, sample_b, n_permutations=2000, seed=2)
    assert p < 0.01
