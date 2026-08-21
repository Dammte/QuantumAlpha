"""Segunda auditoría, Bloque 5: unit tests for the new pure functions added to
scripts/factor_ablation_study.py (setup-type triggers/segmentation, temporal
train/validate split, point-in-time universe resolution). Not exercised
before this file existed - the script only had `docs/factor_ablation_report_v2_*.csv`
outputs from a real run as evidence, no synthetic-input regression tests.

Imported as `scripts.factor_ablation_study` - pytest's `pythonpath = ["."]`
(pyproject.toml) puts `backend/` on sys.path, and `scripts/` has no
`__init__.py`, so this relies on Python's implicit namespace packages (works
the same way the script's own `sys.path.insert` + `from app...` imports do)."""

import numpy as np
import pandas as pd

import scripts.factor_ablation_study as fas
from app.services import technical_analysis as ta
from app.services.market_universe import universe_tickers

# --- compute_triggers_at: the four setup-type triggers (Bloque 3's setups,
# duplicated here against raw indicator series - see the function's own
# docstring for why it can't just call watchlist_service's detectors
# directly) ---


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


# --- segment_by_setup_type ---


_ALL_TRIGGER_KEYS = (
    "trend_up", "trend_down", "stage2", "stage4", "golden_cross", "death_cross", "adx_strong_trend",
    "rsi_overbought_outside_strong_trend", "rsi_oversold_bounce", "atr_parabolic", "obv_bearish",
    "obv_bullish", "minervini_range_position", "market_below_sma200", "vix_stress", *fas.SETUP_TRIGGER_KEYS,
)


def _sample(ticker: str, date_str: str, fwd_return: float = 0.0, **trigger_overrides: bool) -> fas.FactorSample:
    triggers = dict.fromkeys(_ALL_TRIGGER_KEYS, False)
    triggers.update(trigger_overrides)
    return fas.FactorSample(
        ticker=ticker, date=pd.Timestamp(date_str), fwd_return=fwd_return, demeaned_return=fwd_return,
        triggers=triggers,
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


# --- resolve_universe_tickers ---


class _FakeSession:
    def close(self) -> None:
        pass


def test_resolve_universe_tickers_default_uses_the_curated_dict():
    result = fas.resolve_universe_tickers(["us"], use_dynamic_universe=False)
    assert result == sorted(set(universe_tickers("us")))


def test_resolve_universe_tickers_dynamic_uses_the_point_in_time_snapshot_when_present(monkeypatch):
    monkeypatch.setattr(fas, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(fas, "UniverseMembershipRepository", lambda db: object())
    monkeypatch.setattr(fas.dus, "read_dynamic_universe", lambda repo, region: {"AAPL": "Technology"})
    result = fas.resolve_universe_tickers(["us"], use_dynamic_universe=True)
    assert result == ["AAPL"]


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
    assert "AAPL" in result
    assert set(universe_tickers("europe")) <= set(result)
