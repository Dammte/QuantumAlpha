from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import premium_watchlist_service as pws
from app.services import technical_analysis as ta


def _signals(verdict="comprar", score=6, has_sizing=False, entry_timing=None):
    position_sizing = SimpleNamespace() if has_sizing else None
    return SimpleNamespace(
        recommendation=SimpleNamespace(verdict=verdict, score=score),
        backtest=None,
        monte_carlo=None,
        position_sizing=position_sizing,
        entry_timing=entry_timing,
    )


# --- _approval_score: the objectivity gate ----------------------------------
#
# Segunda auditoría, Bloque 3: BACKTEST_EDGE_BONUS/KELLY_SETUP_BONUS and the
# backtest_contradicts rejection are gone (see _approval_score's own
# docstring for why) - MONTE_CARLO_EDGE_WEIGHT is replaced by
# SETUP_PERCENTILE_BONUS_WEIGHT, driven by the candidate's cross-sectional
# setup percentile (watchlist_service.setup_percentile_score), not Monte
# Carlo's own (near-circular) stop/target simulation.


def test_rejects_non_comprar_verdict():
    assert pws._approval_score(_signals(verdict="esperar")) is None
    assert pws._approval_score(_signals(verdict="evitar")) is None


def test_setup_percentile_above_50_adds_a_positive_bonus():
    base = pws._approval_score(_signals(score=6))
    boosted = pws._approval_score(_signals(score=6), setup_percentile=100.0)
    assert boosted - base == pytest.approx(pws.SETUP_PERCENTILE_BONUS_WEIGHT)


def test_setup_percentile_below_50_adds_a_negative_bonus():
    base = pws._approval_score(_signals(score=6))
    penalized = pws._approval_score(_signals(score=6), setup_percentile=0.0)
    assert base - penalized == pytest.approx(pws.SETUP_PERCENTILE_BONUS_WEIGHT)


def test_setup_percentile_of_exactly_50_is_neutral():
    base = pws._approval_score(_signals(score=6))
    neutral = pws._approval_score(_signals(score=6), setup_percentile=50.0)
    assert neutral == pytest.approx(base)


def test_missing_setup_percentile_gets_no_bonus():
    """The weekly/monthly tiers don't carry a setup percentile at all - see
    build_watchlist - and get no substitute bonus, not a guessed one."""
    base = pws._approval_score(_signals(score=6))
    same = pws._approval_score(_signals(score=6), setup_percentile=None)
    assert same == pytest.approx(base)


def test_strong_sector_adds_flat_bonus():
    base = pws._approval_score(_signals(score=6))
    boosted = pws._approval_score(_signals(score=6), sector_rs_rank=pws.STRONG_SECTOR_RS_THRESHOLD)
    assert boosted - base == pytest.approx(pws.STRONG_SECTOR_BONUS)


def test_weak_sector_gets_no_bonus():
    base = pws._approval_score(_signals(score=6))
    same = pws._approval_score(_signals(score=6), sector_rs_rank=pws.STRONG_SECTOR_RS_THRESHOLD - 1)
    assert same == pytest.approx(base)


def test_missing_sector_rs_rank_gets_no_bonus():
    """A sector whose own RS rank couldn't be computed right now contributes
    nothing, rather than guessing - same graceful-degradation posture as
    everywhere else a cross-sectional rank is used in this codebase."""
    base = pws._approval_score(_signals(score=6))
    same = pws._approval_score(_signals(score=6), sector_rs_rank=None)
    assert same == pytest.approx(base)


def _timing(status: str):
    return SimpleNamespace(status=status)


def test_extended_entry_timing_gets_a_ranking_penalty():
    """Found auditing a real premium daily list: half the candidates were
    already "extended" per entry_timing, undermining a list specifically
    meant to be actionable *today*. A fresher, equally-scored setup should
    outrank an already-extended one, without excluding the extended one
    outright (it's still a legitimate "comprar" - see entry_timing.py)."""
    base = pws._approval_score(_signals(score=6, entry_timing=_timing("valid")))
    extended = pws._approval_score(_signals(score=6, entry_timing=_timing("extended")))
    assert base - extended == pytest.approx(pws.EXTENDED_ENTRY_PENALTY)


def test_optimal_or_missing_entry_timing_gets_no_penalty():
    base = pws._approval_score(_signals(score=6, entry_timing=None))
    optimal = pws._approval_score(_signals(score=6, entry_timing=_timing("optimal")))
    late = pws._approval_score(_signals(score=6, entry_timing=_timing("late")))
    assert optimal == pytest.approx(base)
    assert late == pytest.approx(base)


# --- build_premium_watchlist: orchestration ----------------------------------


def _snapshot(ticker: str, rs_rating: int = 80, sector: str = "Tecnología") -> TickerSnapshot:
    return TickerSnapshot(
        ticker=ticker,
        sector=sector,
        industry=None,
        cap_tier="mega",
        price=100.0,
        change_1d=0.01,
        change_1w=0.02,
        change_1m=0.03,
        change_3m=0.05,
        change_6m=0.08,
        change_1y=0.15,
        volume=1_000_000.0,
        relative_volume=1.2,
        rsi14=60.0,
        sma20=99.0,
        sma50=95.0,
        sma150=90.0,
        sma200=85.0,
        dist_52w_high=-0.02,
        dist_52w_low=0.30,
        atr_multiple=1.5,
        adx14=28.0,
        plus_di=25.0,
        minus_di=15.0,
        mansfield_rs=0.02,
        trend=ta.TrendState.UPTREND,
        stage=ta.Stage.STAGE_2,
        ma_cross=None,
        minervini_score=8,
        minervini_pass=True,
        rs_rating=rs_rating,
    )


class _StubMarketData:
    def __init__(
        self, tickers_with_data: set[str], days_to_earnings_by_ticker: dict[str, int] | None = None
    ) -> None:
        self.tickers_with_data = tickers_with_data
        self.requested: list[str] = []
        self._days_to_earnings_by_ticker = days_to_earnings_by_ticker or {}

    def get_bulk_ohlcv(self, tickers, start, end):
        self.requested = list(tickers)
        # Minimal stand-in for a DataFrame: subscriptable by column name only.
        fake_frame = {"close": None, "high": None, "low": None, "volume": None, "open": None}
        return {t: fake_frame for t in tickers if t in self.tickers_with_data}

    def get_next_earnings_date(self, ticker):
        days = self._days_to_earnings_by_ticker.get(ticker)
        if days is None:
            return None
        return date.today() + timedelta(days=days)


def test_build_premium_watchlist_caps_candidates_and_approved_per_tier(monkeypatch):
    # 20 daily-tier-worthy snapshots (all pass the cheap short-term rule via strong ADX/volume)
    snapshots = [_snapshot(f"T{i}", rs_rating=99 - i) for i in range(20)]
    market_data = _StubMarketData({s.ticker for s in snapshots} | {pws.benchmark_for_region("us")})

    monkeypatch.setattr(
        pws,
        "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=rs_rating or 0),
    )

    # Every candidate approved (score always positive, verdict always "comprar")
    results, discard_stats = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])

    assert len(results) <= pws.MAX_APPROVED_PER_TIER
    assert all(r.tier == pws.DAILY for r in results)
    # Highest-score-first
    assert [r.premium_score for r in results] == sorted((r.premium_score for r in results), reverse=True)

    stats = discard_stats[pws.DAILY]
    assert stats.prefilter_matches == 20
    assert stats.analyzed == pws.MAX_CANDIDATES_PER_TIER
    assert stats.approved == len(results)


def test_build_premium_watchlist_skips_tickers_missing_from_ohlcv(monkeypatch):
    snapshots = [_snapshot("HASDATA"), _snapshot("NODATA")]
    market_data = _StubMarketData({"HASDATA", pws.benchmark_for_region("us")})  # NODATA deliberately absent

    monkeypatch.setattr(
        pws,
        "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=10),
    )

    results, _ = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])
    assert [r.ticker for r in results] == ["HASDATA"]


def test_build_premium_watchlist_empty_universe_returns_empty(monkeypatch):
    market_data = _StubMarketData(set())
    results, discard_stats = pws.build_premium_watchlist([], market_data)
    assert results == []
    assert all(s.prefilter_matches == 0 and s.analyzed == 0 and s.approved == 0 for s in discard_stats.values())


def test_build_premium_watchlist_isolates_a_candidate_whose_compute_raises(monkeypatch):
    """The exact production bug this test locks in: one candidate's GARCH
    optimizer failing to converge, a backtest edge case, or any other
    numerical hiccup on up to 15 tickers a request must never take the whole
    tier down with it (previously an uncaught exception propagated straight
    to a 500 on the whole premium watchlist response)."""
    good = _snapshot("GOOD", rs_rating=50)
    bad = _snapshot("BAD", rs_rating=99)
    market_data = _StubMarketData({"GOOD", "BAD", pws.benchmark_for_region("us")})

    def flaky_compute(close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None,
                      ticker=None):
        # Triggered by ticker, not rs_rating (Tercera auditoría, Bloque F-3:
        # the daily tier no longer passes rs_rating through unconverted).
        if ticker == "BAD":
            raise ValueError("simulated GARCH/backtest numerical failure")
        return _signals(score=10)

    monkeypatch.setattr(pws, "compute_core_signals", flaky_compute)

    results, _ = pws.build_premium_watchlist([good, bad], market_data, tiers=[pws.DAILY])
    assert [r.ticker for r in results] == ["GOOD"]


def test_build_premium_watchlist_ranks_strong_sector_candidate_above_equal_scoring_weak_sector_one(monkeypatch):
    strong = _snapshot("STRONG_SECTOR", rs_rating=80, sector="Tecnología")
    weak = _snapshot("WEAK_SECTOR", rs_rating=80, sector="Utilities")
    market_data = _StubMarketData({"STRONG_SECTOR", "WEAK_SECTOR", pws.benchmark_for_region("us")})

    # Identical underlying score for both - only the sector should break the tie.
    monkeypatch.setattr(
        pws,
        "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=6),
    )

    results, _ = pws.build_premium_watchlist(
        [strong, weak],
        market_data,
        tiers=[pws.DAILY],
        sector_rs_rank={"Tecnología": 90, "Utilities": 20},
    )

    assert [r.ticker for r in results] == ["STRONG_SECTOR", "WEAK_SECTOR"]
    strong_result = next(r for r in results if r.ticker == "STRONG_SECTOR")
    weak_result = next(r for r in results if r.ticker == "WEAK_SECTOR")
    assert strong_result.premium_score - weak_result.premium_score == pytest.approx(pws.STRONG_SECTOR_BONUS)


def test_build_premium_watchlist_items_carry_their_setup_type(monkeypatch):
    # All 20 snapshots match trend_continuation (strong, confirmed ADX trend).
    snapshots = [_snapshot(f"T{i}", rs_rating=99 - i) for i in range(20)]
    market_data = _StubMarketData({s.ticker for s in snapshots} | {pws.benchmark_for_region("us")})
    monkeypatch.setattr(
        pws,
        "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=rs_rating or 0),
    )
    results, _ = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])
    assert all(r.setup == "trend_continuation" for r in results)


def test_build_premium_watchlist_discards_the_excess_beyond_max_candidates_per_tier(monkeypatch):
    snapshots = [_snapshot(f"T{i}") for i in range(pws.MAX_CANDIDATES_PER_TIER + 5)]
    market_data = _StubMarketData({s.ticker for s in snapshots} | {pws.benchmark_for_region("us")})
    monkeypatch.setattr(
        pws, "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=6),
    )
    _, discard_stats = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.WEEKLY])
    stats = discard_stats[pws.WEEKLY]
    assert stats.prefilter_matches == pws.MAX_CANDIDATES_PER_TIER + 5
    assert stats.analyzed == pws.MAX_CANDIDATES_PER_TIER  # the pre-filter cut, not silently swallowed


# --- _dedupe_by_ticker / analyzed-count: Tercera auditoría, Bloque A-5/A-7 --


def _watchlist_item(ticker: str, setup: str | None, percentile_score: float | None = 80.0):
    from app.services import watchlist_service as wl

    return wl.WatchlistItem(
        ticker=ticker, sector="Tecnología", industry=None, cap_tier="mega", horizon=wl.SHORT_TERM,
        reasons=[f"reason for {setup}"], snapshot=_snapshot(ticker), setup=setup,
        percentile_score=percentile_score,
    )


def test_dedupe_by_ticker_keeps_the_first_occurrence_and_collects_the_rest_as_secondary_labels():
    # `items` arrives already sorted best-first by watchlist_service._sort_key
    # - the first AAPL entry (breakout_volume) is its highest-scoring match.
    aapl_primary = _watchlist_item("AAPL", "breakout_volume")
    aapl_secondary = _watchlist_item("AAPL", "trend_continuation")
    msft = _watchlist_item("MSFT", "pullback_to_support")
    deduped = pws._dedupe_by_ticker([aapl_primary, aapl_secondary, msft])
    assert [item.ticker for item, _ in deduped] == ["AAPL", "MSFT"]
    aapl_item, aapl_others = deduped[0]
    assert aapl_item is aapl_primary
    assert aapl_others == ["trend_continuation"]
    _, msft_others = deduped[1]
    assert msft_others == []


def test_dedupe_by_ticker_never_duplicates_the_same_setup_twice():
    # Same setup twice - shouldn't happen in practice, but must not double up.
    a1 = _watchlist_item("AAPL", "breakout_volume")
    a2 = _watchlist_item("AAPL", "breakout_volume")
    deduped = pws._dedupe_by_ticker([a1, a2])
    assert deduped[0][1] == []


def test_dedupe_by_ticker_empty_input_returns_empty():
    assert pws._dedupe_by_ticker([]) == []


def test_build_premium_watchlist_analyzes_a_multi_setup_ticker_only_once(monkeypatch):
    # The exact production bug: a ticker matching 3 setups (deliberate,
    # watchlist_service.py's own docstring) used to consume 3 of the
    # candidate slots and run the entire expensive pipeline 3 times over
    # identical OHLCV. After dedup it must be analyzed exactly once, and the
    # other setups it matched must survive as secondary labels, not vanish.
    multi = [
        _watchlist_item("AAPL", "breakout_volume", percentile_score=95.0),
        _watchlist_item("AAPL", "trend_continuation", percentile_score=95.0),
        _watchlist_item("AAPL", "oversold_bounce", percentile_score=95.0),
        _watchlist_item("MSFT", "pullback_to_support", percentile_score=70.0),
    ]
    monkeypatch.setattr(pws.wl, "build_watchlist", lambda snapshots, horizon=None: multi)

    calls: list[str] = []

    def counting_compute(close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None,
                          ticker=None):
        calls.append(ticker)
        return _signals(score=6)

    monkeypatch.setattr(pws, "compute_core_signals", counting_compute)
    market_data = _StubMarketData({"AAPL", "MSFT", pws.benchmark_for_region("us")})

    results, discard_stats = pws.build_premium_watchlist([], market_data, tiers=[pws.DAILY])

    assert calls.count("AAPL") == 1  # analyzed once, not 3 times
    assert calls.count("MSFT") == 1
    aapl_result = next(r for r in results if r.ticker == "AAPL")
    assert aapl_result.setup == "breakout_volume"
    assert set(aapl_result.also_matched_setups) == {"trend_continuation", "oversold_bounce"}
    stats = discard_stats[pws.DAILY]
    assert stats.prefilter_matches == 2  # 2 unique tickers, not 4 (ticker, setup) pairs
    assert stats.analyzed == 2


def test_build_premium_watchlist_analyzed_count_excludes_candidates_that_actually_failed(monkeypatch):
    # Tercera auditoría, Bloque A-7: `analyzed` must reflect candidates that
    # actually got a usable signal, not every pre-filter candidate fed into
    # the loop - a compute failure or missing OHLCV must not be silently
    # counted as "analyzed".
    good = _snapshot("GOOD", rs_rating=50)
    bad = _snapshot("BAD", rs_rating=99)
    missing = _snapshot("MISSING", rs_rating=60)
    market_data = _StubMarketData({"GOOD", "BAD", pws.benchmark_for_region("us")})  # MISSING absent from OHLCV

    def flaky_compute(close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None,
                       ticker=None):
        # Triggered by ticker, not rs_rating (Tercera auditoría, Bloque F-3:
        # the daily tier no longer passes rs_rating through unconverted).
        if ticker == "BAD":
            raise ValueError("simulated failure")
        return _signals(score=10)

    monkeypatch.setattr(pws, "compute_core_signals", flaky_compute)
    _, discard_stats = pws.build_premium_watchlist([good, bad, missing], market_data, tiers=[pws.DAILY])
    stats = discard_stats[pws.DAILY]
    assert stats.prefilter_matches == 3
    assert stats.analyzed == 1  # only GOOD actually produced a signal


# --- PremiumWatchlistService: per-tier cache TTL -----------------------------


class _StubScreener:
    def __init__(self, snapshot: list[TickerSnapshot]) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def get_universe_snapshot(self, region="us"):
        self.calls += 1
        return self.snapshot

    def get_sector_performance(self, region="us"):
        return []


def _fake_build_factory(calls):
    def fake_build(universe_snapshot, market_data, region="us", tiers=None, sector_rs_rank=None):
        tiers = tiers or list(pws.TIERS)
        calls.append(sorted(tiers))
        stats = {t: pws.TierDiscardStats(tier=t, prefilter_matches=0, analyzed=0, approved=0) for t in tiers}
        return [], stats

    return fake_build


def test_service_reuses_cache_within_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(pws, "build_premium_watchlist", _fake_build_factory(calls))
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    service.get_premium_watchlist()
    service.get_premium_watchlist()

    assert len(calls) == 1  # second call served entirely from cache
    assert sorted(calls[0]) == sorted(pws.TIERS)


def test_service_force_refresh_always_recomputes(monkeypatch):
    calls = []
    monkeypatch.setattr(pws, "build_premium_watchlist", _fake_build_factory(calls))
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    service.get_premium_watchlist()
    service.get_premium_watchlist(force_refresh=True)

    assert len(calls) == 2


def test_service_only_recomputes_stale_tiers(monkeypatch):
    calls = []
    monkeypatch.setattr(pws, "build_premium_watchlist", _fake_build_factory(calls))
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    now = datetime.now(UTC)
    # Prime the cache as if daily/weekly/monthly were all just computed...
    service.get_premium_watchlist()
    # ...except pretend the daily tier's cache is already a day old (expired).
    service._cache[("us", pws.DAILY)] = (now - timedelta(days=2), [])

    service.get_premium_watchlist()

    assert calls[-1] == [pws.DAILY]  # only the stale tier gets recomputed


def test_service_get_discard_stats_shares_the_same_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(pws, "build_premium_watchlist", _fake_build_factory(calls))
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    stats = service.get_discard_stats()
    assert {s.tier for s in stats} == set(pws.TIERS)
    assert len(calls) == 1

    # A second call for the same (already-fresh) tiers must not recompute.
    service.get_discard_stats()
    assert len(calls) == 1


def test_service_get_premium_watchlist_with_stats_recomputes_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(pws, "build_premium_watchlist", _fake_build_factory(calls))
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    items, stats = service.get_premium_watchlist_with_stats(force_refresh=True)
    assert items == []
    assert {s.tier for s in stats} == set(pws.TIERS)
    assert len(calls) == 1  # one recompute, not one per accessor


# --- Tercera auditoría, Bloque F-2: weekly tier Monte Carlo horizon --------


def test_weekly_tier_uses_the_1m_monte_carlo_horizon_not_3m(monkeypatch):
    # The weekly tier's own setups (fast-pair cross/imminent cross/Stage 2
    # leadership) are weeks-not-months signals now - its stop/target
    # simulation horizon must match that, not the old "3m" (63 sessions) it
    # ran at when the tier's rules were still months-scale.
    snapshots = [_snapshot("AAPL")]
    market_data = _StubMarketData({"AAPL", pws.benchmark_for_region("us")})
    horizons_seen = []

    def capturing_compute(close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None,
                           ticker=None):
        horizons_seen.append(horizon)
        return _signals(score=6)

    monkeypatch.setattr(pws, "compute_core_signals", capturing_compute)
    pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.WEEKLY])
    assert horizons_seen == ["1m"]


def test_daily_tier_uses_mansfield_rs_4w_percentile_not_rs_rating(monkeypatch):
    # Tercera auditoría, Bloque F-3: a candidate with an extreme rs_rating
    # (3-12 month momentum) but a middling mansfield_rs_4w (4-week relative
    # strength) must have the *latter* reach compute_core_signals for the
    # daily tier - the whole point of replacing it.
    high_rs_weak_mansfield = _snapshot("HIGHRS", rs_rating=99)
    # Give it a below-average mansfield_rs_4w relative to a second ticker.
    from dataclasses import replace

    weak_mansfield = replace(high_rs_weak_mansfield, mansfield_rs_4w=-5.0)
    strong_mansfield = replace(_snapshot("OTHER", rs_rating=10), mansfield_rs_4w=5.0)
    market_data = _StubMarketData({"HIGHRS", "OTHER", pws.benchmark_for_region("us")})

    rs_rating_seen = {}

    def capturing_compute(close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None,
                           ticker=None):
        rs_rating_seen[ticker] = rs_rating
        return _signals(score=6)

    monkeypatch.setattr(pws, "compute_core_signals", capturing_compute)
    pws.build_premium_watchlist([weak_mansfield, strong_mansfield], market_data, tiers=[pws.DAILY])

    # HIGHRS has rs_rating=99 but the *weaker* mansfield_rs_4w of the two -
    # if rs_rating were still being used, it would score higher, not lower.
    assert rs_rating_seen["HIGHRS"] < rs_rating_seen["OTHER"]


def test_weekly_tier_still_uses_rs_rating_unchanged(monkeypatch):
    from dataclasses import replace

    # rs_rating=42 alone wouldn't qualify for any weekly setup (STAGE2_LEADER
    # needs >=80) - ma_cross_short="golden" is the qualifying condition here;
    # rs_rating is only along for the ride, to confirm it passes through.
    snapshots = [replace(_snapshot("AAPL", rs_rating=42), ma_cross_short="golden")]
    market_data = _StubMarketData({"AAPL", pws.benchmark_for_region("us")})
    rs_rating_seen = []

    def capturing_compute(close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None,
                           ticker=None):
        rs_rating_seen.append(rs_rating)
        return _signals(score=6)

    monkeypatch.setattr(pws, "compute_core_signals", capturing_compute)
    pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.WEEKLY])
    assert rs_rating_seen == [42]


# --- _select_diversified (Tercera auditoría, Bloque F-8) --------------------


def _premium_item(ticker, sector, score=6.0, setup=None):
    return pws.PremiumWatchlistItem(
        ticker=ticker, sector=sector, industry=None, cap_tier="mega", currency="USD", region="us",
        tier=pws.DAILY, reasons=["r"], signals=_signals(score=int(score)), premium_score=score,
        raw_score=int(score), setup=setup,
    )


def test_select_diversified_caps_at_max_per_sector():
    scored = [
        (10.0, _premium_item("A", "Tech")),
        (9.0, _premium_item("B", "Tech")),
        (8.0, _premium_item("C", "Tech")),
        (7.0, _premium_item("D", "Tech")),  # 4th Tech name - over the cap, skipped
        (6.0, _premium_item("E", "Utilities")),
    ]
    approved = pws._select_diversified(scored, close_by_ticker={}, max_approved=10, max_per_sector=3)
    assert [i.ticker for i in approved] == ["A", "B", "C", "E"]


def test_select_diversified_skips_a_highly_correlated_candidate():
    dates = pd.bdate_range("2024-01-01", periods=90)
    base = pd.Series(100 + np.cumsum(np.random.RandomState(0).normal(0, 1, 90)), index=dates)
    twin = base * 1.001  # a scalar multiple - identical % returns, correlation exactly 1.0
    independent = pd.Series(100 + np.cumsum(np.random.RandomState(1).normal(0, 1, 90)), index=dates)
    close_by_ticker = {"A": base, "B": twin, "C": independent}
    scored = [
        (10.0, _premium_item("A", "Tech")),
        (9.0, _premium_item("B", "Tech")),  # near-identical to A - skipped despite a real score
        (8.0, _premium_item("C", "Utilities")),
    ]
    approved = pws._select_diversified(scored, close_by_ticker, max_approved=10, max_per_sector=3)
    assert [i.ticker for i in approved] == ["A", "C"]


def test_select_diversified_keeps_uncorrelated_names_from_the_same_sector():
    dates = pd.bdate_range("2024-01-01", periods=90)
    a = pd.Series(100 + np.cumsum(np.random.RandomState(0).normal(0, 1, 90)), index=dates)
    b = pd.Series(100 + np.cumsum(np.random.RandomState(2).normal(0, 1, 90)), index=dates)
    close_by_ticker = {"A": a, "B": b}
    scored = [(10.0, _premium_item("A", "Tech")), (9.0, _premium_item("B", "Tech"))]
    approved = pws._select_diversified(scored, close_by_ticker, max_approved=10, max_per_sector=3)
    assert [i.ticker for i in approved] == ["A", "B"]


def test_select_diversified_stops_at_max_approved():
    scored = [(10.0 - i, _premium_item(f"T{i}", "Tech")) for i in range(5)]
    approved = pws._select_diversified(scored, close_by_ticker={}, max_approved=2, max_per_sector=10)
    assert [i.ticker for i in approved] == ["T0", "T1"]


def test_select_diversified_missing_close_data_still_counts_against_sector_cap():
    scored = [(10.0, _premium_item("A", "Tech")), (9.0, _premium_item("B", "Tech"))]
    approved = pws._select_diversified(scored, close_by_ticker={}, max_approved=10, max_per_sector=1)
    assert [i.ticker for i in approved] == ["A"]


# --- Earnings exclusion (Tercera auditoría, Bloque F-9) ---------------------


def test_daily_tier_excludes_a_candidate_with_earnings_inside_the_holding_horizon(monkeypatch):
    snapshots = [_snapshot("SOON"), _snapshot("SAFE")]
    market_data = _StubMarketData(
        {"SOON", "SAFE", pws.benchmark_for_region("us")},
        days_to_earnings_by_ticker={"SOON": 5, "SAFE": 60},  # SOON inside the 21-day horizon, SAFE well outside
    )
    monkeypatch.setattr(
        pws, "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=6),
    )
    results, _ = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])
    assert [r.ticker for r in results] == ["SAFE"]


def test_weekly_tier_keeps_a_candidate_with_earnings_soon_but_marks_it(monkeypatch):
    from dataclasses import replace

    snapshot = replace(_snapshot("SOON", rs_rating=80), ma_cross_short="golden")
    market_data = _StubMarketData(
        {"SOON", pws.benchmark_for_region("us")}, days_to_earnings_by_ticker={"SOON": 5}
    )
    monkeypatch.setattr(
        pws, "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=6),
    )
    results, _ = pws.build_premium_watchlist([snapshot], market_data, tiers=[pws.WEEKLY])
    assert len(results) == 1
    assert results[0].days_to_earnings == 5


def test_daily_tier_keeps_a_candidate_with_no_earnings_data_at_all(monkeypatch):
    snapshots = [_snapshot("UNKNOWN_DATE")]
    market_data = _StubMarketData({"UNKNOWN_DATE", pws.benchmark_for_region("us")})  # no override -> None
    monkeypatch.setattr(
        pws, "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=6),
    )
    results, _ = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])
    assert [r.ticker for r in results] == ["UNKNOWN_DATE"]
    assert results[0].days_to_earnings is None


def test_daily_tier_keeps_a_candidate_whose_earnings_already_passed(monkeypatch):
    snapshots = [_snapshot("JUST_REPORTED")]
    market_data = _StubMarketData(
        {"JUST_REPORTED", pws.benchmark_for_region("us")}, days_to_earnings_by_ticker={"JUST_REPORTED": -3}
    )
    monkeypatch.setattr(
        pws, "compute_core_signals",
        lambda close, high, low, volume, open_, benchmark_close, rs_rating, horizon, vix_close=None, ticker=None:
        _signals(score=6),
    )
    results, _ = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])
    assert [r.ticker for r in results] == ["JUST_REPORTED"]
