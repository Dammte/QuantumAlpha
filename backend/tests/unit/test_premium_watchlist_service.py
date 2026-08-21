from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

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
    def __init__(self, tickers_with_data: set[str]) -> None:
        self.tickers_with_data = tickers_with_data
        self.requested: list[str] = []

    def get_bulk_ohlcv(self, tickers, start, end):
        self.requested = list(tickers)
        # Minimal stand-in for a DataFrame: subscriptable by column name only.
        fake_frame = {"close": None, "high": None, "low": None, "volume": None, "open": None}
        return {t: fake_frame for t in tickers if t in self.tickers_with_data}


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
        if rs_rating == 99:
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
