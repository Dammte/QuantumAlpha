from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import premium_watchlist_service as pws
from app.services import technical_analysis as ta


def _signals(verdict="comprar", score=6, sig_tests=None, mc_prob=None, has_sizing=False, has_backtest=True):
    backtest = SimpleNamespace(significance_tests=sig_tests or []) if has_backtest else None
    monte_carlo = SimpleNamespace(probability_target_before_stop=mc_prob) if mc_prob is not None else None
    position_sizing = SimpleNamespace() if has_sizing else None
    return SimpleNamespace(
        recommendation=SimpleNamespace(verdict=verdict, score=score),
        backtest=backtest,
        monte_carlo=monte_carlo,
        position_sizing=position_sizing,
    )


def _sig_test(comparison="comprar vs evitar", significant=True, mean_difference=0.02):
    return SimpleNamespace(comparison=comparison, significant_at_5pct=significant, mean_difference=mean_difference)


# --- _approval_score: the objectivity gate ----------------------------------


def test_rejects_non_comprar_verdict():
    assert pws._approval_score(_signals(verdict="esperar")) is None
    assert pws._approval_score(_signals(verdict="evitar")) is None


def test_rejects_when_this_tickers_own_backtest_significantly_contradicts():
    """The core objectivity guard: even a "comprar" verdict today is excluded if
    THIS ticker's own walk-forward history says "evitar" has beaten "comprar"."""
    signals = _signals(sig_tests=[_sig_test(significant=True, mean_difference=-0.03)])
    assert pws._approval_score(signals) is None


def test_tolerates_missing_backtest_data():
    """A ticker with too little history to run the significance test at all
    (backtest=None) is neither rejected nor bonused - just scored at face value."""
    signals = _signals(score=6, has_backtest=False)
    assert pws._approval_score(signals) == pytest.approx(6.0)


def test_bonus_when_backtest_significantly_confirms_the_edge():
    base = pws._approval_score(_signals(score=6, sig_tests=[]))
    confirming_test = _sig_test(significant=True, mean_difference=0.02)
    confirmed = pws._approval_score(_signals(score=6, sig_tests=[confirming_test]))
    assert confirmed - base == pytest.approx(pws.BACKTEST_EDGE_BONUS)


def test_no_bonus_when_backtest_is_inconclusive():
    inconclusive = pws._approval_score(
        _signals(score=6, sig_tests=[_sig_test(significant=False, mean_difference=0.02)])
    )
    assert inconclusive == pytest.approx(6.0)


def test_monte_carlo_edge_adds_proportional_bonus():
    base = pws._approval_score(_signals(score=6))
    boosted = pws._approval_score(_signals(score=6, mc_prob=0.8))
    assert boosted - base == pytest.approx(0.8 * pws.MONTE_CARLO_EDGE_WEIGHT)


def test_kelly_setup_adds_flat_bonus():
    base = pws._approval_score(_signals(score=6))
    boosted = pws._approval_score(_signals(score=6, has_sizing=True))
    assert boosted - base == pytest.approx(pws.KELLY_SETUP_BONUS)


# --- build_premium_watchlist: orchestration ----------------------------------


def _snapshot(ticker: str, rs_rating: int = 80) -> TickerSnapshot:
    return TickerSnapshot(
        ticker=ticker,
        sector="Tecnología",
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
        fake_frame = {"close": None, "high": None, "low": None, "volume": None}
        return {t: fake_frame for t in tickers if t in self.tickers_with_data}


def test_build_premium_watchlist_caps_candidates_and_approved_per_tier(monkeypatch):
    # 20 daily-tier-worthy snapshots (all pass the cheap short-term rule via strong ADX/volume)
    snapshots = [_snapshot(f"T{i}", rs_rating=99 - i) for i in range(20)]
    market_data = _StubMarketData({s.ticker for s in snapshots} | {pws.benchmark_for_region("us")})

    monkeypatch.setattr(
        pws,
        "compute_core_signals",
        lambda close, high, low, volume, benchmark_close, rs_rating, horizon, vix_close=None: _signals(
            score=rs_rating or 0
        ),
    )

    # Every candidate approved (score always positive, verdict always "comprar")
    results = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])

    assert len(results) <= pws.MAX_APPROVED_PER_TIER
    assert all(r.tier == pws.DAILY for r in results)
    # Highest-score-first
    assert [r.premium_score for r in results] == sorted((r.premium_score for r in results), reverse=True)


def test_build_premium_watchlist_skips_tickers_missing_from_ohlcv(monkeypatch):
    snapshots = [_snapshot("HASDATA"), _snapshot("NODATA")]
    market_data = _StubMarketData({"HASDATA", pws.benchmark_for_region("us")})  # NODATA deliberately absent

    monkeypatch.setattr(
        pws,
        "compute_core_signals",
        lambda close, high, low, volume, benchmark_close, rs_rating, horizon, vix_close=None: _signals(score=10),
    )

    results = pws.build_premium_watchlist(snapshots, market_data, tiers=[pws.DAILY])
    assert [r.ticker for r in results] == ["HASDATA"]


def test_build_premium_watchlist_empty_universe_returns_empty(monkeypatch):
    market_data = _StubMarketData(set())
    assert pws.build_premium_watchlist([], market_data) == []


def test_build_premium_watchlist_isolates_a_candidate_whose_compute_raises(monkeypatch):
    """The exact production bug this test locks in: one candidate's GARCH
    optimizer failing to converge, a backtest edge case, or any other
    numerical hiccup on up to 15 tickers a request must never take the whole
    tier down with it (previously an uncaught exception propagated straight
    to a 500 on the whole premium watchlist response)."""
    good = _snapshot("GOOD", rs_rating=50)
    bad = _snapshot("BAD", rs_rating=99)
    market_data = _StubMarketData({"GOOD", "BAD", pws.benchmark_for_region("us")})

    def flaky_compute(close, high, low, volume, benchmark_close, rs_rating, horizon, vix_close=None):
        if rs_rating == 99:
            raise ValueError("simulated GARCH/backtest numerical failure")
        return _signals(score=10)

    monkeypatch.setattr(pws, "compute_core_signals", flaky_compute)

    results = pws.build_premium_watchlist([good, bad], market_data, tiers=[pws.DAILY])
    assert [r.ticker for r in results] == ["GOOD"]


# --- PremiumWatchlistService: per-tier cache TTL -----------------------------


class _StubScreener:
    def __init__(self, snapshot: list[TickerSnapshot]) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def get_universe_snapshot(self, region="us"):
        self.calls += 1
        return self.snapshot


def test_service_reuses_cache_within_ttl(monkeypatch):
    calls = []

    def fake_build(universe_snapshot, market_data, region="us", tiers=None):
        calls.append(list(tiers or pws.TIERS))
        return []

    monkeypatch.setattr(pws, "build_premium_watchlist", fake_build)
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    service.get_premium_watchlist()
    service.get_premium_watchlist()

    assert len(calls) == 1  # second call served entirely from cache
    assert sorted(calls[0]) == sorted(pws.TIERS)


def test_service_force_refresh_always_recomputes(monkeypatch):
    calls = []
    monkeypatch.setattr(pws, "build_premium_watchlist", lambda *a, **k: calls.append(1) or [])
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    service.get_premium_watchlist()
    service.get_premium_watchlist(force_refresh=True)

    assert len(calls) == 2


def test_service_only_recomputes_stale_tiers(monkeypatch):
    calls = []

    def fake_build(universe_snapshot, market_data, region="us", tiers=None):
        calls.append(sorted(tiers))
        return []

    monkeypatch.setattr(pws, "build_premium_watchlist", fake_build)
    service = pws.PremiumWatchlistService(market_data=object(), screener=_StubScreener([]))

    now = datetime.now(UTC)
    # Prime the cache as if daily/weekly/monthly were all just computed...
    service.get_premium_watchlist()
    # ...except pretend the daily tier's cache is already a day old (expired).
    service._cache[("us", pws.DAILY)] = (now - timedelta(days=2), [])

    service.get_premium_watchlist()

    assert calls[-1] == [pws.DAILY]  # only the stale tier gets recomputed
