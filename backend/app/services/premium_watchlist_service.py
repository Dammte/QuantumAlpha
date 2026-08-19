"""Builds a small, curated "premium" watchlist.

`watchlist_service.py` already surfaces every ticker that matches a cheap
technical rule - useful, but with ~170 tickers in the universe that can still
be dozens of names to review by hand. This module is different on purpose:
instead of a rule match, a ticker only makes this list if it's run through the
*exact same* deep-dive pipeline "Analizar activo" uses (recommendation, GARCH,
Markov, Monte Carlo, walk-forward backtest, Kelly sizing - see
`compute_core_signals()` in `ticker_analysis_service.py`) and that full
analysis actually endorses it. This is the same objectivity principle the
market-regime banner and the sector-rotation read already apply at the
market-wide level, now applied per-ticker: a name is never "premium" merely
because a cheap rule matched today.

Three priority tiers, one per review cadence a personal investor actually
follows - daily/weekly/monthly - each reusing the existing short/medium/long
horizon buckets from `watchlist_service.py` as its cheap pre-filter, and each
with a cache TTL matching its own name: there is no point re-running the full
quant pipeline for the "monthly" tier on every request when a month's worth of
history barely moves it.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import watchlist_service as wl
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import DEFAULT_REGION, VIX_TICKER, benchmark_for_region, currency_of
from app.services.ticker_analysis_service import CoreTickerSignals, compute_core_signals

logger = logging.getLogger(__name__)

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
TIERS = (DAILY, WEEKLY, MONTHLY)

# Refresh cadence per tier, matching the tier's own name - the "daily" list is
# worth recomputing once a day, the "monthly" one only once a month.
CACHE_TTL = {
    DAILY: timedelta(days=1),
    WEEKLY: timedelta(days=7),
    MONTHLY: timedelta(days=30),
}

# Which cheap pre-filter horizon (watchlist_service.py) and which Monte Carlo
# horizon preset (ticker_analysis_service.py) each premium tier maps onto.
_PREFILTER_HORIZON = {DAILY: wl.SHORT_TERM, WEEKLY: wl.MEDIUM_TERM, MONTHLY: wl.LONG_TERM}
_MONTE_CARLO_HORIZON = {DAILY: "1m", WEEKLY: "3m", MONTHLY: "6m"}

MAX_CANDIDATES_PER_TIER = 15  # how many cheap pre-filter matches get the expensive full analysis
MAX_APPROVED_PER_TIER = 10  # size cap of the final "premium" list - a handful of excellent names, not a scan
HISTORY_YEARS = 10
BACKTEST_COMPARISON = "comprar vs evitar"
BACKTEST_EDGE_BONUS = 3.0
KELLY_SETUP_BONUS = 1.0
MONTE_CARLO_EDGE_WEIGHT = 2.0
# Same "RS Rating >= 70" bar Minervini's own Trend Template already uses for
# an individual stock (see technical_analysis.minervini_checklist) - applied
# here to the *sector's* cross-sectional RS rank instead
# (MarketScreenerService.get_sector_performance), so a candidate leading in a
# sector that's itself leading gets a modest nudge over an equally-scored one
# in a lagging sector. CANSLIM/IBD's own "buy leaders in leading industries"
# principle, made concrete: two names can clear every bar on their own merits
# and still not be equally good bets if the wind is at one's back and not the
# other's. Deliberately smaller than BACKTEST_EDGE_BONUS - this is a ranking
# nudge among already-qualified "comprar" verdicts, not a validated,
# ablation-tested factor the way the core recommendation score's factors are
# (see recommendation_engine.py's own docstring on that distinction) - same
# character as the Kelly/Monte-Carlo bonuses already here, not the same rigor
# bar as the score itself.
STRONG_SECTOR_RS_THRESHOLD = 70
STRONG_SECTOR_BONUS = 1.0
# entry_timing.py is deliberately informational for the core recommendation -
# "extended" doesn't mean "avoid", it means the easy, low-risk part of the
# move likely already happened, which is a genuinely different question from
# "is this a good stock". But this list exists specifically to be a short,
# curated set worth acting on *today* (see the daily/weekly/monthly cadence
# above) - among two names that clear the exact same bar on every other
# count, the one that isn't already extended is the more actionable pick for
# a list with that stated purpose, so it should rank above the other rather
# than tie with it. A modest ranking penalty, not exclusion: an extended name
# can still be the best (or only) candidate available and will still show up,
# just no longer tied with a fresher setup that did just as well elsewhere.
EXTENDED_ENTRY_PENALTY = 1.0


@dataclass(frozen=True, slots=True)
class PremiumWatchlistItem:
    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    currency: str
    region: str
    tier: str
    reasons: list[str]  # why the cheap pre-filter even considered this ticker
    signals: CoreTickerSignals  # the full deep-dive result that got it approved
    premium_score: float
    # The raw, un-bonused `signals.recommendation.score` - kept alongside
    # `premium_score` (which folds in up to ~+7/-1 of backtest/Monte
    # Carlo/Kelly/sector/entry-timing bonuses, see `_approval_score`) so a
    # caller comparing this candidate against something that was never run
    # through those bonuses - e.g. a held position's plain checklist score in
    # `opportunity_cost.py` - has a same-scale number to compare against
    # instead of reaching for the bonus-inflated one. See that module's
    # docstring for why this distinction matters.
    raw_score: int


def _approval_score(signals: CoreTickerSignals, sector_rs_rank: int | None = None) -> float | None:
    """Returns a ranking score for a candidate that earns a spot on the premium
    list, or None to reject it. A rule-based "comprar" score alone isn't enough:
    a ticker whose OWN walk-forward backtest shows "evitar" has historically beaten
    "comprar" is excluded outright, regardless of today's score - the same
    objectivity principle the market regime and sector rotation reads apply.

    `sector_rs_rank` is this candidate's *sector's* own cross-sectional RS
    rank (MarketScreenerService.get_sector_performance) - None when sector
    performance couldn't be computed, which contributes nothing rather than
    guessing. See STRONG_SECTOR_BONUS."""
    if signals.recommendation.verdict != "comprar":
        return None

    backtest_edge = None
    if signals.backtest is not None:
        backtest_edge = next(
            (t for t in signals.backtest.significance_tests if t.comparison == BACKTEST_COMPARISON), None
        )
    backtest_contradicts = (
        backtest_edge is not None and backtest_edge.significant_at_5pct and backtest_edge.mean_difference <= 0
    )
    if backtest_contradicts:
        return None

    score = float(signals.recommendation.score)
    backtest_confirms = (
        backtest_edge is not None and backtest_edge.significant_at_5pct and backtest_edge.mean_difference > 0
    )
    if backtest_confirms:
        score += BACKTEST_EDGE_BONUS
    if signals.monte_carlo is not None and signals.monte_carlo.probability_target_before_stop is not None:
        score += signals.monte_carlo.probability_target_before_stop * MONTE_CARLO_EDGE_WEIGHT
    if signals.position_sizing is not None:
        score += KELLY_SETUP_BONUS  # a genuinely sizeable, favorable-odds trade setup
    if sector_rs_rank is not None and sector_rs_rank >= STRONG_SECTOR_RS_THRESHOLD:
        score += STRONG_SECTOR_BONUS  # leading in a sector that's itself leading - CANSLIM/IBD's own principle
    if signals.entry_timing is not None and signals.entry_timing.status == "extended":
        score -= EXTENDED_ENTRY_PENALTY  # already ran - a fresher, equally-scored setup ranks above it here
    return score


def build_premium_watchlist(
    universe_snapshot: list[TickerSnapshot],
    market_data: MarketDataService,
    region: str = DEFAULT_REGION,
    tiers: list[str] | None = None,
    sector_rs_rank: dict[str, int] | None = None,
) -> list[PremiumWatchlistItem]:
    tiers = tiers if tiers else list(TIERS)
    sector_rs_rank = sector_rs_rank or {}

    candidates_by_tier: dict[str, list[wl.WatchlistItem]] = {}
    all_candidate_tickers: set[str] = set()
    for t in tiers:
        items = wl.build_watchlist(universe_snapshot, horizon=_PREFILTER_HORIZON[t])
        top = items[:MAX_CANDIDATES_PER_TIER]
        candidates_by_tier[t] = top
        all_candidate_tickers.update(item.ticker for item in top)

    if not all_candidate_tickers:
        return []

    benchmark_ticker = benchmark_for_region(region)
    end = date.today()
    start = end - timedelta(days=365 * HISTORY_YEARS)
    # VIX rides along in the same batched call - one shared market-regime
    # input for every candidate, not fetched per ticker.
    ohlcv = market_data.get_bulk_ohlcv([*all_candidate_tickers, benchmark_ticker, VIX_TICKER], start, end)
    benchmark_df = ohlcv.get(benchmark_ticker)
    benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
    vix_df = ohlcv.get(VIX_TICKER)
    vix_close = vix_df["close"] if vix_df is not None else None

    results: list[PremiumWatchlistItem] = []
    for t in tiers:
        scored: list[tuple[float, PremiumWatchlistItem]] = []
        for candidate in candidates_by_tier[t]:
            df = ohlcv.get(candidate.ticker)
            if df is None:
                continue
            # One candidate's GARCH optimizer failing to converge, a backtest
            # edge case, or any other numerical hiccup on 15+ tickers a request
            # must never take the other 14 down with it - isolated per-ticker so
            # the list still ships with whatever *did* compute cleanly, which is
            # the whole reason this list exists ("premium" endorses individually
            # analyzed names, one bad name shouldn't erase the rest of the work).
            try:
                signals = compute_core_signals(
                    df["close"],
                    df["high"],
                    df["low"],
                    df["volume"],
                    df["open"],
                    benchmark_close,
                    candidate.snapshot.rs_rating,
                    horizon=_MONTE_CARLO_HORIZON[t],
                    vix_close=vix_close,
                )
            except Exception:
                logger.exception(
                    "Premium watchlist: skipping %s (%s tier) after a compute failure", candidate.ticker, t
                )
                continue
            if signals is None:
                continue
            score = _approval_score(signals, sector_rs_rank.get(candidate.sector))
            if score is None:
                continue
            scored.append(
                (
                    score,
                    PremiumWatchlistItem(
                        ticker=candidate.ticker,
                        sector=candidate.sector,
                        industry=candidate.industry,
                        cap_tier=candidate.cap_tier,
                        currency=currency_of(candidate.ticker),
                        region=region,
                        tier=t,
                        reasons=candidate.reasons,
                        signals=signals,
                        premium_score=score,
                        raw_score=signals.recommendation.score,
                    ),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results.extend(item for _, item in scored[:MAX_APPROVED_PER_TIER])
    return results


class PremiumWatchlistService:
    """Singleton wrapper (see `deps.py`) holding a per-(region, tier) in-process
    cache so the daily list only actually re-runs the full pipeline once a day,
    the weekly one once a week, and the monthly one once a month - independently
    for each region."""

    def __init__(self, market_data: MarketDataService, screener: MarketScreenerService) -> None:
        self.market_data = market_data
        self.screener = screener
        self._cache: dict[tuple[str, str], tuple[datetime, list[PremiumWatchlistItem]]] = {}

    def get_premium_watchlist(
        self, region: str = DEFAULT_REGION, tier: str | None = None, force_refresh: bool = False
    ) -> list[PremiumWatchlistItem]:
        tiers = [tier] if tier else list(TIERS)
        now = datetime.now(UTC)
        stale = [
            t
            for t in tiers
            if force_refresh
            or (region, t) not in self._cache
            or now - self._cache[(region, t)][0] >= CACHE_TTL[t]
        ]

        if stale:
            universe_snapshot = self.screener.get_universe_snapshot(region)
            # Already cheaply cached on its own (MarketScreenerService.CACHE_TTL) -
            # this rarely triggers a real fetch of its own. See STRONG_SECTOR_BONUS.
            sector_performance = self.screener.get_sector_performance(region)
            sector_rs_rank = {s.sector: s.rs_rank for s in sector_performance if s.rs_rank is not None}
            recomputed = build_premium_watchlist(
                universe_snapshot, self.market_data, region=region, tiers=stale, sector_rs_rank=sector_rs_rank
            )
            recomputed_by_tier: dict[str, list[PremiumWatchlistItem]] = {t: [] for t in stale}
            for item in recomputed:
                recomputed_by_tier[item.tier].append(item)
            for t in stale:
                self._cache[(region, t)] = (now, recomputed_by_tier[t])

        result: list[PremiumWatchlistItem] = []
        for t in tiers:
            result.extend(self._cache.get((region, t), (now, []))[1])
        return result
