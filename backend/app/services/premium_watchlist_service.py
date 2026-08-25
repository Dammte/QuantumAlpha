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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import portfolio_construction_service as pcs
from app.services import watchlist_service as wl
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import DEFAULT_REGION, VIX_TICKER, benchmark_for_region, currency_of
from app.services.ticker_analysis_service import SIGN_CHECK_HORIZON_DAYS, CoreTickerSignals, compute_core_signals

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
# Tercera auditoría, Bloque F-2: WEEKLY used to run its Monte Carlo at "3m"
# (63 sessions) - the weekly tier's own setups (fast-pair cross/imminent
# cross/Stage 2 leadership, see watchlist_service.MEDIUM_TERM_SETUPS) are
# now short-pair-driven, weeks-not-months signals, so its stop/target
# simulation horizon should match that, not the old months-scale rules it
# replaced.
_MONTE_CARLO_HORIZON = {DAILY: "1m", WEEKLY: "1m", MONTHLY: "6m"}

MAX_CANDIDATES_PER_TIER = 15  # how many cheap pre-filter matches get the expensive full analysis
MAX_APPROVED_PER_TIER = 10  # size cap of the final "premium" list - a handful of excellent names, not a scan
HISTORY_YEARS = 10
# Segunda auditoría, Bloque 3: replaces MONTE_CARLO_EDGE_WEIGHT
# (probability_target_before_stop x 2.0) as the ranking-nudge factor - that
# read was near-circular (Monte Carlo's own stop/target simulation is derived
# from the same recommendation this score is already built from). This
# rewards a candidate's cross-sectional, setup-specific percentile
# (watchlist_service.setup_percentile_score) instead - only set for the daily
# tier's setup-based candidates (see build_watchlist); weekly/monthly
# candidates simply don't get this nudge, rather than a guessed substitute.
SETUP_PERCENTILE_BONUS_WEIGHT = 2.0
# Same "RS Rating >= 70" bar Minervini's own Trend Template already uses for
# an individual stock (see technical_analysis.minervini_checklist) - applied
# here to the *sector's* cross-sectional RS rank instead
# (MarketScreenerService.get_sector_performance), so a candidate leading in a
# sector that's itself leading gets a modest nudge over an equally-scored one
# in a lagging sector. CANSLIM/IBD's own "buy leaders in leading industries"
# principle, made concrete: two names can clear every bar on their own merits
# and still not be equally good bets if the wind is at one's back and not the
# other's. This is a ranking nudge among already-qualified "comprar" verdicts,
# not a validated, ablation-tested factor the way the core recommendation
# score's factors are (see recommendation_engine.py's own docstring on that
# distinction) - same character as SETUP_PERCENTILE_BONUS_WEIGHT/
# EXTENDED_ENTRY_PENALTY, not the same rigor bar as the score itself.
STRONG_SECTOR_RS_THRESHOLD = 70
STRONG_SECTOR_BONUS = 1.0

# Tercera auditoría, Bloque F-8: STRONG_SECTOR_BONUS actively rewards piling
# into whichever sector is leading right now (+1 to *every* candidate in it)
# with nothing on the other side capping how much of the final list that
# sector could take - for a 5-21 day hold, 8 semiconductor names correlated
# at 0.85 with each other are one position with eight tickers, not eight
# independent bets. MAX_PER_SECTOR is the countervailing force
# STRONG_SECTOR_BONUS needed and never had: a hard cap on the *approved*
# list, applied in the same greedy pass as the correlation cut below, is a
# clearer, more auditable "compensation" than a second soft score penalty
# stacked on top of a bonus that already isn't ablation-calibrated.
MAX_PER_SECTOR = 3
# Stricter than portfolio_construction_service.HIGH_CORRELATION_THRESHOLD
# (0.8) on purpose - that one flags an existing portfolio risk after the
# fact; this one decides whether a *candidate* even earns a slot in the
# first place, a higher bar for a name that hasn't been bought yet.
MAX_CANDIDATE_CORRELATION = 0.7
CORRELATION_RETURNS_WINDOW = 60  # trading days - matches portfolio_construction_service.CORRELATION_WINDOW
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
    # `premium_score` (which folds in the setup-percentile/sector/entry-timing
    # bonuses, see `_approval_score`) so a caller comparing this candidate
    # against something that was never run through those bonuses - e.g. a
    # held position's plain checklist score in `opportunity_cost.py` - has a
    # same-scale number to compare against instead of reaching for the
    # bonus-inflated one. See that module's docstring for why this
    # distinction matters.
    raw_score: int
    # The setup type this candidate matched (see watchlist_service.py) -
    # `None` for the monthly tier only, which isn't split into setup types
    # (Segunda auditoría, Bloque 3; the weekly tier got its own setups in
    # Tercera auditoría, Bloque F-2).
    setup: str | None = None
    # Every *other* setup this same ticker also matched, folded in here
    # instead of each one consuming its own separate candidate slot and
    # re-running the entire expensive pipeline again (Tercera auditoría,
    # Bloque A-5) - see `_dedupe_by_ticker`.
    also_matched_setups: list[str] = field(default_factory=list)
    # Tercera auditoría, Bloque F-9: `None` when the provider has no
    # earnings-calendar data for this ticker (never a fabricated "no
    # earnings risk"). The daily tier excludes a candidate outright when
    # this falls inside SIGN_CHECK_HORIZON_DAYS (see build_premium_watchlist)
    # rather than just flagging it - a breakout 3 days before earnings is a
    # different trade, not the same one with an asterisk. Other tiers keep
    # the candidate and only mark it, since their own effective horizon is
    # already long enough that one earnings print mid-trade is normal, not
    # a reason to exclude.
    days_to_earnings: int | None = None


def _approval_score(
    signals: CoreTickerSignals, sector_rs_rank: int | None = None, setup_percentile: float | None = None
) -> float | None:
    """Returns a ranking score for a candidate that earns a spot on the premium
    list, or None to reject it.

    Segunda auditoría, Bloque 3 - three things this used to do, removed:
    - The `backtest_contradicts` rejection (a ticker whose own *legacy*
      walk-forward backtest showed "evitar" historically beating "comprar"
      was excluded outright) ran on `walk_forward_backtest.py`
      (MIN_BUCKET_SIZE=8 with a Bonferroni correction over 3 comparisons is a
      noise detector at that sample size, and it systematically penalizes
      thin-history tickers). Reintroduce only once this can run on
      `backtest_engine`'s triple-barrier labeling with >=30 trades/bucket -
      not before.
    - `BACKTEST_EDGE_BONUS` (same legacy-backtest dependency) and
      `KELLY_SETUP_BONUS` (fired for almost every "comprar" verdict - added
      variance, not discrimination) are gone, not replaced.
    - `MONTE_CARLO_EDGE_WEIGHT` (`probability_target_before_stop x 2.0`) was
      near-circular (Monte Carlo's own stop/target simulation is derived
      from the same recommendation this score is already built from) -
      replaced by `setup_percentile` below.

    `sector_rs_rank` is this candidate's *sector's* own cross-sectional RS
    rank (MarketScreenerService.get_sector_performance) - `None` when sector
    performance couldn't be computed, which contributes nothing rather than
    guessing. `setup_percentile` is the candidate's cross-sectional,
    setup-specific percentile (watchlist_service.setup_percentile_score) -
    `None` for the weekly/monthly tiers, which still order by RS Rating (see
    build_watchlist) and get no substitute nudge here."""
    if signals.recommendation.verdict != "comprar":
        return None

    score = float(signals.recommendation.score)
    if setup_percentile is not None:
        score += (setup_percentile - 50.0) / 50.0 * SETUP_PERCENTILE_BONUS_WEIGHT
    if sector_rs_rank is not None and sector_rs_rank >= STRONG_SECTOR_RS_THRESHOLD:
        score += STRONG_SECTOR_BONUS  # leading in a sector that's itself leading - CANSLIM/IBD's own principle
    if signals.entry_timing is not None and signals.entry_timing.status == "extended":
        score -= EXTENDED_ENTRY_PENALTY  # already ran - a fresher, equally-scored setup ranks above it here
    return score


@dataclass(frozen=True, slots=True)
class TierDiscardStats:
    """How many candidates got dropped at each cut, per tier (Segunda
    auditoría, Bloque 3): `MAX_CANDIDATES_PER_TIER` used to truncate the
    cheap pre-filter's matches silently - "15 de 47 candidatos analizados"
    is now a real, surfaced number, not just a constant nobody could see the
    effect of."""

    tier: str
    prefilter_matches: int  # passed the cheap technical pre-filter, before any cut
    analyzed: int  # of those, how many got the expensive full analysis (<= MAX_CANDIDATES_PER_TIER)
    approved: int  # of the analyzed ones, how many made the final list (<= MAX_APPROVED_PER_TIER)


def _dedupe_by_ticker(items: list[wl.WatchlistItem]) -> list[tuple[wl.WatchlistItem, list[str]]]:
    """(best_item, other_setups_also_matched) per unique ticker - Tercera
    auditoría, Bloque A-5. A ticker matching several setups is deliberate
    (watchlist_service.py's own docstring: not mutually exclusive) and
    `items` arrives already sorted best-first (`wl._sort_key`), so the first
    occurrence of a ticker is its highest-scoring match; every subsequent
    occurrence is the *same* ticker under a different setup label, not a
    distinct candidate. Before this, each one consumed its own slot in
    `MAX_CANDIDATES_PER_TIER`/`MAX_APPROVED_PER_TIER` and re-ran the entire
    expensive per-ticker pipeline (GARCH+Markov+Monte Carlo+walk-forward+
    Kelly) again on identical data - measured 3x on a real universe (one
    ticker matching breakout_volume + trend_continuation + oversold_bounce).
    The other setups aren't dropped, just folded into a secondary label list
    instead of a separate row."""
    order: list[str] = []
    best: dict[str, wl.WatchlistItem] = {}
    other_setups: dict[str, list[str]] = {}
    for item in items:
        if item.ticker not in best:
            best[item.ticker] = item
            other_setups[item.ticker] = []
            order.append(item.ticker)
        elif (
            item.setup is not None
            and item.setup != best[item.ticker].setup
            and item.setup not in other_setups[item.ticker]
        ):
            other_setups[item.ticker].append(item.setup)
    return [(best[ticker], other_setups[ticker]) for ticker in order]


def _select_diversified(
    scored: list[tuple[float, PremiumWatchlistItem]],
    close_by_ticker: dict[str, pd.Series],
    max_approved: int = MAX_APPROVED_PER_TIER,
    max_per_sector: int = MAX_PER_SECTOR,
) -> list[PremiumWatchlistItem]:
    """Greedy, highest-score-first selection with two independent gates a
    plain `scored[:MAX_APPROVED_PER_TIER]` slice never applied (Tercera
    auditoría, Bloque F-8): a candidate is *skipped* (not swapped in for a
    retry - the next-best candidate simply gets its slot instead) if
    approving it would push its own sector past `max_per_sector` in this
    tier's approved list, or if its trailing `CORRELATION_RETURNS_WINDOW`-day
    returns correlate at or above `MAX_CANDIDATE_CORRELATION` with a name
    already approved this same pass. A candidate missing close data (a
    tickerless edge case that shouldn't happen in practice) skips the
    correlation check but still counts against its sector's cap."""
    approved: list[PremiumWatchlistItem] = []
    per_sector_count: dict[str, int] = defaultdict(int)
    returns_by_approved: dict[str, pd.Series] = {}

    for _, item in scored:
        if per_sector_count[item.sector] >= max_per_sector:
            continue

        candidate_close = close_by_ticker.get(item.ticker)
        candidate_returns = candidate_close.pct_change().dropna() if candidate_close is not None else None
        if candidate_returns is not None and returns_by_approved:
            corr_matrix = pcs.compute_correlation_matrix(
                {**returns_by_approved, item.ticker: candidate_returns}, window=CORRELATION_RETURNS_WINDOW
            )
            too_correlated = any(
                pd.notna(corr_matrix.loc[item.ticker, other]) and abs(corr_matrix.loc[item.ticker, other])
                >= MAX_CANDIDATE_CORRELATION
                for other in returns_by_approved
            )
            if too_correlated:
                continue

        approved.append(item)
        per_sector_count[item.sector] += 1
        if candidate_returns is not None:
            returns_by_approved[item.ticker] = candidate_returns
        if len(approved) >= max_approved:
            break
    return approved


def build_premium_watchlist(
    universe_snapshot: list[TickerSnapshot],
    market_data: MarketDataService,
    region: str = DEFAULT_REGION,
    tiers: list[str] | None = None,
    sector_rs_rank: dict[str, int] | None = None,
) -> tuple[list[PremiumWatchlistItem], dict[str, TierDiscardStats]]:
    tiers = tiers if tiers else list(TIERS)
    sector_rs_rank = sector_rs_rank or {}

    candidates_by_tier: dict[str, list[tuple[wl.WatchlistItem, list[str]]]] = {}
    prefilter_counts: dict[str, int] = {}
    all_candidate_tickers: set[str] = set()
    for t in tiers:
        items = wl.build_watchlist(universe_snapshot, horizon=_PREFILTER_HORIZON[t])
        deduped = _dedupe_by_ticker(items)
        # Tickers passing the pre-filter, not (ticker, setup) pairs - counting
        # pairs inflated this denominator ~1.5-2x (Bloque A-7) since a single
        # ticker matching 2-3 setups used to count as 2-3 "candidates".
        prefilter_counts[t] = len(deduped)
        top = deduped[:MAX_CANDIDATES_PER_TIER]
        candidates_by_tier[t] = top
        all_candidate_tickers.update(item.ticker for item, _ in top)
        if len(deduped) > MAX_CANDIDATES_PER_TIER:
            logger.info(
                "Premium watchlist: %s tier pre-filter found %d candidates, analyzing only the top %d",
                t, len(deduped), MAX_CANDIDATES_PER_TIER,
            )

    if not all_candidate_tickers:
        discard_stats = {
            t: TierDiscardStats(tier=t, prefilter_matches=prefilter_counts.get(t, 0), analyzed=0, approved=0)
            for t in tiers
        }
        return [], discard_stats

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

    # Tercera auditoría, Bloque F-3: the daily (5-21 day) tier used to pass
    # candidate.snapshot.rs_rating (a 3-12-month momentum percentile) into
    # compute_core_signals - recommendation_engine.py turns rs_rating>=80
    # into a flat +2, the same weight as the setup-specific percentile bonus
    # (SETUP_PERCENTILE_BONUS_WEIGHT, max ±2) that's supposed to be the
    # daily tier's actual ordering criterion. A 4-week relative-strength
    # percentile (mansfield_rs_4w, already computed per snapshot) is the
    # same-scale, same-cross-section substitute - a genuinely short-term
    # momentum read instead of a 3-12-month one scoring a 5-21 day trade.
    daily_rs_substitute = wl.percentile_rank_by_ticker(universe_snapshot, "mansfield_rs_4w")

    results: list[PremiumWatchlistItem] = []
    discard_stats: dict[str, TierDiscardStats] = {}
    for t in tiers:
        scored: list[tuple[float, PremiumWatchlistItem]] = []
        actually_analyzed = 0
        for candidate, also_matched_setups in candidates_by_tier[t]:
            df = ohlcv.get(candidate.ticker)
            if df is None:
                continue
            # One candidate's GARCH optimizer failing to converge, a backtest
            # edge case, or any other numerical hiccup on 15+ tickers a request
            # must never take the other 14 down with it - isolated per-ticker so
            # the list still ships with whatever *did* compute cleanly, which is
            # the whole reason this list exists ("premium" endorses individually
            # analyzed names, one bad name shouldn't erase the rest of the work).
            if t == DAILY:
                substitute = daily_rs_substitute.get(candidate.ticker)
                rs_rating_input = int(round(substitute)) if substitute is not None else None
            else:
                rs_rating_input = candidate.snapshot.rs_rating
            try:
                signals = compute_core_signals(
                    df["close"],
                    df["high"],
                    df["low"],
                    df["volume"],
                    df["open"],
                    benchmark_close,
                    rs_rating_input,
                    horizon=_MONTE_CARLO_HORIZON[t],
                    vix_close=vix_close,
                    ticker=candidate.ticker,
                )
            except Exception:
                logger.exception(
                    "Premium watchlist: skipping %s (%s tier) after a compute failure", candidate.ticker, t
                )
                continue
            if signals is None:
                continue
            # Tercera auditoría, Bloque A-7: counted here, not before the
            # OHLCV-missing/compute-failure `continue`s above - this used to
            # report every pre-filter candidate fed into the loop as
            # "analyzed", so "15 analizados" could mean 5 of them silently
            # failed and never actually got a real signal.
            actually_analyzed += 1

            # Tercera auditoría, Bloque F-9: a breakout 3 days before
            # earnings isn't the same trade as one with no event risk in
            # the holding window. Only fetched here (a bounded, <=15/tier
            # loop - never the whole universe screener, which would be a
            # per-ticker network call in a hot path).
            next_earnings = market_data.get_next_earnings_date(candidate.ticker)
            days_to_earnings = (next_earnings - date.today()).days if next_earnings else None
            if t == DAILY and days_to_earnings is not None and 0 <= days_to_earnings <= SIGN_CHECK_HORIZON_DAYS:
                logger.info(
                    "Premium watchlist: excluding %s (daily tier) - earnings in %d days, inside the "
                    "%d-day holding horizon", candidate.ticker, days_to_earnings, SIGN_CHECK_HORIZON_DAYS,
                )
                continue

            score = _approval_score(signals, sector_rs_rank.get(candidate.sector), candidate.percentile_score)
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
                        setup=candidate.setup,
                        also_matched_setups=also_matched_setups,
                        days_to_earnings=days_to_earnings,
                    ),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Tercera auditoría, Bloque F-8: a plain top-N slice let one leading
        # sector (rewarded by STRONG_SECTOR_BONUS above) fill the whole
        # list with names that are, for a 5-21 day hold, effectively the
        # same correlated bet repeated - see _select_diversified's own
        # docstring for the sector cap + correlation gate that replaces it.
        close_by_ticker = {
            candidate.ticker: ohlcv[candidate.ticker]["close"]
            for candidate, _ in candidates_by_tier[t]
            if candidate.ticker in ohlcv
        }
        approved = _select_diversified(scored, close_by_ticker)
        results.extend(approved)
        discard_stats[t] = TierDiscardStats(
            tier=t,
            prefilter_matches=prefilter_counts.get(t, 0),
            analyzed=actually_analyzed,
            approved=len(approved),
        )
    return results, discard_stats


class PremiumWatchlistService:
    """Singleton wrapper (see `deps.py`) holding a per-(region, tier) in-process
    cache so the daily list only actually re-runs the full pipeline once a day,
    the weekly one once a week, and the monthly one once a month - independently
    for each region."""

    def __init__(self, market_data: MarketDataService, screener: MarketScreenerService) -> None:
        self.market_data = market_data
        self.screener = screener
        self._cache: dict[tuple[str, str], tuple[datetime, list[PremiumWatchlistItem]]] = {}
        # Segunda auditoría, Bloque 3 - see TierDiscardStats. Same
        # per-(region, tier) cache lifetime as `_cache` above, just tracked
        # separately since it isn't part of the item list itself.
        self._discard_stats_cache: dict[tuple[str, str], TierDiscardStats] = {}

    def get_premium_watchlist(
        self, region: str = DEFAULT_REGION, tier: str | None = None, force_refresh: bool = False
    ) -> list[PremiumWatchlistItem]:
        self._ensure_fresh(region, tier, force_refresh)
        tiers = [tier] if tier else list(TIERS)
        result: list[PremiumWatchlistItem] = []
        for t in tiers:
            result.extend(self._cache.get((region, t), (datetime.now(UTC), []))[1])
        return result

    def get_discard_stats(
        self, region: str = DEFAULT_REGION, tier: str | None = None, force_refresh: bool = False
    ) -> list[TierDiscardStats]:
        """"15 de 47 candidatos analizados" per tier - see TierDiscardStats.
        Shares the exact same cache lifetime/computation as
        `get_premium_watchlist` (calling this alone still triggers the same
        full recompute on a stale/missing cache - there's no cheaper way to
        get these counts than actually running the pipeline). Prefer
        `get_premium_watchlist_with_stats` when a caller needs both - calling
        this *and* `get_premium_watchlist` separately with `force_refresh=True`
        would otherwise recompute twice."""
        self._ensure_fresh(region, tier, force_refresh)
        tiers = [tier] if tier else list(TIERS)
        return [self._discard_stats_cache[(region, t)] for t in tiers if (region, t) in self._discard_stats_cache]

    def get_premium_watchlist_with_stats(
        self, region: str = DEFAULT_REGION, tier: str | None = None, force_refresh: bool = False
    ) -> tuple[list[PremiumWatchlistItem], list[TierDiscardStats]]:
        """Both `get_premium_watchlist` and `get_discard_stats` off a single
        freshness check - what the API endpoint uses, so a `refresh=true`
        request never pays for the full recompute twice."""
        self._ensure_fresh(region, tier, force_refresh)
        tiers = [tier] if tier else list(TIERS)
        items: list[PremiumWatchlistItem] = []
        for t in tiers:
            items.extend(self._cache.get((region, t), (datetime.now(UTC), []))[1])
        stats = [self._discard_stats_cache[(region, t)] for t in tiers if (region, t) in self._discard_stats_cache]
        return items, stats

    def _ensure_fresh(self, region: str, tier: str | None, force_refresh: bool) -> None:
        tiers = [tier] if tier else list(TIERS)
        now = datetime.now(UTC)
        stale = [
            t
            for t in tiers
            if force_refresh
            or (region, t) not in self._cache
            or now - self._cache[(region, t)][0] >= CACHE_TTL[t]
        ]
        if not stale:
            return

        universe_snapshot = self.screener.get_universe_snapshot(region)
        # Already cheaply cached on its own (MarketScreenerService.CACHE_TTL) -
        # this rarely triggers a real fetch of its own. See STRONG_SECTOR_BONUS.
        sector_performance = self.screener.get_sector_performance(region)
        sector_rs_rank = {s.sector: s.rs_rank for s in sector_performance if s.rs_rank is not None}
        recomputed, discard_stats = build_premium_watchlist(
            universe_snapshot, self.market_data, region=region, tiers=stale, sector_rs_rank=sector_rs_rank
        )
        recomputed_by_tier: dict[str, list[PremiumWatchlistItem]] = {t: [] for t in stale}
        for item in recomputed:
            recomputed_by_tier[item.tier].append(item)
        for t in stale:
            self._cache[(region, t)] = (now, recomputed_by_tier[t])
            if t in discard_stats:
                self._discard_stats_cache[(region, t)] = discard_stats[t]
