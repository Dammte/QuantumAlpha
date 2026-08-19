"""Cross-references what's actually held against the premium watchlist to
answer a question the risk column alone doesn't: not just "should I sell or
hold this", but "is my capital sitting in the *best* opportunity available to
me right now, or would it work harder somewhere else".

Holding a position that isn't actively broken (a `watch` or plain `hold`
signal, never mind an outright `exit_warning`) is still an opportunity cost
if a materially stronger, already-vetted setup exists elsewhere and that
capital could be redeployed into it - the same logic a discretionary trader
already applies by hand when reviewing a book, made explicit here. This is
deliberately a *suggestion to consider*, not an instruction: it never sells
anything, and a real decision should weigh taxes, conviction, and the reason
the position was opened in the first place - none of which this can see.
"""

from dataclasses import dataclass

from app.services.portfolio_risk_service import ADD_CANDIDATE, PositionRisk
from app.services.premium_watchlist_service import PremiumWatchlistItem

# How much higher a candidate's score must be than a held position's before this
# is worth surfacing at all - roughly the gap between the "esperar" zone and
# BUY_THRESHOLD in recommendation_engine.py, so this only fires on a genuinely
# material difference, not noise-level scoring gaps between two decent setups.
SWAP_SCORE_MARGIN = 4.0


@dataclass(frozen=True, slots=True)
class SwapSuggestion:
    held_ticker: str
    held_score: int
    held_signal: str  # exit_warning | watch | hold - never add_candidate, see find_swap_suggestions
    candidate_ticker: str
    candidate_score: float  # raw checklist score (PremiumWatchlistItem.raw_score) - same scale as held_score
    candidate_sector: str
    candidate_currency: str


def find_swap_suggestions(
    positions_risk: list[PositionRisk], premium_candidates: list[PremiumWatchlistItem]
) -> list[SwapSuggestion]:
    """One suggestion per held position that isn't already the best use of its
    own capital (`add_candidate` positions are skipped outright - there's
    nothing to improve on), naming the single strongest premium candidate not
    already held, when and only when it clears `SWAP_SCORE_MARGIN` over that
    position's own score. A ticker already held is never suggested as its own
    replacement, and each held position is only ever compared against the
    *one* best idea available - this is meant to prompt "is there something
    better", not to dump the whole watchlist on every weak position.

    `premium_score` (used below only to *rank* candidates against each other,
    picking the single best idea) folds in up to ~+7/-1 of
    backtest/Monte-Carlo/Kelly/sector/entry-timing bonuses on top of the plain
    checklist score - `position.score` never receives any of those. Comparing
    `premium_score` directly against `position.score` (a prior bug) let a
    candidate look up to ~7 points stronger than it actually was on the same
    checklist, so the margin below is always `raw_score` against `raw_score` -
    the same scale on both sides - never the bonus-inflated one."""
    held_tickers = {p.ticker for p in positions_risk}
    best_first = sorted(
        (c for c in premium_candidates if c.ticker not in held_tickers),
        key=lambda c: c.premium_score,
        reverse=True,
    )
    if not best_first:
        return []
    best = best_first[0]

    suggestions = []
    for position in positions_risk:
        if position.signal == ADD_CANDIDATE:
            continue
        if best.raw_score - position.score < SWAP_SCORE_MARGIN:
            continue
        suggestions.append(
            SwapSuggestion(
                held_ticker=position.ticker,
                held_score=position.score,
                held_signal=position.signal,
                candidate_ticker=best.ticker,
                candidate_score=float(best.raw_score),
                candidate_sector=best.sector,
                candidate_currency=best.currency,
            )
        )
    return suggestions
