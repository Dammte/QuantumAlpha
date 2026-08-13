from app.services.opportunity_cost import find_swap_suggestions
from app.services.portfolio_risk_service import ADD_CANDIDATE, EXIT_WARNING, HOLD, WATCH, PositionRisk
from app.services.premium_watchlist_service import PremiumWatchlistItem


def _position(ticker: str, signal: str, score: int) -> PositionRisk:
    return PositionRisk(
        ticker=ticker,
        currency="USD",
        price=100.0,
        trend="uptrend",
        stage=None,
        ma_cross=None,
        rs_rating=None,
        nearest_support=None,
        nearest_resistance=None,
        signal=signal,
        score=score,
        reasons=["reason"],
        signals=None,  # not touched by find_swap_suggestions
    )


def _candidate(ticker: str, score: float, sector: str = "Tecnología") -> PremiumWatchlistItem:
    return PremiumWatchlistItem(
        ticker=ticker,
        sector=sector,
        industry=None,
        cap_tier="large",
        currency="USD",
        region="us",
        tier="daily",
        reasons=["reason"],
        signals=None,  # not touched by find_swap_suggestions
        premium_score=score,
    )


def test_no_suggestion_when_no_candidates() -> None:
    assert find_swap_suggestions([_position("AAPL", WATCH, 2)], []) == []


def test_no_suggestion_for_add_candidate_positions() -> None:
    """Already the best use of its own capital - nothing to improve on."""
    positions = [_position("AAPL", ADD_CANDIDATE, 8)]
    candidates = [_candidate("NVDA", 14.0)]
    assert find_swap_suggestions(positions, candidates) == []


def test_no_suggestion_when_margin_not_cleared() -> None:
    # 8 - 6 = 2, below SWAP_SCORE_MARGIN (4.0)
    positions = [_position("AAPL", WATCH, 6)]
    candidates = [_candidate("NVDA", 8.0)]
    assert find_swap_suggestions(positions, candidates) == []


def test_suggestion_fires_when_margin_cleared() -> None:
    positions = [_position("AAPL", WATCH, 2)]
    candidates = [_candidate("NVDA", 14.0)]
    suggestions = find_swap_suggestions(positions, candidates)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s.held_ticker == "AAPL"
    assert s.held_score == 2
    assert s.held_signal == WATCH
    assert s.candidate_ticker == "NVDA"
    assert s.candidate_score == 14.0


def test_exit_warning_positions_are_eligible_too() -> None:
    """A position already flagged to sell is a natural pairing with "and here's
    where that capital could go instead", not redundant with the sell signal."""
    positions = [_position("XYZ", EXIT_WARNING, -4)]
    candidates = [_candidate("NVDA", 12.0)]
    suggestions = find_swap_suggestions(positions, candidates)
    assert len(suggestions) == 1
    assert suggestions[0].held_signal == EXIT_WARNING


def test_already_held_candidate_is_never_suggested() -> None:
    positions = [_position("AAPL", WATCH, 2), _position("NVDA", HOLD, 3)]
    candidates = [_candidate("NVDA", 14.0)]  # already held - can't be "the alternative"
    assert find_swap_suggestions(positions, candidates) == []


def test_best_candidate_reused_across_multiple_weak_positions() -> None:
    positions = [_position("AAPL", WATCH, 2), _position("MSFT", HOLD, 1)]
    candidates = [_candidate("NVDA", 14.0), _candidate("AMD", 10.0)]
    suggestions = find_swap_suggestions(positions, candidates)
    assert len(suggestions) == 2
    assert all(s.candidate_ticker == "NVDA" for s in suggestions)  # always the single best idea


def test_score_comparison_is_against_the_single_best_candidate_only() -> None:
    positions = [_position("AAPL", WATCH, 9)]
    # AMD (10.0) alone wouldn't clear the margin over a score of 9, but NVDA does
    candidates = [_candidate("AMD", 10.0), _candidate("NVDA", 14.0)]
    suggestions = find_swap_suggestions(positions, candidates)
    assert len(suggestions) == 1
    assert suggestions[0].candidate_ticker == "NVDA"
