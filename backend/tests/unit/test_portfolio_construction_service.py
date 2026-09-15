import numpy as np
import pandas as pd
import pytest

from app.services import portfolio_construction_service as pcs
from app.services.trade_geometry import EntryType, TradeGeometry

# --- compute_correlation_matrix / find_correlated_pairs ----------------------


def _returns(seed: int, n: int = 200, scale: float = 0.01) -> pd.Series:
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=dates)


def test_correlation_matrix_diagonal_is_one():
    a, b = _returns(1), _returns(2)
    matrix = pcs.compute_correlation_matrix({"A": a, "B": b})
    assert matrix.loc["A", "A"] == pytest.approx(1.0)
    assert matrix.loc["B", "B"] == pytest.approx(1.0)


def test_correlation_matrix_identical_series_is_fully_correlated():
    a = _returns(1)
    matrix = pcs.compute_correlation_matrix({"A": a, "B": a})
    assert matrix.loc["A", "B"] == pytest.approx(1.0)


def test_correlation_matrix_empty_input():
    assert pcs.compute_correlation_matrix({}).empty


def test_find_correlated_pairs_flags_identical_series():
    a = _returns(1)
    matrix = pcs.compute_correlation_matrix({"AMD": a, "NVDA": a, "KO": _returns(99)})
    pairs = pcs.find_correlated_pairs(matrix, threshold=0.8)
    assert len(pairs) == 1
    assert {pairs[0].ticker_a, pairs[0].ticker_b} == {"AMD", "NVDA"}
    assert pairs[0].correlation == pytest.approx(1.0)


def test_find_correlated_pairs_none_below_threshold():
    a, b = _returns(1), _returns(2)
    matrix = pcs.compute_correlation_matrix({"A": a, "B": b})
    # Two independent random series essentially never hit 0.8+ correlation.
    assert pcs.find_correlated_pairs(matrix, threshold=0.8) == []


# --- compute_sector_concentration / flag_concentrated_sectors ----------------


def test_sector_concentration_groups_and_sums_weights():
    weights = {"AAPL": 0.2, "MSFT": 0.15, "KO": 0.1}
    sectors = {"AAPL": "Tecnología", "MSFT": "Tecnología", "KO": "Consumo"}
    result = pcs.compute_sector_concentration(weights, sectors)
    by_sector = {c.sector: c.weight_pct for c in result}
    assert by_sector["Tecnología"] == pytest.approx(0.35)
    assert by_sector["Consumo"] == pytest.approx(0.1)


def test_sector_concentration_unknown_sector_groups_as_desconocido():
    weights = {"XYZ": 0.1}
    sectors: dict[str, str | None] = {}
    result = pcs.compute_sector_concentration(weights, sectors)
    assert result[0].sector == "Desconocido"
    assert result[0].tickers == ["XYZ"]


def test_sector_concentration_sorted_descending_by_weight():
    weights = {"A": 0.05, "B": 0.4, "C": 0.1}
    sectors = {"A": "S1", "B": "S2", "C": "S3"}
    result = pcs.compute_sector_concentration(weights, sectors)
    assert [c.sector for c in result] == ["S2", "S3", "S1"]


def test_flag_concentrated_sectors_above_threshold_only():
    concentrations = [
        pcs.SectorConcentration("Tecnología", 0.35, ["AAPL", "MSFT"]),
        pcs.SectorConcentration("Consumo", 0.1, ["KO"]),
    ]
    flagged = pcs.flag_concentrated_sectors(concentrations, max_pct=0.30)
    assert len(flagged) == 1
    assert flagged[0].sector == "Tecnología"


# --- compute_risk_contributions -----------------------------------------------


def test_risk_contributions_sum_to_one():
    a = _returns(1, scale=0.01)
    b = _returns(2, scale=0.03)
    contributions = pcs.compute_risk_contributions({"A": 0.5, "B": 0.5}, {"A": a, "B": b})
    total = sum(c.risk_contribution_pct for c in contributions)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_risk_contributions_favor_the_higher_volatility_position_at_equal_weight():
    # Equal capital weight, uncorrelated, but B is 3x A's volatility - B's
    # share of *risk* must exceed its share of *capital*.
    a = _returns(1, scale=0.01)
    b = _returns(2, scale=0.03)
    contributions = pcs.compute_risk_contributions({"A": 0.5, "B": 0.5}, {"A": a, "B": b})
    by_ticker = {c.ticker: c.risk_contribution_pct for c in contributions}
    assert by_ticker["B"] > by_ticker["A"]
    assert by_ticker["B"] > 0.5  # more risk than its 50% capital weight


def test_risk_contributions_empty_with_too_little_history():
    a = pd.Series([0.01, -0.01], index=pd.bdate_range("2024-01-01", periods=2))
    assert pcs.compute_risk_contributions({"A": 1.0}, {"A": a}) == []


# --- compute_portfolio_volatility ---------------------------------------------


def test_portfolio_volatility_no_diversification_when_perfectly_correlated():
    # B is literally the same series as A (correlation exactly 1.0) - equal
    # weights sum to the same volatility either one alone would have, no
    # diversification benefit at all.
    a = _returns(1, scale=0.015)
    solo_vol = pcs.compute_portfolio_volatility({"A": 1.0}, {"A": a})
    combined_vol = pcs.compute_portfolio_volatility({"A": 0.5, "B": 0.5}, {"A": a, "B": a})
    assert combined_vol == pytest.approx(solo_vol, rel=1e-6)


def test_portfolio_volatility_is_zero_for_a_perfect_hedge():
    # B's returns are the exact negation of A's, so an equal-weighted
    # portfolio return is 0 every single day - a perfect hedge.
    a = _returns(1, scale=0.02)
    b = -a
    combined_vol = pcs.compute_portfolio_volatility({"A": 0.5, "B": 0.5}, {"A": a, "B": b})
    assert combined_vol == pytest.approx(0.0, abs=1e-9)


def test_portfolio_volatility_none_without_matching_tickers():
    assert pcs.compute_portfolio_volatility({"A": 1.0}, {}) is None


# --- suggest_volatility_reduction ---------------------------------------------


def test_suggest_volatility_reduction_empty_when_under_target():
    contributions = [pcs.PositionRiskContribution("A", 0.5, 0.6), pcs.PositionRiskContribution("B", 0.5, 0.4)]
    assert pcs.suggest_volatility_reduction(0.10, target=0.15, risk_contributions=contributions) == []


def test_suggest_volatility_reduction_ranks_by_risk_contribution_when_over_target():
    contributions = [
        pcs.PositionRiskContribution("A", 0.3, 0.2),
        pcs.PositionRiskContribution("B", 0.3, 0.6),
        pcs.PositionRiskContribution("C", 0.4, 0.2),
    ]
    result = pcs.suggest_volatility_reduction(0.20, target=0.15, risk_contributions=contributions, top_n=2)
    assert [c.ticker for c in result] == ["B", "A"] or [c.ticker for c in result] == ["B", "C"]
    assert result[0].ticker == "B"  # highest risk contribution always first
    assert len(result) == 2


def test_suggest_volatility_reduction_empty_without_a_resolvable_volatility():
    contributions = [pcs.PositionRiskContribution("A", 1.0, 1.0)]
    assert pcs.suggest_volatility_reduction(None, target=0.15, risk_contributions=contributions) == []


# --- compute_aggregate_risk -----------------------------------------------------


def test_aggregate_risk_sums_distance_to_stop_times_quantity():
    positions = [
        pcs.HeldPositionRisk("A", price=100.0, stop=95.0, quantity=10.0),  # risk = 50
        pcs.HeldPositionRisk("B", price=50.0, stop=48.0, quantity=20.0),  # risk = 40
    ]
    report = pcs.compute_aggregate_risk(positions, portfolio_capital=1000.0)
    assert report.total_risk_amount == pytest.approx(90.0)
    assert report.total_risk_pct_of_capital == pytest.approx(0.09)
    assert report.exceeds_limit  # 9% > MAX_AGGREGATE_RISK_PCT (6%)


def test_aggregate_risk_position_already_below_stop_contributes_zero_not_negative():
    positions = [pcs.HeldPositionRisk("A", price=90.0, stop=95.0, quantity=10.0)]
    report = pcs.compute_aggregate_risk(positions, portfolio_capital=1000.0)
    assert report.total_risk_amount == pytest.approx(0.0)


def test_aggregate_risk_within_limit_not_flagged():
    positions = [pcs.HeldPositionRisk("A", price=100.0, stop=99.0, quantity=1.0)]  # risk = 1
    report = pcs.compute_aggregate_risk(positions, portfolio_capital=1000.0)  # 0.1%
    assert not report.exceeds_limit


def test_aggregate_risk_none_pct_without_capital():
    positions = [pcs.HeldPositionRisk("A", price=100.0, stop=95.0, quantity=1.0)]
    report = pcs.compute_aggregate_risk(positions, portfolio_capital=0.0)
    assert report.total_risk_pct_of_capital is None
    assert not report.exceeds_limit


# --- final_position_size --------------------------------------------------------


def test_final_position_size_is_the_tightest_constraint():
    assert pcs.final_position_size(100.0, portfolio_risk_limit_size=80.0, sector_limit_size=120.0) == 80.0
    assert pcs.final_position_size(50.0, portfolio_risk_limit_size=80.0, sector_limit_size=120.0) == 50.0
    assert pcs.final_position_size(100.0, portfolio_risk_limit_size=80.0, sector_limit_size=30.0) == 30.0


# --- sector_limit_shares ------------------------------------------------------


def test_sector_limit_shares_returns_the_remaining_headroom():
    concentrations = [pcs.SectorConcentration(sector="Tecnología", weight_pct=0.20, tickers=["AAPL"])]
    # 30% cap - 20% already held = 10% headroom = $10,000 of $100,000 capital.
    shares = pcs.sector_limit_shares(concentrations, "Tecnología", capital_total=100_000.0, entry_price=50.0)
    assert shares == pytest.approx(200.0)


def test_sector_limit_shares_is_zero_once_the_sector_is_already_at_or_past_the_cap():
    concentrations = [pcs.SectorConcentration(sector="Tecnología", weight_pct=0.35, tickers=["AAPL"])]
    shares = pcs.sector_limit_shares(concentrations, "Tecnología", capital_total=100_000.0, entry_price=50.0)
    assert shares == 0.0


def test_sector_limit_shares_full_headroom_for_a_sector_not_held_at_all():
    shares = pcs.sector_limit_shares([], "Salud", capital_total=100_000.0, entry_price=50.0)
    assert shares == pytest.approx(0.30 * 100_000.0 / 50.0)


def test_sector_limit_shares_unmapped_candidate_uses_the_desconocido_bucket():
    concentrations = [pcs.SectorConcentration(sector="Desconocido", weight_pct=0.25, tickers=["XYZ"])]
    shares = pcs.sector_limit_shares(concentrations, None, capital_total=100_000.0, entry_price=50.0)
    assert shares == pytest.approx(0.05 * 100_000.0 / 50.0)


# --- apply_portfolio_limits ----------------------------------------------------


def _sized_geometry(**overrides) -> TradeGeometry:
    defaults = dict(
        entry_price=50.0, stop_price=45.0, stop_basis="bajo el soporte en 45.00",
        entry_type=EntryType.PULLBACK_SUPPORT, risk_pct=0.10, risk_atr=2.5, risk_ceiling_pct=0.07,
        target_price=60.0, target_basis="objetivo 2:1 sobre el riesgo", reward_pct=0.20,
        risk_reward_gross=2.0, risk_reward_net=1.9,
        shares_for_risk_budget=20.0, position_value=1_000.0, pct_of_portfolio=0.10,
        viable=True, rejection_reason=None,
    )
    defaults.update(overrides)
    return TradeGeometry(**defaults)


_NO_AGGREGATE_RISK = pcs.AggregateRiskReport(
    total_risk_amount=0.0, total_risk_pct_of_capital=0.0, exceeds_limit=False
)


def test_apply_portfolio_limits_is_a_noop_on_a_non_viable_geometry():
    geometry = _sized_geometry(viable=False, rejection_reason="algo", shares_for_risk_budget=None)
    result = pcs.apply_portfolio_limits(geometry, "Tecnología", [], _NO_AGGREGATE_RISK, 10_000.0)
    assert result == geometry


def test_apply_portfolio_limits_is_a_noop_on_a_geometry_size_position_never_sized():
    # compute_entry_geometry alone (no capital) - shares_for_risk_budget is None even though viable.
    geometry = _sized_geometry(shares_for_risk_budget=None, position_value=None, pct_of_portfolio=None)
    result = pcs.apply_portfolio_limits(geometry, "Tecnología", [], _NO_AGGREGATE_RISK, 10_000.0)
    assert result == geometry


def test_apply_portfolio_limits_narrows_to_the_tightest_of_risk_and_sector_caps():
    # size_position alone already sized this at 20 shares ($1,000, 10% of a $10,000 book).
    geometry = _sized_geometry(shares_for_risk_budget=20.0, position_value=1_000.0, pct_of_portfolio=0.10)
    capital_total = 10_000.0
    # Aggregate risk already at 5.5% of a 6% budget - only 0.5% of headroom left.
    aggregate_risk = pcs.AggregateRiskReport(
        total_risk_amount=550.0, total_risk_pct_of_capital=0.055, exceeds_limit=False
    )
    # 0.5% headroom * $10,000 / $5 risk-per-share (50-45) = 10 shares - tighter than the sector's room.
    sector_concentrations = [pcs.SectorConcentration(sector="Tecnología", weight_pct=0.0, tickers=[])]

    result = pcs.apply_portfolio_limits(
        geometry, "Tecnología", sector_concentrations, aggregate_risk, capital_total
    )

    assert result.viable is True
    assert result.shares_for_risk_budget == pytest.approx(10.0)
    assert result.position_value == pytest.approx(500.0)
    assert result.pct_of_portfolio == pytest.approx(0.05)
    # Stop/target/entry_type untouched by the narrowing.
    assert result.stop_price == 45.0
    assert result.entry_type == EntryType.PULLBACK_SUPPORT


def test_apply_portfolio_limits_rejects_when_the_narrowed_position_is_too_small():
    geometry = _sized_geometry(shares_for_risk_budget=20.0, position_value=1_000.0, pct_of_portfolio=0.10)
    # A sector already essentially maxed out leaves almost no room at all.
    sector_concentrations = [pcs.SectorConcentration(sector="Tecnología", weight_pct=0.2999, tickers=["OTHER"])]

    result = pcs.apply_portfolio_limits(
        geometry, "Tecnología", sector_concentrations, _NO_AGGREGATE_RISK, 10_000.0
    )

    assert result.viable is False
    assert "pequeña" in result.rejection_reason
