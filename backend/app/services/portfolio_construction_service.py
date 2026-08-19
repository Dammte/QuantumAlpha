"""Portfolio-level risk that position-by-position analysis structurally
cannot see (D12): two positions correlated at 0.9 are one bet with two
names, not two independent ones; a sector at 60% of capital is a
concentration risk no single ticker's exit_urgency will ever flag; the money
actually lost if every stop got hit at once is a real number nobody
currently computes; and a position that's 10% of capital but three times as
volatile as the rest is a much bigger share of what can go wrong than its
capital weight suggests. `exit_engine.py`/`trade_manager.py` decide what to
do with *one* position; this is the missing "does the portfolio as a whole
make sense" layer.

Pure functions, no I/O - same discipline as `technical_analysis.py`. Callers
supply already-fetched daily return series and already-known position
sizes/stops; this module never fetches anything itself, and never places a
trade - like `opportunity_cost.py`, everything here is a number or a
suggestion, not an instruction.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

CORRELATION_WINDOW = 60  # trading days
TRADING_DAYS_PER_YEAR = 252

# First-pass thresholds, not ablation-calibrated yet - same status
# BUY_THRESHOLD/AVOID_THRESHOLD started at in recommendation_engine.py before
# their own audit, and MAX_POSITION_RISK_PCT/CHANDELIER_MULTIPLIER_BY_REGIME
# in trade_manager.py.
HIGH_CORRELATION_THRESHOLD = 0.8
MAX_SECTOR_CONCENTRATION_PCT = 0.30
PORTFOLIO_VOLATILITY_TARGET = 0.15  # 15% annualized
MAX_AGGREGATE_RISK_PCT = 0.06  # 6% of capital at risk at once, across every stop (1%/position in trade_manager.py)


@dataclass(frozen=True, slots=True)
class CorrelationWarning:
    ticker_a: str
    ticker_b: str
    correlation: float


def compute_correlation_matrix(
    returns_by_ticker: dict[str, pd.Series], window: int = CORRELATION_WINDOW
) -> pd.DataFrame:
    """Pairwise correlation of the last `window` daily returns for every
    held ticker - aligned on shared trading days first (an inner join via
    `dropna`), so a European and a US holding's mismatched calendars don't
    silently corrupt the correlation with gap-filled rows."""
    if not returns_by_ticker:
        return pd.DataFrame()
    aligned = pd.DataFrame(returns_by_ticker).dropna(how="any")
    recent = aligned.iloc[-window:] if len(aligned) > window else aligned
    return recent.corr()


def find_correlated_pairs(
    correlation_matrix: pd.DataFrame, threshold: float = HIGH_CORRELATION_THRESHOLD
) -> list[CorrelationWarning]:
    """Every pair of held tickers correlated at or beyond `threshold` -
    "two semis at 0.9 correlation are one position with two names, not two"."""
    warnings = []
    tickers = list(correlation_matrix.columns)
    for i, ticker_a in enumerate(tickers):
        for ticker_b in tickers[i + 1 :]:
            corr = correlation_matrix.loc[ticker_a, ticker_b]
            if pd.notna(corr) and abs(corr) >= threshold:
                warnings.append(CorrelationWarning(ticker_a, ticker_b, float(corr)))
    return warnings


@dataclass(frozen=True, slots=True)
class SectorConcentration:
    sector: str
    weight_pct: float  # fraction of total portfolio capital in this sector
    tickers: list[str]


def compute_sector_concentration(
    weight_by_ticker: dict[str, float], sector_by_ticker: dict[str, str | None]
) -> list[SectorConcentration]:
    """`weight_by_ticker` = each position's market value as a fraction of
    total portfolio capital. `sector_by_ticker` is the caller's own lookup
    (typically `market_universe.sector_of` per ticker) - a ticker outside
    the curated universe maps to `None`/missing and is grouped under
    "Desconocido" rather than silently dropped, since an unmapped sector is
    itself worth seeing, not hiding."""
    totals: dict[str, float] = defaultdict(float)
    tickers_by_sector: dict[str, list[str]] = defaultdict(list)
    for ticker, weight in weight_by_ticker.items():
        sector = sector_by_ticker.get(ticker) or "Desconocido"
        totals[sector] += weight
        tickers_by_sector[sector].append(ticker)
    return sorted(
        (SectorConcentration(sector, weight, tickers_by_sector[sector]) for sector, weight in totals.items()),
        key=lambda s: s.weight_pct,
        reverse=True,
    )


def flag_concentrated_sectors(
    concentrations: list[SectorConcentration], max_pct: float = MAX_SECTOR_CONCENTRATION_PCT
) -> list[SectorConcentration]:
    return [c for c in concentrations if c.weight_pct > max_pct]


@dataclass(frozen=True, slots=True)
class PositionRiskContribution:
    ticker: str
    weight_pct: float  # capital weight
    risk_contribution_pct: float  # share of total portfolio *risk*, not just capital - see docstring


def compute_risk_contributions(
    weight_by_ticker: dict[str, float], returns_by_ticker: dict[str, pd.Series], window: int = CORRELATION_WINDOW
) -> list[PositionRiskContribution]:
    """A position's contribution to portfolio risk is not its capital
    weight - a position that's 10% of capital with triple the volatility of
    the rest is a much larger share of what can actually go wrong. Standard
    risk-contribution decomposition: RC_i = w_i * (Sigma . w)_i / sigma_p,
    normalized to sum to 1.0 across all positions (Euler's theorem on the
    portfolio variance, a standard result - not a bespoke formula)."""
    tickers = [t for t in weight_by_ticker if t in returns_by_ticker]
    if not tickers:
        return []
    aligned = pd.DataFrame({t: returns_by_ticker[t] for t in tickers}).dropna(how="any")
    recent = aligned.iloc[-window:] if len(aligned) > window else aligned
    if len(recent) < 5:
        return []

    cov = recent.cov().to_numpy() * TRADING_DAYS_PER_YEAR  # annualized covariance
    weights = np.array([weight_by_ticker[t] for t in tickers])
    portfolio_variance = float(weights @ cov @ weights)
    if portfolio_variance <= 0:
        return []
    portfolio_vol = float(np.sqrt(portfolio_variance))

    marginal_contribution = cov @ weights  # Sigma . w
    risk_contribution = weights * marginal_contribution / portfolio_vol  # sums to portfolio_vol
    total = float(risk_contribution.sum())
    normalized = risk_contribution / total if total > 0 else np.zeros_like(risk_contribution)

    return [
        PositionRiskContribution(
            ticker=t, weight_pct=weight_by_ticker[t], risk_contribution_pct=float(normalized[i])
        )
        for i, t in enumerate(tickers)
    ]


def compute_portfolio_volatility(
    weight_by_ticker: dict[str, float], returns_by_ticker: dict[str, pd.Series], window: int = CORRELATION_WINDOW
) -> float | None:
    """Annualized realized volatility of the portfolio *as a whole* - not
    the weighted average of individual position volatilities, which ignores
    correlation entirely (two 30%-vol positions that move in lockstep
    produce a much more volatile portfolio than two that offset each
    other)."""
    tickers = [t for t in weight_by_ticker if t in returns_by_ticker]
    if not tickers:
        return None
    aligned = pd.DataFrame({t: returns_by_ticker[t] for t in tickers}).dropna(how="any")
    recent = aligned.iloc[-window:] if len(aligned) > window else aligned
    if len(recent) < 5:
        return None
    cov = recent.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    weights = np.array([weight_by_ticker[t] for t in tickers])
    variance = float(weights @ cov @ weights)
    return float(np.sqrt(variance)) if variance > 0 else 0.0


def suggest_volatility_reduction(
    portfolio_vol: float | None,
    target: float,
    risk_contributions: list[PositionRiskContribution],
    top_n: int = 3,
) -> list[PositionRiskContribution]:
    """When realized portfolio volatility exceeds `target`, the positions
    contributing the *most* to that risk - not necessarily the biggest
    capital weights - are the natural first candidates to trim. Empty
    (nothing to suggest) whenever the target isn't actually exceeded, or
    volatility couldn't be computed at all."""
    if portfolio_vol is None or portfolio_vol <= target:
        return []
    return sorted(risk_contributions, key=lambda r: r.risk_contribution_pct, reverse=True)[:top_n]


@dataclass(frozen=True, slots=True)
class HeldPositionRisk:
    ticker: str
    price: float
    stop: float
    quantity: float


@dataclass(frozen=True, slots=True)
class AggregateRiskReport:
    total_risk_amount: float  # sum of (price - stop) x quantity, in the portfolio's base currency
    total_risk_pct_of_capital: float | None
    exceeds_limit: bool


def compute_aggregate_risk(
    position_risks: list[HeldPositionRisk], portfolio_capital: float, max_risk_pct: float = MAX_AGGREGATE_RISK_PCT
) -> AggregateRiskReport:
    """The real money lost if every stop got hit at once - invisible when
    each position is only ever judged on its own (D12). A position already
    trading *below* its own stop contributes 0 here, not a negative number -
    that position's loss is exit_engine.py's problem (it should already be
    flagged exit_now), not something to net against everyone else's risk
    budget."""
    total = sum(max(0.0, p.price - p.stop) * p.quantity for p in position_risks)
    pct = total / portfolio_capital if portfolio_capital > 0 else None
    exceeds = pct is not None and pct > max_risk_pct
    return AggregateRiskReport(total_risk_amount=total, total_risk_pct_of_capital=pct, exceeds_limit=exceeds)


def final_position_size(kelly_size: float, portfolio_risk_limit_size: float, sector_limit_size: float) -> float:
    """The actual number of shares to size a new position at: the tightest
    of Kelly's own suggestion (`kelly_criterion.recommend_position_size`),
    the per-position portfolio-risk cap (`trade_manager.max_shares_for_position_risk`),
    and whatever a sector-concentration limit would still allow before
    breaching `MAX_SECTOR_CONCENTRATION_PCT`. Each constraint exists for a
    different reason and all three must hold at once - this is never wider
    than any individual one, only ever the same or narrower."""
    return min(kelly_size, portfolio_risk_limit_size, sector_limit_size)
