"""First-order Markov chain over volatility-normalized return states, used to
forecast the probability distribution of an asset's near-term state and an
expected-return path - directly implementing the "predict what an asset will
do next based on prior steps" idea, grounded in this ticker's own history
rather than a universal assumption.

Two honesty checks ship alongside the forecast rather than presenting it
unconditionally as valid:

- A Wald-Wolfowitz runs test on the up/down sequence: if the sequence looks
  statistically indistinguishable from iid randomness, *any* Markov chain
  (of any order) is unlikely to have real predictive edge on this ticker, and
  the forecast should be read with that skepticism in mind.
- A likelihood-ratio (G^2) test of whether an order-2 chain (conditioning on
  the last 2 states) explains transitions significantly better than order-1 -
  answering "does more than one prior step actually matter here" with
  evidence instead of assumption, per Anderson & Goodman's classic test of
  Markov chain order.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

N_STATES = 5
STATE_LABELS = ["fuerte bajista", "bajista", "lateral", "alcista", "fuerte alcista"]
Z_CUTPOINTS = (-1.5, -0.5, 0.5, 1.5)
LAPLACE_SMOOTHING = 0.5
ROLLING_WINDOW = 20
MIN_OBSERVATIONS = 300


def discretize_states(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Bins each day's return by how many rolling standard deviations it sits
    from the rolling mean - stationary cut points even as raw volatility
    regimes change, unlike fixed-percentage thresholds."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    z = (returns - rolling_mean) / rolling_std.replace(0, np.nan)
    return pd.Series(np.digitize(z, Z_CUTPOINTS), index=returns.index).where(~z.isna())


def estimate_transition_matrix(
    states: np.ndarray, n_states: int = N_STATES, smoothing: float = LAPLACE_SMOOTHING
) -> np.ndarray:
    counts = np.full((n_states, n_states), smoothing)
    for i, j in zip(states[:-1], states[1:], strict=True):
        counts[int(i), int(j)] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def state_mean_returns(returns: np.ndarray, states: np.ndarray, n_states: int = N_STATES) -> np.ndarray:
    means = np.zeros(n_states)
    for s in range(n_states):
        mask = states == s
        means[s] = returns[mask].mean() if mask.any() else 0.0
    return means


def stationary_distribution(matrix: np.ndarray, iterations: int = 2000, tol: float = 1e-10) -> np.ndarray:
    n = matrix.shape[0]
    dist = np.full(n, 1.0 / n)
    for _ in range(iterations):
        new_dist = dist @ matrix
        if np.abs(new_dist - dist).sum() < tol:
            return new_dist
        dist = new_dist
    return dist


def forecast_distribution(matrix: np.ndarray, current_state: int, n_steps: int) -> np.ndarray:
    dist = np.zeros(matrix.shape[0])
    dist[current_state] = 1.0
    for _ in range(n_steps):
        dist = dist @ matrix
    return dist


def expected_cumulative_return(matrix: np.ndarray, means: np.ndarray, current_state: int, n_steps: int) -> float:
    """Compounds the state-conditional expected return step by step (each step
    uses the state distribution known at the *start* of that step), rather than
    naively summing single-day expectations - the correct way to project a
    multi-day expected return from a Markov chain."""
    dist = np.zeros(len(means))
    dist[current_state] = 1.0
    growth = 1.0
    for _ in range(n_steps):
        growth *= 1 + float(dist @ means)
        dist = dist @ matrix
    return growth - 1


def runs_test(is_up: np.ndarray) -> tuple[float, bool]:
    """Wald-Wolfowitz runs test on a binary (up/down) sequence. Returns
    (z_statistic, looks_random_at_5pct)."""
    n1 = int(is_up.sum())
    n2 = int(len(is_up) - n1)
    if n1 == 0 or n2 == 0:
        return 0.0, True
    runs = 1 + int(np.sum(is_up[1:] != is_up[:-1]))
    n = n1 + n2
    expected_runs = (2 * n1 * n2) / n + 1
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
    if variance <= 0:
        return 0.0, True
    z = (runs - expected_runs) / np.sqrt(variance)
    return float(z), bool(abs(z) < 1.96)


def order2_justification_test(states: np.ndarray, n_states: int = N_STATES) -> tuple[float, float, bool]:
    """Likelihood-ratio (G^2) test: does conditioning on the last *two* states
    explain the next state significantly better than just the last one?
    Returns (G2 statistic, p_value, order2_justified)."""
    triple_counts: dict[tuple[int, int, int], int] = {}
    for i, j, k in zip(states[:-2], states[1:-1], states[2:], strict=True):
        key = (int(i), int(j), int(k))
        triple_counts[key] = triple_counts.get(key, 0) + 1

    if len(triple_counts) < 2:
        return 0.0, 1.0, False

    context_ij_totals: dict[tuple[int, int], int] = {}
    from_j_to_k_totals: dict[tuple[int, int], int] = {}
    j_totals: dict[int, int] = {}
    for (i, j, k), n_ijk in triple_counts.items():
        context_ij_totals[(i, j)] = context_ij_totals.get((i, j), 0) + n_ijk
        from_j_to_k_totals[(j, k)] = from_j_to_k_totals.get((j, k), 0) + n_ijk
        j_totals[j] = j_totals.get(j, 0) + n_ijk

    g2 = 0.0
    for (i, j, k), n_ijk in triple_counts.items():
        expected = context_ij_totals[(i, j)] * from_j_to_k_totals[(j, k)] / j_totals[j]
        if expected > 0 and n_ijk > 0:
            g2 += 2 * n_ijk * np.log(n_ijk / expected)

    df = n_states * (n_states - 1) ** 2
    p_value = float(1 - stats.chi2.cdf(g2, df))
    return float(g2), p_value, bool(p_value < 0.05)


@dataclass(frozen=True, slots=True)
class MarkovChainResult:
    current_state: int
    current_state_label: str
    state_labels: list[str]
    transition_matrix: list[list[float]]
    state_mean_returns: list[float]
    stationary_distribution: list[float]
    forecast_5d_return: float
    forecast_21d_return: float
    forecast_21d_distribution: list[float]
    prob_bullish_21d: float
    runs_test_z: float
    sequence_looks_random: bool
    order2_justified: bool
    order2_p_value: float


def analyze_markov_chain(returns: pd.Series) -> MarkovChainResult | None:
    clean_returns = returns.dropna()
    if len(clean_returns) < MIN_OBSERVATIONS:
        return None

    states_series = discretize_states(clean_returns).dropna()
    if len(states_series) < MIN_OBSERVATIONS:
        return None

    aligned_returns = clean_returns.loc[states_series.index].to_numpy()
    states = states_series.to_numpy(dtype=int)

    matrix = estimate_transition_matrix(states, N_STATES)
    means = state_mean_returns(aligned_returns, states, N_STATES)
    current_state = int(states[-1])

    forecast_21d_dist = forecast_distribution(matrix, current_state, 21)
    is_up = (np.sign(clean_returns.to_numpy()) > 0).astype(int)
    runs_z, is_random = runs_test(is_up)
    g2, p_value, order2_justified = order2_justification_test(states, N_STATES)

    return MarkovChainResult(
        current_state=current_state,
        current_state_label=STATE_LABELS[current_state],
        state_labels=list(STATE_LABELS),
        transition_matrix=matrix.tolist(),
        state_mean_returns=means.tolist(),
        stationary_distribution=stationary_distribution(matrix).tolist(),
        forecast_5d_return=expected_cumulative_return(matrix, means, current_state, 5),
        forecast_21d_return=expected_cumulative_return(matrix, means, current_state, 21),
        forecast_21d_distribution=forecast_21d_dist.tolist(),
        prob_bullish_21d=float(forecast_21d_dist[3:].sum()),
        runs_test_z=runs_z,
        sequence_looks_random=is_random,
        order2_justified=order2_justified,
        order2_p_value=p_value,
    )
