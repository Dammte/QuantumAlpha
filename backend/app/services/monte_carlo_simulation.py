"""Monte Carlo price-path simulation for probabilistic risk analysis: percentile
price bands at several horizons, and - when a stop-loss/take-profit pair is
available from the recommendation engine - the probability of hitting one
barrier before the other.

Uses **filtered historical simulation** when a GARCH fit is available (resample
GARCH-standardized residuals via a block bootstrap, then re-inflate through a
simulated forward volatility path) rather than plain i.i.d. resampling of raw
returns - this preserves the fat tails and short-horizon volatility clustering
that iid resampling destroys, which matters specifically for stop-loss/drawdown
risk (see `volatility_model.py`'s docstring for the same reasoning). Falls back
to a plain block bootstrap of raw returns when no GARCH fit is available (e.g.
too little history), which is still far more defensible than iid resampling.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.volatility_model import GarchResult, conditional_variance_path

MIN_OBSERVATIONS = 250
DEFAULT_N_SIMULATIONS = 2000
DEFAULT_BLOCK_SIZE = 5
DEFAULT_CHECKPOINTS = (5, 21, 42, 63)


def _block_bootstrap_matrix(
    source: np.ndarray, n_sims: int, n_days: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Resamples contiguous blocks (not single iid points) to preserve
    short-horizon autocorrelation/vol clustering. Falls back to iid resampling
    only if the source history is shorter than a single block."""
    n_source = len(source)
    max_start = n_source - block_size
    if max_start < 1:
        return rng.choice(source, size=(n_sims, n_days), replace=True)

    n_blocks = int(np.ceil(n_days / block_size))
    starts = rng.integers(0, max_start + 1, size=(n_sims, n_blocks))
    offsets = np.arange(block_size)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_sims, n_blocks * block_size)
    return source[idx][:, :n_days]


def simulate_paths(
    returns: pd.Series,
    current_price: float,
    n_days: int,
    garch: GarchResult | None = None,
    n_sims: int = DEFAULT_N_SIMULATIONS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int | None = None,
) -> tuple[np.ndarray, str]:
    """Returns (prices of shape (n_sims, n_days), method label)."""
    rng = np.random.default_rng(seed)
    clean = returns.dropna()
    mean_return = float(clean.mean())

    if garch is not None:
        demeaned = (clean - mean_return).to_numpy()
        sigma2_path = conditional_variance_path(demeaned, garch.omega, garch.alpha, garch.beta)
        z = demeaned / np.sqrt(sigma2_path)
        z_sampled = _block_bootstrap_matrix(z, n_sims, n_days, block_size, rng)

        prices = np.empty((n_sims, n_days))
        price_t = np.full(n_sims, current_price)
        sigma2_t = np.full(n_sims, sigma2_path[-1])
        for day in range(n_days):
            eps = np.sqrt(sigma2_t) * z_sampled[:, day]
            price_t = price_t * (1 + mean_return + eps)
            prices[:, day] = price_t
            sigma2_t = garch.omega + garch.alpha * eps**2 + garch.beta * sigma2_t
        return prices, "garch_filtered"

    r_sampled = _block_bootstrap_matrix(clean.to_numpy(), n_sims, n_days, block_size, rng)
    prices = current_price * np.cumprod(1 + r_sampled, axis=1)
    return prices, "block_bootstrap"


def barrier_probabilities(
    paths: np.ndarray, stop_loss: float, take_profit: float
) -> tuple[float, float, float]:
    """Walks each simulated path day by day (not just the endpoint - a path can
    cross a barrier mid-horizon and end up elsewhere) and returns
    (P(stop hit first), P(target hit first), P(neither hit))."""
    n_sims, n_days = paths.shape
    stop_first = np.zeros(n_sims, dtype=bool)
    target_first = np.zeros(n_sims, dtype=bool)
    resolved = np.zeros(n_sims, dtype=bool)

    for day in range(n_days):
        price_day = paths[:, day]
        newly_stop = (price_day <= stop_loss) & ~resolved
        newly_target = (price_day >= take_profit) & ~resolved & ~newly_stop
        stop_first |= newly_stop
        target_first |= newly_target
        resolved |= newly_stop | newly_target

    return float(stop_first.mean()), float(target_first.mean()), float((~resolved).mean())


@dataclass(frozen=True, slots=True)
class PricePercentiles:
    day: int
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    n_simulations: int
    method: str
    percentiles: list[PricePercentiles]
    probability_of_loss: float
    probability_stop_before_target: float | None
    probability_target_before_stop: float | None
    probability_neither_hit: float | None


def simulate_and_analyze(
    returns: pd.Series,
    current_price: float,
    garch: GarchResult | None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    n_days: int = 63,
    n_sims: int = DEFAULT_N_SIMULATIONS,
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
    seed: int | None = None,
) -> MonteCarloResult | None:
    if returns.dropna().shape[0] < MIN_OBSERVATIONS:
        return None

    paths, method = simulate_paths(returns, current_price, n_days, garch, n_sims, seed=seed)

    percentiles = []
    for day in checkpoints:
        if day > n_days:
            continue
        column = paths[:, day - 1]
        p5, p25, p50, p75, p95 = np.percentile(column, [5, 25, 50, 75, 95])
        percentiles.append(PricePercentiles(day=day, p5=p5, p25=p25, p50=p50, p75=p75, p95=p95))

    probability_of_loss = float((paths[:, -1] < current_price).mean())

    stop_first = target_first = neither = None
    if stop_loss is not None and take_profit is not None:
        stop_first, target_first, neither = barrier_probabilities(paths, stop_loss, take_profit)

    return MonteCarloResult(
        n_simulations=n_sims,
        method=method,
        percentiles=percentiles,
        probability_of_loss=probability_of_loss,
        probability_stop_before_target=stop_first,
        probability_target_before_stop=target_first,
        probability_neither_hit=neither,
    )
