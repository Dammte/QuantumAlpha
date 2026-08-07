import numpy as np
import pandas as pd
import pytest

from app.services import monte_carlo_simulation as mcs
from app.services import volatility_model as vm


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * rng.normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.normal()
    return eps


def test_block_bootstrap_matrix_shape_and_values_come_from_source():
    rng = np.random.default_rng(1)
    source = np.arange(100).astype(float)
    matrix = mcs._block_bootstrap_matrix(source, n_sims=50, n_days=21, block_size=5, rng=rng)
    assert matrix.shape == (50, 21)
    assert set(np.unique(matrix)).issubset(set(source))


def test_block_bootstrap_matrix_falls_back_to_iid_when_source_shorter_than_block():
    rng = np.random.default_rng(1)
    source = np.array([1.0, 2.0, 3.0])
    matrix = mcs._block_bootstrap_matrix(source, n_sims=10, n_days=5, block_size=10, rng=rng)
    assert matrix.shape == (10, 5)
    assert set(np.unique(matrix)).issubset({1.0, 2.0, 3.0})


def test_simulate_paths_without_garch_shape_and_positivity():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0005, 0.015, 500))
    paths, method = mcs.simulate_paths(returns, current_price=100.0, n_days=21, garch=None, n_sims=200, seed=1)
    assert method == "block_bootstrap"
    assert paths.shape == (200, 21)
    assert np.all(paths > 0)


def test_simulate_paths_with_garch_shape_and_positivity():
    eps = _simulate_garch(2000, 0.00002, 0.08, 0.88)
    returns = pd.Series(eps)
    garch = vm.fit_garch(returns)
    assert garch is not None

    paths, method = mcs.simulate_paths(returns, current_price=50.0, n_days=21, garch=garch, n_sims=200, seed=2)
    assert method == "garch_filtered"
    assert paths.shape == (200, 21)
    assert np.all(paths > 0)


def test_simulate_paths_mean_drift_roughly_matches_historical_mean():
    rng = np.random.default_rng(9)
    mean_return = 0.002
    returns = pd.Series(rng.normal(mean_return, 0.01, 800))
    paths, _ = mcs.simulate_paths(returns, current_price=100.0, n_days=1, garch=None, n_sims=5000, seed=3)
    simulated_mean_return = (paths[:, 0].mean() - 100.0) / 100.0
    assert simulated_mean_return == pytest.approx(mean_return, abs=0.01)


def test_barrier_probabilities_stop_hit_when_all_paths_decline_below_stop():
    paths = np.tile(np.linspace(100, 80, 10), (5, 1))
    stop_first, target_first, neither = mcs.barrier_probabilities(paths, stop_loss=90.0, take_profit=150.0)
    assert stop_first == pytest.approx(1.0)
    assert target_first == pytest.approx(0.0)
    assert neither == pytest.approx(0.0)


def test_barrier_probabilities_target_hit_when_all_paths_rise_above_target():
    paths = np.tile(np.linspace(100, 130, 10), (5, 1))
    stop_first, target_first, neither = mcs.barrier_probabilities(paths, stop_loss=50.0, take_profit=120.0)
    assert target_first == pytest.approx(1.0)
    assert stop_first == pytest.approx(0.0)
    assert neither == pytest.approx(0.0)


def test_barrier_probabilities_neither_hit_when_barriers_far_away():
    paths = np.tile(np.linspace(99, 101, 10), (5, 1))
    stop_first, target_first, neither = mcs.barrier_probabilities(paths, stop_loss=10.0, take_profit=1000.0)
    assert neither == pytest.approx(1.0)
    assert stop_first == pytest.approx(0.0)
    assert target_first == pytest.approx(0.0)


def test_barrier_probabilities_mixed_outcomes():
    # 2 paths hit stop, 2 hit target, 1 hits neither - all resolved on distinct days.
    paths = np.array(
        [
            [95, 89, 89, 89],
            [95, 88, 88, 88],
            [105, 111, 111, 111],
            [105, 112, 112, 112],
            [100, 100, 100, 100],
        ]
    )
    stop_first, target_first, neither = mcs.barrier_probabilities(paths, stop_loss=90.0, take_profit=110.0)
    assert stop_first == pytest.approx(0.4)
    assert target_first == pytest.approx(0.4)
    assert neither == pytest.approx(0.2)


def test_simulate_and_analyze_none_for_insufficient_history():
    returns = pd.Series(np.random.default_rng(5).normal(0, 0.01, 100))
    assert mcs.simulate_and_analyze(returns, current_price=100.0, garch=None) is None


def test_simulate_and_analyze_percentiles_are_ordered_and_probabilities_valid():
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0002, 0.015, 1500))
    result = mcs.simulate_and_analyze(
        returns,
        current_price=100.0,
        garch=None,
        stop_loss=90.0,
        take_profit=115.0,
        n_days=63,
        n_sims=500,
        seed=42,
    )
    assert result is not None
    assert result.method == "block_bootstrap"
    assert len(result.percentiles) == 4
    for p in result.percentiles:
        assert p.p5 <= p.p25 <= p.p50 <= p.p75 <= p.p95
    assert 0.0 <= result.probability_of_loss <= 1.0
    assert result.probability_stop_before_target is not None
    assert result.probability_target_before_stop is not None
    assert result.probability_neither_hit is not None
    total = (
        result.probability_stop_before_target
        + result.probability_target_before_stop
        + result.probability_neither_hit
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_simulate_and_analyze_uses_garch_when_provided():
    eps = _simulate_garch(2000, 0.00002, 0.08, 0.88)
    returns = pd.Series(eps)
    garch = vm.fit_garch(returns)
    assert garch is not None

    result = mcs.simulate_and_analyze(returns, current_price=50.0, garch=garch, n_sims=300, seed=1)
    assert result is not None
    assert result.method == "garch_filtered"
    assert result.probability_stop_before_target is None
    assert result.probability_target_before_stop is None
    assert result.probability_neither_hit is None


def test_simulate_and_analyze_no_barrier_probabilities_when_stop_or_target_missing():
    rng = np.random.default_rng(12)
    returns = pd.Series(rng.normal(0.0002, 0.015, 1500))
    result = mcs.simulate_and_analyze(returns, current_price=100.0, garch=None, stop_loss=90.0, n_sims=100)
    assert result is not None
    assert result.probability_stop_before_target is None
