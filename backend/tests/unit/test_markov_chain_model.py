import numpy as np
import pandas as pd
import pytest

from app.services import markov_chain_model as mc


def test_discretize_states_bins_by_rolling_zscore():
    # Constant small noise around zero, then one huge spike - the spike should
    # land in the extreme "fuerte alcista" bin (state 4) once the rolling
    # window has enough history to establish a tight mean/std.
    values = [0.001, -0.001, 0.001, -0.001] * 10 + [0.10]
    returns = pd.Series(values)
    states = mc.discretize_states(returns, window=20)
    assert states.iloc[-1] == 4


def test_estimate_transition_matrix_rows_sum_to_one():
    states = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3])
    matrix = mc.estimate_transition_matrix(states, n_states=5)
    assert matrix.shape == (5, 5)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


def test_estimate_transition_matrix_deterministic_sequence_dominant_transition():
    # State 0 always goes to state 1 in this toy sequence - with enough
    # repetitions the smoothed probability should still clearly favor 0->1.
    states = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] * 5)
    matrix = mc.estimate_transition_matrix(states, n_states=5, smoothing=0.5)
    assert matrix[0].argmax() == 1
    assert matrix[0, 1] > 0.8


def test_state_mean_returns_matches_manual_average():
    returns = np.array([0.01, 0.02, -0.01, -0.02, 0.03])
    states = np.array([0, 0, 1, 1, 0])
    means = mc.state_mean_returns(returns, states, n_states=3)
    assert means[0] == pytest.approx((0.01 + 0.02 + 0.03) / 3)
    assert means[1] == pytest.approx((-0.01 - 0.02) / 2)
    assert means[2] == pytest.approx(0.0)  # never observed - defaults to 0


def test_stationary_distribution_is_a_fixed_point():
    matrix = mc.estimate_transition_matrix(
        np.array([0, 1, 2, 3, 4, 2, 1, 3, 0, 2, 4, 1, 3, 2, 0] * 10), n_states=5
    )
    pi = mc.stationary_distribution(matrix)
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ matrix, pi, atol=1e-6)


def test_forecast_distribution_zero_steps_is_one_hot():
    matrix = np.eye(5) * 0.6 + 0.1
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    dist = mc.forecast_distribution(matrix, current_state=2, n_steps=0)
    expected = np.zeros(5)
    expected[2] = 1.0
    np.testing.assert_allclose(dist, expected)


def test_forecast_distribution_matches_matrix_power():
    matrix = mc.estimate_transition_matrix(np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 1] * 8), n_states=3)
    dist = mc.forecast_distribution(matrix, current_state=1, n_steps=3)
    expected = np.linalg.matrix_power(matrix, 3)[1]
    np.testing.assert_allclose(dist, expected, atol=1e-8)


def test_expected_cumulative_return_compounds_a_fixed_return_state():
    # An absorbing state (always transitions to itself) with a known mean
    # return - the n-step expectation must equal simple compounding (1+mu)^n-1.
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    means = np.array([0.02, -0.01])
    result = mc.expected_cumulative_return(matrix, means, current_state=0, n_steps=10)
    assert result == pytest.approx(1.02**10 - 1)


def test_runs_test_detects_a_clearly_non_random_alternating_sequence():
    is_up = np.array([1, 0] * 50)  # perfectly alternating - far too many runs
    z, is_random = mc.runs_test(is_up)
    assert not is_random
    assert abs(z) > 1.96


def test_runs_test_detects_a_clearly_clustered_sequence():
    is_up = np.array([1] * 25 + [0] * 25)  # only 2 runs - far too few
    z, is_random = mc.runs_test(is_up)
    assert not is_random


def test_runs_test_handles_all_one_class():
    z, is_random = mc.runs_test(np.array([1, 1, 1, 1]))
    assert is_random


def test_order2_justification_test_detects_genuine_second_order_dependence():
    # Build a sequence where the next state is deterministically the SUM of the
    # last two states (mod 3) - a real, strong order-2 dependency an order-1
    # model cannot capture, repeated enough times to be statistically obvious.
    rng = np.random.default_rng(3)
    states = [int(rng.integers(0, 3)), int(rng.integers(0, 3))]
    for _ in range(3000):
        states.append((states[-1] + states[-2]) % 3)
    g2, p_value, justified = mc.order2_justification_test(np.array(states), n_states=3)
    assert justified
    assert p_value < 0.01


def test_order2_justification_test_not_justified_for_iid_noise():
    rng = np.random.default_rng(4)
    states = rng.integers(0, 3, size=1500)
    g2, p_value, justified = mc.order2_justification_test(states, n_states=3)
    assert not justified
    assert p_value > 0.01


def test_analyze_markov_chain_none_for_insufficient_history():
    returns = pd.Series(np.random.default_rng(5).normal(0, 0.01, 100))
    assert mc.analyze_markov_chain(returns) is None


def test_analyze_markov_chain_full_result_shape_on_realistic_data():
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0.0003, 0.015, 1500))
    result = mc.analyze_markov_chain(returns)

    assert result is not None
    assert 0 <= result.current_state < mc.N_STATES
    assert result.current_state_label in mc.STATE_LABELS
    assert len(result.transition_matrix) == mc.N_STATES
    assert all(len(row) == mc.N_STATES for row in result.transition_matrix)
    assert len(result.state_mean_returns) == mc.N_STATES
    assert len(result.stationary_distribution) == mc.N_STATES
    assert sum(result.stationary_distribution) == pytest.approx(1.0, abs=1e-6)
    assert len(result.forecast_21d_distribution) == mc.N_STATES
    assert sum(result.forecast_21d_distribution) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= result.prob_bullish_21d <= 1.0
    assert isinstance(result.sequence_looks_random, bool)
    assert isinstance(result.order2_justified, bool)
