import numpy as np
import pandas as pd
import pytest

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


def test_conditional_variance_path_matches_manual_recursion():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    omega, alpha, beta = 0.0001, 0.1, 0.85
    path = vm.conditional_variance_path(returns, omega, alpha, beta)

    expected = [max(np.var(returns), vm.VARIANCE_FLOOR)]
    for t in range(1, len(returns)):
        expected.append(omega + alpha * returns[t - 1] ** 2 + beta * expected[t - 1])

    assert path == pytest.approx(expected)


def test_fit_garch_none_with_insufficient_data():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50))
    assert vm.fit_garch(returns) is None


def test_fit_garch_none_for_constant_series():
    returns = pd.Series([0.0] * 300)
    assert vm.fit_garch(returns) is None


def test_fit_garch_recovers_plausible_parameters_on_synthetic_data():
    true_omega, true_alpha, true_beta = 0.00002, 0.08, 0.88
    eps = _simulate_garch(3000, true_omega, true_alpha, true_beta)
    result = vm.fit_garch(pd.Series(eps))

    assert result is not None
    assert 0 <= result.alpha < 1
    assert 0 <= result.beta < 1
    assert result.persistence < 1
    # Estimation noise on 3000 points is real - check the ballpark, not exact recovery.
    assert result.persistence == pytest.approx(true_alpha + true_beta, abs=0.15)
    assert result.current_vol_annualized > 0
    assert result.unconditional_vol_annualized > 0
    assert result.forecast_vol_21d_annualized > 0


def test_fit_garch_regime_label_reflects_percentile():
    true_omega, true_alpha, true_beta = 0.00002, 0.08, 0.88
    eps = _simulate_garch(2000, true_omega, true_alpha, true_beta)
    result = vm.fit_garch(pd.Series(eps))
    assert result is not None
    assert result.regime in {"baja", "normal", "elevada", "alta"}
    assert 0.0 <= result.vol_percentile <= 1.0


@pytest.mark.parametrize(
    "percentile,expected",
    [(0.05, "baja"), (0.4, "normal"), (0.7, "elevada"), (0.95, "alta")],
)
def test_regime_label_thresholds(percentile, expected):
    assert vm._regime_label(percentile) == expected


def test_standardized_residuals_have_roughly_unit_variance():
    true_omega, true_alpha, true_beta = 0.00002, 0.08, 0.88
    eps = _simulate_garch(3000, true_omega, true_alpha, true_beta)
    series = pd.Series(eps)
    result = vm.fit_garch(series)
    assert result is not None

    z = vm.standardized_residuals(series, result)
    assert np.std(z) == pytest.approx(1.0, abs=0.25)
