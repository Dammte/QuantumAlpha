import numpy as np
import pandas as pd

from app.services import statistical_structure as ss


def test_hurst_none_with_insufficient_history():
    prices = pd.Series(100 + np.arange(100) * 0.1)
    assert ss.hurst_exponent(prices) is None


def test_hurst_near_half_for_a_random_walk():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 1500)))
    h = ss.hurst_exponent(prices)
    assert h is not None
    assert 0.35 < h < 0.65  # random walk - no strong persistence either way


def test_hurst_above_half_for_a_persistently_trending_series():
    rng = np.random.default_rng(2)
    prices = pd.Series(100 + np.arange(1500) * 0.1 + rng.normal(0, 0.5, 1500).cumsum() * 0.05)
    h = ss.hurst_exponent(prices)
    assert h is not None
    assert h > 0.55


def test_hurst_below_half_for_a_mean_reverting_series():
    rng = np.random.default_rng(3)
    n = 1500
    prices = np.zeros(n)
    prices[0] = 100.0
    theta, mu, sigma = 0.05, 100.0, 1.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + rng.normal(0, sigma)
    h = ss.hurst_exponent(pd.Series(prices))
    assert h is not None
    assert h < 0.45


def test_adf_stationarity_none_with_insufficient_history():
    prices = pd.Series(100 + np.arange(100) * 0.1)
    assert ss.adf_stationarity(prices) is None


def test_adf_stationarity_detects_a_mean_reverting_series_as_stationary():
    rng = np.random.default_rng(4)
    n = 1500
    prices = np.zeros(n)
    prices[0] = 100.0
    theta, mu, sigma = 0.05, 100.0, 1.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + rng.normal(0, sigma)
    result = ss.adf_stationarity(pd.Series(prices))
    assert result is not None
    _, p_value = result
    assert p_value < 0.05


def test_classify_regime_bands():
    assert ss.classify_regime(None) == ss.REGIME_UNKNOWN
    assert ss.classify_regime(0.7) == ss.REGIME_TRENDING
    assert ss.classify_regime(0.3) == ss.REGIME_MEAN_REVERTING
    assert ss.classify_regime(0.5) == ss.REGIME_RANDOM


def test_compute_statistical_structure_returns_none_fields_on_insufficient_data():
    prices = pd.Series(100 + np.arange(50) * 0.1)
    result = ss.compute_statistical_structure(prices)
    assert result.hurst_exponent is None
    assert result.regime == ss.REGIME_UNKNOWN
    assert result.adf_statistic is None
    assert result.adf_p_value is None
    assert result.is_stationary is None


def test_compute_statistical_structure_full_pipeline_on_trending_series():
    rng = np.random.default_rng(5)
    prices = pd.Series(100 + np.arange(1500) * 0.1 + rng.normal(0, 0.5, 1500).cumsum() * 0.05)
    result = ss.compute_statistical_structure(prices)
    assert result.hurst_exponent is not None
    assert result.regime == ss.REGIME_TRENDING
    assert result.adf_p_value is not None
    assert result.is_stationary is not None
