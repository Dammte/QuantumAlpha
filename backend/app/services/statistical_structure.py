"""Hurst exponent + Augmented Dickey-Fuller (ADF) test: whether a ticker's own
price history actually behaves as trending/persistent or mean-reverting/
anti-persistent, and whether it's statistically stationary.

An external audit of this app correctly pointed out that a rule-based
checklist mixing trend-following factors (MA alignment, Weinstein stage) with
mean-reversion ones (RSI oversold bounce) is silently assuming every ticker
behaves the same way - when whether momentum or reversion actually describes
a given name is itself an empirical, testable question, not a universal
constant. This module answers that question directly:

- Hurst exponent (H), estimated via rescaled-range (R/S) analysis on log
  prices: H > 0.5 means the series is persistent/trending (an up move tends
  to be followed by another up move - momentum has room to work). H < 0.5
  means anti-persistent/mean-reverting (an up move tends to be followed by a
  down move - oscillators have more to work with). H ~= 0.5 is indistinguishable
  from a random walk, where neither family has a real edge.
- Augmented Dickey-Fuller test: rejects (low p-value) the null hypothesis of
  a unit root, i.e. the series has genuine mean-reversion structure rather
  than behaving like an unpredictable random walk with drift. Uses
  statsmodels' implementation (the standard, correctly-calibrated one - the
  critical values for this test come from response-surface tables that
  aren't worth re-deriving by hand) rather than a from-scratch reimplementation,
  unlike this app's other statistical models (GARCH, the Markov chain), which
  are simple enough to implement correctly without needing a heavy dependency.

Only used as a small, one-directional caution flag in the recommendation
engine (see `recommendation_engine.py`) - a genuinely mean-reverting Hurst
reading (H < 0.45) softens confidence in the trend-following factors already
scored elsewhere, rather than being scored as its own bullish/bearish vote
(that would double-count the same trend/stage facts from a different angle,
the exact mistake this whole audit set out to fix).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

MIN_OBSERVATIONS = 252  # need at least a year of daily bars for a stable R/S estimate
MIN_WINDOW = 10
HURST_TRENDING_THRESHOLD = 0.55
HURST_MEAN_REVERTING_THRESHOLD = 0.45
ADF_SIGNIFICANCE = 0.05

REGIME_TRENDING = "tendencial"
REGIME_MEAN_REVERTING = "reversion"
REGIME_RANDOM = "aleatorio"
REGIME_UNKNOWN = "desconocido"


@dataclass(frozen=True, slots=True)
class StatisticalStructure:
    hurst_exponent: float | None
    regime: str  # "tendencial" | "reversion" | "aleatorio" | "desconocido"
    adf_statistic: float | None
    adf_p_value: float | None
    is_stationary: bool | None  # ADF rejects the unit-root null at 5%


def hurst_exponent(prices: pd.Series, min_window: int = MIN_WINDOW, max_window: int | None = None) -> float | None:
    """Rescaled-range (R/S) Hurst exponent estimate on log prices: split the
    series into windows of several sizes, compute the mean R/S statistic
    (range of the mean-adjusted cumulative return series, divided by its
    standard deviation) at each size, then take the slope of log(R/S) vs
    log(window size) - that slope IS the Hurst exponent, by definition of
    the R/S method (Hurst 1951; Mandelbrot's later formalization)."""
    clean = prices.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        return None

    log_prices = np.log(clean.to_numpy())
    n = len(log_prices)
    max_window = max_window or n // 2
    if max_window <= min_window:
        return None

    candidate_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int))
    candidate_sizes = candidate_sizes[candidate_sizes >= min_window]

    log_sizes: list[float] = []
    log_rs: list[float] = []
    for window in candidate_sizes:
        n_chunks = n // window
        if n_chunks < 1:
            continue
        rs_for_window = []
        for chunk_idx in range(n_chunks):
            chunk = log_prices[chunk_idx * window : (chunk_idx + 1) * window]
            returns = np.diff(chunk)
            if len(returns) == 0:
                continue
            std = returns.std()
            if std == 0:
                continue
            mean_adjusted = returns - returns.mean()
            cumulative_deviation = np.cumsum(mean_adjusted)
            rescaled_range = cumulative_deviation.max() - cumulative_deviation.min()
            rs_for_window.append(rescaled_range / std)
        if rs_for_window:
            log_sizes.append(float(np.log(window)))
            log_rs.append(float(np.log(np.mean(rs_for_window))))

    if len(log_sizes) < 4:  # need enough distinct window sizes for a stable regression
        return None

    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def adf_stationarity(prices: pd.Series) -> tuple[float, float] | None:
    """(test statistic, p-value) from an Augmented Dickey-Fuller test on log
    prices, lag order chosen by AIC. Returns None rather than raising on
    insufficient data or a numerical failure inside statsmodels - treated the
    same as every other quant model in this app that can't always produce a
    reading (GARCH, Markov)."""
    clean = prices.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        return None
    try:
        statistic, p_value, *_ = adfuller(np.log(clean.to_numpy()), autolag="AIC")
    except (ValueError, np.linalg.LinAlgError):
        return None
    return float(statistic), float(p_value)


def classify_regime(hurst: float | None) -> str:
    if hurst is None:
        return REGIME_UNKNOWN
    if hurst >= HURST_TRENDING_THRESHOLD:
        return REGIME_TRENDING
    if hurst <= HURST_MEAN_REVERTING_THRESHOLD:
        return REGIME_MEAN_REVERTING
    return REGIME_RANDOM


def compute_statistical_structure(prices: pd.Series) -> StatisticalStructure:
    hurst = hurst_exponent(prices)
    adf_result = adf_stationarity(prices)
    adf_statistic, adf_p_value = adf_result if adf_result is not None else (None, None)
    return StatisticalStructure(
        hurst_exponent=hurst,
        regime=classify_regime(hurst),
        adf_statistic=adf_statistic,
        adf_p_value=adf_p_value,
        is_stationary=(adf_p_value < ADF_SIGNIFICANCE) if adf_p_value is not None else None,
    )
