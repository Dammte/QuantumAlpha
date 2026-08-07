"""GARCH(1,1) conditional-volatility model, fit from scratch via maximum
likelihood (no external GARCH library).

sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

Fit by maximizing the Gaussian log-likelihood over (omega, alpha, beta),
reparameterized so the optimizer runs unconstrained yet the recovered
parameters always satisfy omega>0, alpha>=0, beta>=0, alpha+beta<1 (required
for a finite, mean-reverting long-run variance):

  omega = exp(theta1)
  alpha = sigmoid(theta2)
  beta  = (1 - alpha) * sigmoid(theta3)

Starting values follow the standard heuristic for daily equity/stock returns
(persistence alpha+beta ~ 0.95, alpha ~ 0.05) via variance targeting - this is
what resolves most "GARCH won't converge" problems when fitting from scratch.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 250
VARIANCE_FLOOR = 1e-10


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _unpack(theta: np.ndarray) -> tuple[float, float, float]:
    omega = float(np.exp(theta[0]))
    alpha = float(_sigmoid(theta[1]))
    beta = float((1 - alpha) * _sigmoid(theta[2]))
    return omega, alpha, beta


def conditional_variance_path(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """sigma_t^2 for every t, seeded with the full-sample unconditional variance."""
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = max(np.var(returns), VARIANCE_FLOOR)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], VARIANCE_FLOOR)
    return sigma2


def _neg_log_likelihood(theta: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = _unpack(theta)
    sigma2 = conditional_variance_path(returns, omega, alpha, beta)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -ll if np.isfinite(ll) else 1e10


@dataclass(frozen=True, slots=True)
class GarchResult:
    omega: float
    alpha: float
    beta: float
    persistence: float  # alpha + beta: how slowly volatility shocks decay
    unconditional_vol_annualized: float
    current_vol_annualized: float
    forecast_vol_21d_annualized: float
    vol_percentile: float  # current vol's percentile rank in its own 2y history
    regime: str  # "baja" | "normal" | "elevada" | "alta"


def _regime_label(percentile: float) -> str:
    if percentile < 0.20:
        return "baja"
    if percentile < 0.60:
        return "normal"
    if percentile < 0.85:
        return "elevada"
    return "alta"


def fit_garch(returns: pd.Series) -> GarchResult | None:
    """Fits GARCH(1,1) on demeaned daily returns. Returns None rather than
    raising if there isn't enough history or the optimizer fails to find a
    stable (alpha+beta<1) solution - callers should treat a missing volatility
    forecast as a graceful degradation, not an error."""
    clean = returns.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        return None

    values = (clean - clean.mean()).to_numpy()
    sample_var = np.var(values)
    if sample_var <= 0:
        return None

    alpha0, beta0 = 0.05, 0.90
    omega0 = sample_var * (1 - alpha0 - beta0)
    theta0 = np.array(
        [
            np.log(max(omega0, 1e-12)),
            np.log(alpha0 / (1 - alpha0)),  # inverse sigmoid
            np.log(beta0 / (1 - beta0)),
        ]
    )

    result = minimize(_neg_log_likelihood, theta0, args=(values,), method="BFGS")
    if not result.success and result.status not in (0, 2):  # 2 = "precision loss", usually still usable
        return None

    omega, alpha, beta = _unpack(result.x)
    persistence = alpha + beta
    if not (0 < omega < 10 and 0 <= alpha < 1 and 0 <= beta < 1 and persistence < 0.9999):
        return None

    sigma2 = conditional_variance_path(values, omega, alpha, beta)
    unconditional_var = omega / (1 - persistence)
    current_var = sigma2[-1]

    # Multi-step-ahead forecast: variance mean-reverts geometrically to the
    # unconditional level. E[sigma^2_{t+n}] = unconditional + persistence^(n-1) * (sigma^2_{t+1} - unconditional)
    horizon = 21
    next_var = omega + alpha * values[-1] ** 2 + beta * current_var
    forecast_var = unconditional_var + (persistence ** (horizon - 1)) * (next_var - unconditional_var)
    forecast_var = max(forecast_var, VARIANCE_FLOOR)

    trailing = sigma2[-504:] if len(sigma2) > 504 else sigma2
    vol_percentile = float((trailing < current_var).mean())

    return GarchResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        unconditional_vol_annualized=float(np.sqrt(unconditional_var * TRADING_DAYS_PER_YEAR)),
        current_vol_annualized=float(np.sqrt(current_var * TRADING_DAYS_PER_YEAR)),
        forecast_vol_21d_annualized=float(np.sqrt(forecast_var * TRADING_DAYS_PER_YEAR)),
        vol_percentile=vol_percentile,
        regime=_regime_label(vol_percentile),
    )


def standardized_residuals(returns: pd.Series, garch: GarchResult) -> np.ndarray:
    """z_t = eps_t / sigma_t - used by the Monte Carlo module for filtered
    historical simulation (resample these instead of raw returns, then
    re-inflate by a simulated forward volatility path)."""
    clean = returns.dropna()
    values = (clean - clean.mean()).to_numpy()
    sigma2 = conditional_variance_path(values, garch.omega, garch.alpha, garch.beta)
    return values / np.sqrt(sigma2)
