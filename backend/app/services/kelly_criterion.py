"""Kelly Criterion position sizing: converts a win-probability / reward-risk
estimate into a suggested fraction of capital to risk, then deliberately
gives back growth for safety - exactly the way real quant/CTA shops do it.

Full Kelly (`f* = p - (1-p)/R`) is growth-optimal only under perfectly known,
static probabilities. Real probability estimates are noisy, so betting full
Kelly systematically overbets relative to the true edge and produces far
larger drawdowns than the formula assumes. This module defaults to **half
Kelly**: `G(c)/G(f*) = c*(2-c)`, so betting half of full Kelly retains 75% of
the theoretical long-run growth rate while cutting bet size (and therefore
variance/drawdown) in half. It then scales the result down further in
elevated-volatility regimes (from the GARCH fit) and hard-caps the final
suggestion regardless of how favorable the inputs look. Varying position size
to protect capital - not any single predictive signal - is the specific
practice the research on quant funds like Renaissance Technologies points to
as central to their edge.
"""

from dataclasses import dataclass

MAX_POSITION_FRACTION = 0.25
DEFAULT_KELLY_FRACTION = 0.5
VOL_REGIME_SIZE_MULTIPLIER = {"baja": 1.0, "normal": 1.0, "elevada": 0.75, "alta": 0.5}


def kelly_fraction_binary(win_probability: float, reward_risk_ratio: float) -> float:
    """f* = p - (1-p)/R - the growth-optimal bet fraction for a binary payoff
    (lose 1 unit or win R units) with known win probability p."""
    if reward_risk_ratio <= 0:
        return 0.0
    return win_probability - (1 - win_probability) / reward_risk_ratio


def win_probability_from_barriers(prob_target_first: float, prob_stop_first: float) -> float | None:
    """Conditional win probability given the trade actually resolves one way or
    the other within the simulated horizon - paths that hit neither barrier
    don't inform a win/loss estimate, so they're excluded from this ratio."""
    resolved = prob_target_first + prob_stop_first
    if resolved <= 0:
        return None
    return prob_target_first / resolved


@dataclass(frozen=True, slots=True)
class KellyResult:
    win_probability: float
    reward_risk_ratio: float
    full_kelly_fraction: float
    fractional_kelly_fraction: float
    recommended_position_pct: float
    growth_rate_retained_pct: float
    rationale: str


def recommend_position_size(
    win_probability: float,
    reward_risk_ratio: float,
    vol_regime: str | None = None,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> KellyResult:
    full_kelly = kelly_fraction_binary(win_probability, reward_risk_ratio)
    fractional = full_kelly * kelly_fraction
    vol_multiplier = VOL_REGIME_SIZE_MULTIPLIER.get(vol_regime, 1.0) if vol_regime else 1.0
    vol_adjusted = fractional * vol_multiplier
    recommended = max(0.0, min(vol_adjusted, MAX_POSITION_FRACTION))
    growth_retained = kelly_fraction * (2 - kelly_fraction)

    if full_kelly <= 0:
        rationale = (
            "Kelly negativo o nulo: con esta probabilidad de éxito y relación riesgo:beneficio, la "
            "ventaja esperada no compensa el riesgo - no se recomienda abrir posición según este criterio."
        )
    elif vol_adjusted > MAX_POSITION_FRACTION:
        rationale = (
            f"El Kelly fraccional ({vol_adjusted:.1%}) supera el techo máximo de posición "
            f"({MAX_POSITION_FRACTION:.0%} del capital) - limitado por disciplina de riesgo, no por el cálculo."
        )
    else:
        vol_note = f", ajustado por volatilidad GARCH ({vol_regime})" if vol_multiplier != 1.0 else ""
        rationale = (
            f"{kelly_fraction:.0%} del Kelly completo ({full_kelly:.1%}){vol_note}, reteniendo "
            f"~{growth_retained:.0%} de la tasa de crecimiento óptima con menor varianza y drawdown."
        )

    return KellyResult(
        win_probability=win_probability,
        reward_risk_ratio=reward_risk_ratio,
        full_kelly_fraction=full_kelly,
        fractional_kelly_fraction=fractional,
        recommended_position_pct=recommended,
        growth_rate_retained_pct=growth_retained,
        rationale=rationale,
    )
