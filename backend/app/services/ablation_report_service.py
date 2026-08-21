"""Reads `scripts/factor_ablation_study.py`'s own saved CSV output
(`docs/factor_ablation_report_v2_h*.csv`) and exposes it to "Rendimiento del
sistema" and the ticker card - Segunda auditoría, Bloque 4:
docs/quant_methodology.md documents 24 of these CSVs, and until now nothing
in the app ever read them back - the weight-vs-measured-sign comparison this
whole audit's philosophy depends on ("mide antes de confiar") was only ever
visible to someone opening a CSV by hand.

Never re-runs the study - reads whatever the last actual run wrote, exactly
as written. `scripts/factor_ablation_study.py` is still the only thing that
regenerates these files (see docs/quant_methodology.md §12 on why that's a
slow, offline, owner-triggered process, not something an API request does).
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# Every horizon scripts/factor_ablation_study.py has actually been run at and
# saved a v2 report for (docs/quant_methodology.md §12) - not every one of
# these files is guaranteed to exist at any given moment (a fresh clone, or a
# future re-run that only covers a subset), so `load_ablation_report` checks
# the file itself rather than trusting this list blindly.
AVAILABLE_HORIZONS_DAYS = (5, 21, 63, 126)

# recommendation_engine.py's RecommendationFactor.label (a free-text Spanish
# sentence, not a programmatic key) -> the ablation study's own factor key
# (see factor_ablation_study.CURRENT_POINTS, which documents this exact
# correspondence). Hand-maintained on purpose: a label with no entry here
# simply never gets an ablation comparison (nothing crashes, nothing is
# assumed consistent) - some engine factors (RS Rating threshold, support
# proximity, the 8/8 Minervini AND-gate, fundamentals, GARCH regime, Markov,
# Hurst) were never individually tracked as their own boolean ablation key,
# so they're intentionally absent, not overlooked.
FACTOR_LABEL_TO_ABLATION_KEY: dict[str, str] = {
    "Tendencia alcista (MA20 > MA50 > MA200)": "trend_up",
    "Tendencia bajista - evitar entradas largas": "trend_down",
    "Fase 2 de Weinstein (avance)": "stage2",
    "Fase 4 de Weinstein (declive)": "stage4",
    "Movimiento confirmado: precio 25%+ sobre su mínimo anual y dentro del 25% de su máximo anual": (
        "minervini_range_position"
    ),
    "Golden cross reciente (MA50/MA200)": "golden_cross",
    "Death cross reciente (MA50/MA200)": "death_cross",
    "Tendencia fuerte y confirmada (ADX ≥ 25, +DI > -DI)": "adx_strong_trend",
    "Sobrecompra extrema (RSI ≥ 80) fuera de una tendencia fuerte confirmada": (
        "rsi_overbought_outside_strong_trend"
    ),
    "Sobreventa (RSI ≤ 30): posible rebote técnico": "rsi_oversold_bounce",
    "Extensión parabólica (riesgo de reversión a corto plazo)": "atr_parabolic",
    "Divergencia bajista de volumen (OBV): el avance no está respaldado por compras reales": "obv_bearish",
    "Divergencia alcista de volumen (OBV): la presión vendedora se agota pese a la caída de precio": "obv_bullish",
}


@dataclass(frozen=True, slots=True)
class FactorAblationResult:
    factor: str
    current_points: int
    mean_difference_pct: float
    directionally_consistent: bool
    significant_at_1pct_bh: bool
    mean_ic: float | None
    ic_ir: float | None
    n_ic_buckets: int
    multivariate_coef_pct: float | None
    multivariate_p_value: float | None


def _report_path(horizon_days: int) -> Path:
    return DOCS_DIR / f"factor_ablation_report_v2_h{horizon_days}.csv"


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_ablation_report(horizon_days: int) -> list[FactorAblationResult]:
    """Whatever `scripts/factor_ablation_study.py`'s last real run wrote for
    this horizon - `[]` if that horizon was never run (the file doesn't
    exist) or the file can't be parsed, never a fabricated result."""
    path = _report_path(horizon_days)
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as f:
            return [
                FactorAblationResult(
                    factor=row["factor"],
                    current_points=int(row["current_points"]),
                    mean_difference_pct=float(row["mean_difference_pct"]),
                    directionally_consistent=row["directionally_consistent"] == "True",
                    significant_at_1pct_bh=row["significant_at_1pct_bh"] == "True",
                    mean_ic=_float_or_none(row.get("mean_ic")),
                    ic_ir=_float_or_none(row.get("ic_ir")),
                    n_ic_buckets=int(row["n_ic_buckets"]),
                    multivariate_coef_pct=_float_or_none(row.get("multivariate_coef_pct")),
                    multivariate_p_value=_float_or_none(row.get("multivariate_p_value")),
                )
                for row in csv.DictReader(f)
            ]
    except Exception:
        logger.exception("Ablation report: failed to read/parse horizon=%d", horizon_days)
        return []


def sign_mismatched_factor_keys(horizon_days: int) -> set[str]:
    """Which ablation factor keys measured an effect opposite their current
    `recommendation_engine.py` weight's sign, at this horizon -
    `directionally_consistent == False` in the study's own saved output,
    nothing recomputed here."""
    return {r.factor for r in load_ablation_report(horizon_days) if not r.directionally_consistent}


def triggered_factors_with_contradicted_sign(factors: list, horizon_days: int) -> list[str]:
    """Which of a `Recommendation`'s own *triggered* factors
    (`recommendation_engine.RecommendationFactor`) map to an ablation-measured
    sign contradiction at this horizon - human-readable labels, ready to show
    on a ticker card. A triggered factor with no entry in
    `FACTOR_LABEL_TO_ABLATION_KEY` (no ablation key, or never individually
    measured) is never flagged - absence of evidence isn't evidence of a
    contradiction."""
    mismatched_keys = sign_mismatched_factor_keys(horizon_days)
    if not mismatched_keys:
        return []
    return [
        f.label for f in factors if f.triggered and FACTOR_LABEL_TO_ABLATION_KEY.get(f.label) in mismatched_keys
    ]
