from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RecommendationSnapshot:
    """Domain read of `RecommendationSnapshotORM` - see that class's docstring
    for why it exists (the audit trail of every real "Analizar activo" call).
    Only the fields `signal_performance_service.py` actually needs to measure
    whether past verdicts panned out - not the full factor breakdown, which
    stays ORM-only for now (nothing outside the single-ticker history
    endpoint reads it yet)."""

    ticker: str
    created_at: datetime
    verdict: str  # "comprar" | "esperar" | "evitar"
    score: int
    price: float
