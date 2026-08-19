from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PositionSignalSnapshot:
    """One held position's signal at one point in time - the audit trail
    `RecommendationSnapshotORM` never had, because it only ever recorded the
    buy-side verdict from "Analizar activo", not `portfolio_risk_service.py`'s
    position-level signal/exit_urgency (see docs/quant_methodology.md's Fase 0
    note: this is why "how many times did the system say hold and the stock
    fell" could only be answered going *forward* from when this was added,
    never retroactively - that data was simply never kept before now).

    Written only on a genuinely fresh (non-cached) risk evaluation - see
    `PortfolioRiskService`'s cache, and `assess_position_risk`'s
    `position_signal_snapshot_repo` parameter - so this reflects real,
    distinct evaluations over time, not one row per dashboard reload.
    """

    portfolio_id: int
    ticker: str
    created_at: datetime
    signal: str  # exit_warning | add_candidate | watch | hold (legacy, always present)
    exit_urgency: str | None  # exit_now | reduce | tighten_stop | watch | hold | None (no trade plan yet)
    score: int
    price: float
    r_multiple: float | None
    engine_version: str
