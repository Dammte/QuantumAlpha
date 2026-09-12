"""Historically, this module held the single-ticker buy/wait/avoid checklist
- a weighted, transparent sum of factors (trend/stage/RS/Minervini/ADX/RSI/
OBV/etc.) with a suggested stop-loss and price target. Retired outright in
the 2026-09 reconstruction (Parte 2.3/6): `levels_engine.evaluate_gate` is
now the sole live "is this specific entry good" read (Parte 0, pregunta 3) -
a boolean AND-gate, not a weighted score, so a strong unrelated factor can no
longer paper over one genuinely disqualifying condition. See
`levels_engine.py`'s own docstring for the full reasoning and for exactly
which of this checklist's old factors became hard gate conditions and which
were deliberately left out.

`build_recommendation`/`Recommendation`/`RecommendationFactor`/
`BUY_THRESHOLD`/`AVOID_THRESHOLD` are gone, not merely unused - nothing in
the live API, a job, or a script called `build_recommendation` any more
(confirmed by grep before deletion; `scripts/factor_ablation_study.py`
mirrors the old point values as its own literal constants rather than
importing this module's classes, so its measurement is unaffected). This
file is kept, deliberately small, for two things only:

- Re-exporting `StopAndTarget`/`compute_stop_and_target` (and their
  `ATR_STOP_MULTIPLE`/`REWARD_RISK_RATIO`/`MAX_RESISTANCE_TARGET_DISTANCE`
  constants) from `trade_geometry.py`, where the actual "at what price"
  logic has lived since Fase 3 - kept here so `scripts/
  factor_ablation_study.py`'s existing import keeps working unedited.
- `ENGINE_VERSION`, per Parte 6/17 of the reconstruction brief: the single
  version string marking which decision engine is live. Distinct from
  `levels_engine.GATE_VERSION` (bumped for a gate-condition/threshold change
  specifically, and what `TradePlan`/`PositionSignalSnapshot` actually stamp
  on each persisted row - see `trade_plan_service.py`) - this one marks the
  coarser fact that the levels/triggers gate is the live engine at all,
  superseding every score-based verdict that came before it.
"""

from app.services.trade_geometry import (
    ATR_STOP_MULTIPLE,
    MAX_RESISTANCE_TARGET_DISTANCE,
    REWARD_RISK_RATIO,
    StopAndTarget,
    compute_stop_and_target,
)

__all__ = [
    "ATR_STOP_MULTIPLE",
    "ENGINE_VERSION",
    "MAX_RESISTANCE_TARGET_DISTANCE",
    "REWARD_RISK_RATIO",
    "StopAndTarget",
    "compute_stop_and_target",
]

ENGINE_VERSION = "2026-09-v6-levels"
