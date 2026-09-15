"""Reconstruction (2026-09), Parte 19: every configurable trading parameter
in one module, with the value the reconstruction brief specified and a short
sourcing comment - so a decision can always be traced to "why this number",
not just "where in the code it lives". Exposed read-only via
`GET /system/params` (see `api/v1/endpoints/system.py`) so the UI can show
which config produced a given decision, per Parte 19.

Plain module-level constants, not a `Settings`/pydantic model - these are
trading-strategy parameters an owner tunes deliberately (same
"cualquier peso nuevo... necesita el visto bueno del propietario" discipline
CLAUDE.md applies to `recommendation_engine.py`'s old weights), not
deployment config that varies by environment (`app/core/config.py` already
covers that: database URL, CORS, API keys). Every consumer imports the exact
names it needs from here rather than redefining its own local constant, so
there is exactly one place to look up or change a number - `trade_manager.py`,
`portfolio_construction_service.py`, `trade_geometry.py` and `levels_engine.py`
re-export the subset they use unchanged, so existing call sites and tests
that reference e.g. `trade_manager.CHANDELIER_WINDOW` keep working.

Recalibration note (Parte 3.2/20): several of these values replace an
earlier, less risk-aware first pass (a flat 2.5 ATR stop, a 22-bar/
2.5-3.5 ATR Chandelier, a 0.8 correlation threshold) that predates this
reconstruction. The new values are the reconstruction brief's own explicit,
numbered recalibration table - not independently re-derived here - and
Parte 20 already flags them as *coherent but not yet ablation-calibrated*:
`scripts/factor_ablation_study.py`/`chandelier_calibration_study.py` should
revisit them once a real trigger-based sample exists (Fase 8), the same
"medir antes de confiar" discipline every other weight in this system
follows.
"""

# --- Sizing (Parte 7/19) ------------------------------------------------------

RISK_PER_TRADE_PCT = 0.01  # 1% del capital por operación
MAX_POSITION_PCT = 0.15  # tope por posición
MAX_AGGREGATE_RISK_PCT = 0.06  # riesgo total si saltan todos los stops
MAX_OPEN_POSITIONS = 10  # aviso, no bloqueo (Parte 10.3)
MIN_POSITION_USD = 80.0  # por debajo, el coste se come el trade
MIN_POSITION_FOR_SCALING = 150.0  # por debajo, no se escalona (Parte 8)
TRANSACTION_COST_PCT = 0.001  # por lado (0.2% ida y vuelta)

# --- Stop/target geometry (Parte 7) -------------------------------------------

STOP_ATR_CEILING = 2.0  # techo duro del stop en ATR, cualquiera sea el nivel
RISK_CEILING_ATR_MULTIPLE = 2.5  # techo de riesgo = 2.5 x atr_pct
RISK_CEILING_MIN_PCT = 0.020
RISK_CEILING_MAX_PCT = 0.070
MIN_RISK_REWARD_NET = 1.5

# --- Scale-out ladder (Parte 8) ------------------------------------------------

SCALE_OUT_1R_FRACTION = 0.33
SCALE_OUT_2R_FRACTION = 0.33
LAST_TRANCHE_TIME_STOP_BARS = 15

# --- Chandelier trailing stop (Parte 3.2/9) ------------------------------------

CHANDELIER_WINDOW = 10
CHANDELIER_MULT_BY_VOL = {"baja": 1.75, "normal": 2.0, "elevada": 2.25, "alta": 2.5}
CHANDELIER_PROFIT_LOCK_R = 1.5
CHANDELIER_PROFIT_LOCK_MULT = 1.5

# --- Gate / triggers (Parte 5/6/9) ---------------------------------------------

TRIGGER_MAX_DISTANCE_ATR = 2.0
BREAKOUT_MIN_REL_VOLUME = 1.2
STALL_MIN_BARS = 5
STALL_MAX_BARS = 15
HIGH_CORRELATION_THRESHOLD = 0.7
