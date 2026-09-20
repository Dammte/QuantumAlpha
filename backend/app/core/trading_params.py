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

# Auditoria del Radar, bloque H: ya no un techo que MUEVE el stop dentro de
# `compute_entry_geometry` (bloque H2, paso 4: "el stop no se mueve para
# caber; el tamaño sí") - `_stop_cascade` ya no lo consulta. Sobrevive como
# la única fuente de verdad del multiplicador fijo que
# `compute_stop_and_target` (la función simple, todavía usada por el
# `stop_and_target` sencillo del Radar) sigue necesitando - sustituye al
# `ATR_STOP_MULTIPLE=2,5` local de `trade_geometry.py`, que contradecía a
# este mismo número sin que nadie lo hubiera notado.
STOP_ATR_CEILING = 2.0
RISK_CEILING_ATR_MULTIPLE = 2.5  # techo de riesgo INFORMATIVO = 2.5 x atr_pct (bloque H2, paso 4.1)
RISK_CEILING_MIN_PCT = 0.020
RISK_CEILING_MAX_PCT = 0.070
MIN_RISK_REWARD_NET = 1.5

# Auditoria del Radar, bloque H2: perfil de volatilidad - "la diferencia no
# está en el porcentaje que tolero, está en qué nivel del gráfico es lo
# bastante robusto para ese valor" (literal). Umbrales de `atr_pct`
# (ATR14/precio) que separan los cuatro perfiles - por encima de
# `VOLATILE_MAX` es "extremo".
VOLATILITY_PROFILE_CALM_MAX_ATR_PCT = 0.02
VOLATILITY_PROFILE_NORMAL_MAX_ATR_PCT = 0.04
VOLATILITY_PROFILE_VOLATILE_MAX_ATR_PCT = 0.07

# El colchón bajo el nivel escala con el perfil, no es fijo (bloque H2, paso
# 3) - sustituye `LEVEL_STOP_CUSHION_ATR`/`MA_STOP_CUSHION_ATR` (ambos
# fijos en 0,3/0,4 sin importar la volatilidad del valor). "Extremo" usa el
# mismo colchón que "volátil" - a esas alturas el anclaje ya es
# estructuralmente más ancho (swing semanal/MA30 semanal), no hace falta
# que el colchón siga creciendo también.
STOP_CUSHION_ATR_CALM = 0.25
STOP_CUSHION_ATR_NORMAL = 0.35
STOP_CUSHION_ATR_VOLATILE = 0.5

# Un stop más cerca que esto es ruido disfrazado de nivel (bloque H2, paso
# 4.3, literal: "un mínimo de ayer en un valor con 8% de ATR lo perfora el
# ruido de una mañana cualquiera") - sube al siguiente anclaje de la
# cascada en vez de aceptarlo.
STOP_MIN_DISTANCE_ATR = 0.8

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

# --- Score compuesto del Radar (Auditoria del Radar, bloque E2) ----------------
# "necesito saber por qué un valor está el primero" - pesos explícitos y
# nombrados, no una tupla lexicográfica opaca. `setups/scoring.py` es el
# único consumidor - los cinco pesos suman 100.

RADAR_SCORE_WEIGHT_SETUP_QUALITY = 30  # grado del gate x confianza medida del setup
RADAR_SCORE_WEIGHT_RELATIVE_STRENGTH = 25  # RS del valor + percentil RS de su sector
RADAR_SCORE_WEIGHT_TRIGGER_PROXIMITY = 20  # distancia al gatillo en ATR, invertida
RADAR_SCORE_WEIGHT_GEOMETRY_QUALITY = 15  # R:R neto + calidad del anclaje del stop
RADAR_SCORE_WEIGHT_VOLUME_CONFIRMATION = 10  # volumen de hoy vs su propia media, según la etapa

# Calidad del setup: el grado A/B/C dentro de [0, 1], multiplicado por la
# confianza medida (Parte 10.3) - "si es unvalidated, penaliza, no premia"
# (literal): un multiplicador de 0.5, no 1.0, para lo nunca medido.
RADAR_SCORE_GRADE_POINTS = {"A": 1.0, "B": 0.6, "C": 0.2}
RADAR_SCORE_CONFIDENCE_MULTIPLIER = {"measured": 1.0, "thin": 0.75, "unvalidated": 0.5}

# Fuerza relativa: RS del propio valor pesa más que el de su sector (60/40) -
# el sector es contexto que suma, no el protagonista.
RADAR_SCORE_TICKER_RS_WEIGHT = 0.6
RADAR_SCORE_SECTOR_RS_WEIGHT = 0.4

# Proximidad al gatillo: a esta distancia (en ATR) o más, cero puntos; a 0
# ATR, el máximo.
RADAR_SCORE_PROXIMITY_ATR_SCALE = 2.0

# Calidad de la geometría: R:R neto normalizado entre el mínimo viable
# (MIN_RISK_REWARD_NET, de la sección de arriba) y este techo, combinado con
# un plus/menos según el anclaje del stop sea un nivel real (ruptura/soporte)
# o una media móvil (más blando, pero nunca "sin anclaje" - `_stop_cascade`
# no deja pasar una geometría viable sin alguno de los dos).
RADAR_SCORE_RISK_REWARD_CEILING = 4.0
RADAR_SCORE_REAL_LEVEL_ANCHOR_SCORE = 1.0
RADAR_SCORE_MA_ANCHOR_SCORE = 0.6
RADAR_SCORE_RISK_REWARD_SUBWEIGHT = 0.7
RADAR_SCORE_ANCHOR_SUBWEIGHT = 0.3

# Confirmación de volumen: dirección distinta según la etapa del setup - más
# volumen es bueno en el disparo (confirma la ruptura), menos volumen es
# bueno mientras la base todavía se contrae (acumulación silenciosa).
RADAR_SCORE_VOLUME_TRIGGERED_FLOOR = 0.5
RADAR_SCORE_VOLUME_TRIGGERED_CEILING = 2.5
RADAR_SCORE_VOLUME_FORMING_FLOOR = 0.5
RADAR_SCORE_VOLUME_FORMING_CEILING = 1.5

# Penalizaciones (bloque E2, literal) - restan del score ya calculado, nunca
# mezcladas dentro de un componente (el desglose las muestra aparte).
RADAR_SCORE_EARNINGS_PENALTY_SHORT = 15  # solo en horizonte corto - "es mi horizonte entero"
RADAR_SCORE_EARNINGS_WITHIN_SESSIONS = 10  # próximas 10 sesiones de calendario
RADAR_SCORE_HIGH_ATR_PENALTY = 10
RADAR_SCORE_HIGH_ATR_PERCENTILE = 90  # percentil de atr_pct dentro del universo del día

# Umbral de convicción para marcar el primero de `short_term` como "principal
# a entrar" (bloque E4) - "si el mejor candidato del día no llega al umbral,
# ninguno es primario" (literal). Ajustable si un estudio de replay futuro
# dice otro número más calibrado.
RADAR_PRIMARY_SCORE_THRESHOLD = 70

# Auditoria del Radar, bloque G/10: "si el régimen es bajista... el umbral
# del primario sube a 80" (literal) - "el contexto tiene que tener
# consecuencia, no ser decorado". Ver `market_regime_service.py`.
RADAR_PRIMARY_SCORE_THRESHOLD_BEARISH = 80

# "A punto de disparar" (bloque G/10): valores a menos de esta distancia (en
# ATR) de su disparador que no están todavía en ninguna de las dos listas.
RADAR_ABOUT_TO_TRIGGER_MAX_DISTANCE_ATR = 0.3
RADAR_ABOUT_TO_TRIGGER_MAX_ITEMS = 5

# "Rompiendo por abajo" (bloque G/10): mismo tope de 5, ver
# `ticker_daily_state_builder._broken_levels` para la ventana de sesiones.
RADAR_BREAKING_DOWN_MAX_ITEMS = 5

HIGH_CORRELATION_THRESHOLD = 0.7

# --- Rotación de cartera (Auditoria del Radar, bloque 9) -----------------------
# "rotación contra cartera con topes" (literal) - como mucho esta cantidad de
# sugerencias de swap por llamada (día, en la práctica: `/today` se lee una
# vez al día) - "aviso, no automatización", mismo espíritu que
# `MAX_OPEN_POSITIONS` de arriba (Parte 10.3: un aviso que no bloquea nada).
ROTATION_MAX_SUGGESTIONS_PER_DAY = 2
