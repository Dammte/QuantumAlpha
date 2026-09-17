"""Parte 6 del encargo: modificadores de contexto - condiciones que
acompañan un setup pero NUNCA generan su propia fila en el Radar ("no son
setups"). A diferencia de `setups/*.py` (que devuelven `SetupMatch`), estos
modificadores ajustan el GRADO YA CALCULADO por `levels_engine.compute_grade`
- el mismo grado A/B/C que ya usa "Analizar activo", `/risk` de cartera y el
Radar, no un concepto nuevo y paralelo. `apply_context_modifiers` es la
tercera función de este tipo, junto a `levels_engine.compute_grade`
(geometría + RS/sector/SMA200) y `levels_engine.apply_portfolio_grade_modifiers`
(correlación/cartera llena) - a diferencia de la última, esta no necesita
una cartera concreta (todo lo que usa ya está en `SetupContext`/la lista de
setups del propio ticker), así que corre en `scripts/daily_close.py` junto
al resto del cálculo diario, no en el endpoint de lectura.

Disciplina obligatoria (Parte 6, literal): "el conjunto de TODOS los
modificadores sube como máximo un escalón de grado" - igual que
`compute_grade` ya hace con RS/sector, un único booleano decide si se sube
un escalón, sin importar cuántas de las razones de subida se cumplan a la
vez. De los seis modificadores de la tabla del encargo, solo tres afectan
al grado al alza (squeeze, contracción de ATR, coincidencia de setups -
"puede subir un escalón"/"igual que el squeeze"/"puede subir un solo
escalón"), uno lo limita a la baja (ruptura fallida reciente - "limita el
grado a C"), y dos son puramente informativos (pocket pivot, secado de
volumen - el encargo solo dice "marca .../confirma ..." para esos dos, sin
ningún "puede subir/limita"; no se les inventa un efecto que no piden).
Ver docs/quant_methodology.md §28.x."""

import numpy as np

from app.services import levels_engine as le
from app.services import technical_analysis as ta
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupMatch, SetupStage

SQUEEZE_BB_WINDOW = 20
SQUEEZE_BB_STD = 2.0
SQUEEZE_KC_WINDOW = 20
SQUEEZE_KC_ATR_MULTIPLIER = 1.5

POCKET_PIVOT_LOOKBACK = 10

# Umbrales propios de este modificador - deliberadamente distintos de
# `pullback.PULLBACK_VOLUME_DRYUP_RATIO` (0,75 sobre 3/20 sesiones), que mide
# el secado de UN retroceso concreto. Este es un badge de contexto más
# macro ("¿esta base ya maduró?"), con la ventana y el umbral que el propio
# encargo da literalmente (5/50, 0,7).
CONTEXT_VOLUME_DRYUP_SHORT_WINDOW = 5
CONTEXT_VOLUME_DRYUP_LONG_WINDOW = 50
CONTEXT_VOLUME_DRYUP_RATIO = 0.7

ATR_CONTRACTION_LONG_WINDOW = 50
ATR_CONTRACTION_RATIO = 0.75

FAILED_BREAKOUT_LOOKBACK_SESSIONS = 20
FAILED_BREAKOUT_MAX_RECOVERY_SESSIONS = 3
FAILED_BREAKOUT_PIVOT_LEFT_RIGHT = 3
FAILED_BREAKOUT_TOTAL_LOOKBACK_BARS = 90
# Sin número literal en el encargo para "cuánto hay que romper/perder" - el
# mismo vacío que ya obligó a `classic_patterns.TRIANGLE_FLAT_SLOPE_ATR_FRACTION`
# a existir, resuelto con la misma convención de todo el proyecto ("todos
# los umbrales del sistema van en ATR", ver CLAUDE.md). Necesario incluso
# con pivotes confirmados: verificado con un script que ruido puro
# alrededor de un nivel plano sigue "rompiendo" y "perdiendo" un pivote
# diminuto por pura varianza sin este margen - un pivote de fractal exige
# barras más bajas alrededor para EXISTIR, pero no exige nada sobre cuánto
# hay que superarlo para que romperlo signifique algo.
FAILED_BREAKOUT_MIN_MARGIN_ATR_FRACTION = 0.1

SETUP_COINCIDENCE_MIN_READY = 3


def _detect_squeeze(ctx: SetupContext) -> bool:
    if len(ctx.close) < max(SQUEEZE_BB_WINDOW, SQUEEZE_KC_WINDOW) + 1:
        return False
    _, bb_upper, bb_lower = ta.bollinger_bands(ctx.close, SQUEEZE_BB_WINDOW, SQUEEZE_BB_STD)
    _, kc_upper, kc_lower = ta.keltner_channel(
        ctx.high, ctx.low, ctx.close, SQUEEZE_KC_WINDOW, SQUEEZE_KC_ATR_MULTIPLIER
    )
    bb_up, bb_low, kc_up, kc_low = bb_upper.iloc[-1], bb_lower.iloc[-1], kc_upper.iloc[-1], kc_lower.iloc[-1]
    if np.isnan(bb_up) or np.isnan(bb_low) or np.isnan(kc_up) or np.isnan(kc_low):
        return False
    return bool(bb_up < kc_up and bb_low > kc_low)


def _detect_pocket_pivot(ctx: SetupContext) -> bool:
    if len(ctx.close) < POCKET_PIVOT_LOOKBACK + 2:
        return False
    close = ctx.close.to_numpy()
    volume = ctx.volume.to_numpy()
    if close[-1] <= close[-2]:
        return False  # no es una sesión al alza

    # Las POCKET_PIVOT_LOOKBACK sesiones previas a hoy, más una de apoyo
    # para poder calcular el cambio día a día de la primera de ellas.
    prior_close = close[-(POCKET_PIVOT_LOOKBACK + 2):-1]
    prior_volume = volume[-(POCKET_PIVOT_LOOKBACK + 1):-1]
    change = np.diff(prior_close)
    down_day_volume = prior_volume[change < 0]
    if down_day_volume.size == 0:
        return False
    if volume[-1] <= down_day_volume.max():
        return False

    sma10 = close[-POCKET_PIVOT_LOOKBACK:].mean()
    return bool(close[-1] > sma10)


def _detect_volume_dryup(ctx: SetupContext) -> bool:
    if len(ctx.volume) < CONTEXT_VOLUME_DRYUP_LONG_WINDOW:
        return False
    avg_short = ctx.volume.iloc[-CONTEXT_VOLUME_DRYUP_SHORT_WINDOW:].mean()
    avg_long = ctx.volume.iloc[-CONTEXT_VOLUME_DRYUP_LONG_WINDOW:].mean()
    if avg_long <= 0:
        return False
    return bool(avg_short < CONTEXT_VOLUME_DRYUP_RATIO * avg_long)


def _detect_atr_contraction(ctx: SetupContext) -> bool:
    if ctx.atr14 is None or ctx.atr14 <= 0:
        return False
    atr_series = ctx.atr_series.dropna()
    if len(atr_series) < ATR_CONTRACTION_LONG_WINDOW:
        return False
    avg_atr = float(atr_series.iloc[-ATR_CONTRACTION_LONG_WINDOW:].mean())
    if avg_atr <= 0:
        return False
    return bool(ctx.atr14 / avg_atr < ATR_CONTRACTION_RATIO)


def _detect_recent_failed_breakout(ctx: SetupContext) -> bool:
    """"Nivel" = un máximo local genuino (`ta.indexed_fractal_pivots`, misma
    primitiva que `vcp.py`/`classic_patterns.py`), no el máximo de una
    ventana móvil cruda - verificado con un script antes de fijar esto: un
    máximo móvil se rompe por pura varianza en cualquier serie lateral
    ruidosa (cada barra nueva prueba un "nivel" recién recalculado), dando
    una tasa de falsos positivos altísima en el caso más común (un valor
    lateral sin ninguna ruptura real). Un pivote de fractal exige barras más
    bajas a ambos lados para siquiera existir, así que ya filtra el ruido de
    un solo día antes de que la ruptura/pérdida entre en juego."""
    if ctx.atr14 is None or ctx.atr14 <= 0:
        return False
    close = ctx.close.iloc[-FAILED_BREAKOUT_TOTAL_LOOKBACK_BARS:].reset_index(drop=True)
    n = len(close)
    if n < FAILED_BREAKOUT_LOOKBACK_SESSIONS + 10:
        return False
    highs = ta.indexed_fractal_pivots(
        close, FAILED_BREAKOUT_PIVOT_LEFT_RIGHT, FAILED_BREAKOUT_PIVOT_LEFT_RIGHT, "high"
    )
    if not highs:
        return False
    margin = FAILED_BREAKOUT_MIN_MARGIN_ATR_FRACTION * ctx.atr14
    close_arr = close.to_numpy()
    earliest_break_idx = n - 1 - FAILED_BREAKOUT_LOOKBACK_SESSIONS
    for level_idx, level_price in highs:
        for break_idx in range(level_idx + 1, n):
            if close_arr[break_idx] <= level_price + margin or break_idx < earliest_break_idx:
                continue
            recovery_end = min(break_idx + FAILED_BREAKOUT_MAX_RECOVERY_SESSIONS, n - 1)
            if np.any(close_arr[break_idx + 1:recovery_end + 1] < level_price - margin):
                return True
    return False


def _detect_setup_coincidence(setups: list[SetupMatch]) -> int:
    """Cuenta literal del encargo: "≥ 3 setups distintos en READY a la vez"
    - TRIGGERED no cuenta aquí a propósito (no es lo mismo "tres señales
    todavía esperando su gatillo, todas de acuerdo" que una que ya disparó;
    el encargo nombra la etapa exacta, no "READY o más adelante")."""
    return sum(1 for m in setups if m.stage == SetupStage.READY)


def apply_context_modifiers(
    grade_result: le.GradeResult, ctx: SetupContext, setups: list[SetupMatch]
) -> le.GradeResult:
    """Punto de entrada único de la Parte 6 - `scripts/daily_close.py` la
    llama justo después de `levels_engine.compute_grade`, con la misma lista
    de setups (ya ordenada por `arbitration.order_by_rank`) que ese mismo
    ticker acaba de calcular. No-op si `grade_result.grade` ya es `None`
    (nada que modificar - mismo criterio que `apply_portfolio_grade_modifiers`)."""
    if grade_result.grade is None:
        return grade_result
    grade = grade_result.grade
    reasons = list(grade_result.reasons)

    upgraded = False
    if _detect_squeeze(ctx):
        reasons.append("Squeeze de volatilidad (Bollinger dentro de Keltner) - expansión probable")
        upgraded = True
    if _detect_atr_contraction(ctx):
        reasons.append(f"ATR14 contraído frente a su media de {ATR_CONTRACTION_LONG_WINDOW} sesiones")
        upgraded = True
    ready_count = _detect_setup_coincidence(setups)
    if ready_count >= SETUP_COINCIDENCE_MIN_READY:
        reasons.append(f"Coincidencia de {ready_count} setups en READY a la vez")
        upgraded = True
    if upgraded:
        grade = le.upgrade_grade_one_step(grade)

    if _detect_recent_failed_breakout(ctx):
        reasons.append(
            f"Ruptura fallida reciente (rompió y volvió a perder el nivel en "
            f"<= {FAILED_BREAKOUT_MAX_RECOVERY_SESSIONS} sesiones) - limita a C"
        )
        grade = le.cap_grade_at_most(grade, le.Grade.C)

    # Puramente informativos - el encargo no dice "sube un escalón" para
    # estos dos, a diferencia de los tres de arriba.
    if _detect_pocket_pivot(ctx):
        reasons.append("Pocket pivot - entrada institucional temprana")
    if _detect_volume_dryup(ctx):
        reasons.append(
            f"Volumen secándose ({CONTEXT_VOLUME_DRYUP_SHORT_WINDOW} sesiones < "
            f"{CONTEXT_VOLUME_DRYUP_RATIO * 100:.0f}% de {CONTEXT_VOLUME_DRYUP_LONG_WINDOW}) - base madura"
        )

    return le.GradeResult(grade=grade, reasons=reasons, distance_atr=grade_result.distance_atr)
