"""Reconstruction (2026-09), Fase 3, reescrito en la Sexta auditoría (texto
literal completo, ver docs/quant_methodology.md §27.6): el gate booleano de
elegibilidad de la Parte 6.2 - "is this specific entry good" (Parte 0,
pregunta 3). See `trade_geometry.py`'s docstring for the "at what price"
half of the same Fase.

Por qué un gate y no un score: una suma ponderada puede aprobar con una
fuerza relativa alta compensando una entrada fea y sobreextendida, o
suspender por un punto un setup por lo demás limpio - exactamente la opacidad
"no veo por qué" que el checklist retirado tenía. Cada condición de abajo es
un AND estricto: todas deben cumplirse o el gate simplemente no aprueba, y
cuál falló es siempre visible.

**Los 5 criterios eliminatorios son literalmente los de la Parte 6.2 - ya no
los 6 de la reconstrucción anterior a esta auditoría (tendencia/parabólico/
sobrecompra/OBV/par rápido/R:R), que eran una aproximación razonada escrita
sin el texto literal en contexto:**
- `liquidity_ok`: volumen-dólar 20d >= $20M y precio >= $5 - el mismo suelo
  que ya filtra el universo dinámico mensual (`dynamic_universe_service.
  passes_liquidity_floor`), evaluado aquí a diario por ticker.
- `data_quality_ok`: >= 250 barras (`MIN_BARS_REQUIRED`, Parte 3.2) - el
  propio llamador ya garantiza esto antes de construir cualquier lectura, así
  que en la práctica es casi siempre `True`; se mantiene como criterio
  explícito y persistido, no implícito, tal como pide la Parte 6.2.
- `weekly_not_stage4`: Weinstein semanal (la MA30 semanal real de
  `multi_timeframe.py`, no una proxy diaria) no en Fase 4 - `unknown` (menos
  de ~3,85 años de historial semanal) tampoco pasa, literal.
- `no_fast_bearish_cross`: sin veto bajista del par rápido EMA21/55
  (`technical_analysis.detect_fast_pair_bearish_veto`), sin cambios respecto
  a la reconstrucción anterior - éste sí coincidía con el literal.
- `no_event_risk`: sin resultados conocidos en los próximos 10 días
  naturales (aproximación a "10 sesiones" - ver `EVENT_RISK_WINDOW_DAYS`).

**Qué se retira del gate, y por qué no es una pérdida silenciosa:** el
"parabólico"/"sobrecompra extrema"/"tendencia alcista o Fase 2" del gate
anterior no tienen respaldo en el texto literal como criterios de
*elegibilidad* - la extensión parabólica ya vive en `exit_engine.py` (REDUCE,
Parte 9) para posiciones abiertas, y la dirección de la tendencia la exige
la propia cascada de la geometría (`trade_geometry._stop_cascade`, Parte 7:
los peldaños de retroceso a EMA21/continuación sobre EMA55 sólo aplican en
`TrendState.UPTREND`) - un ticker sin tendencia alcista simplemente no
produce un disparador viable, sin necesidad de un criterio de gate aparte.
"Sin divergencia bajista de volumen (OBV)" no aparece en ninguna de las 20
partes como criterio de entrada - `technical_analysis.obv_divergence` queda
sin llamador tras este cambio, marcado para revisión de código muerto en un
sub-paso posterior, no borrado en el mismo commit que reescribe el gate. El
R:R mínimo (antes parte del gate, sobre `compute_stop_and_target`) se separa
hacia la *viabilidad del disparador* (Parte 5.2/7.4) - ya vive ahí, en
`trade_geometry.compute_entry_geometry`'s propio chequeo de
`MIN_RISK_REWARD_NET`, no se duplica aquí.

RS Rating y Minervini's 8/8 siguen sin ser gates duros, por la misma razón
que antes: un setup genuinamente bueno en un nombre que todavía no se ha
ganado una fuerza relativa alta es justo el tipo de entrada que esta
reconstrucción no quiere excluir por construcción.

`GATE_VERSION` sigue la misma disciplina de versionado que
`recommendation_engine.ENGINE_VERSION` (se graba en cada resultado
persistido, para poder atribuir un veredicto pasado a la lógica exacta que
lo produjo).
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd

from app.core.trading_params import HIGH_CORRELATION_THRESHOLD, MAX_OPEN_POSITIONS
from app.services.technical_analysis import (
    Level,
    PriceLevel,
    Stage,
    TrendState,
    classify_stage,
    classify_trend,
    detect_fast_pair_bearish_veto,
)
from app.services.trade_geometry import (
    EntryTrigger,
    StopAndTarget,
    TradeGeometry,
    compute_entry_geometry,
    compute_entry_trigger,
    compute_stop_and_target,
)

# Bumped whenever a gate condition or threshold changes materially - same
# discipline as recommendation_engine.ENGINE_VERSION, its own separate
# version string (see that module's docstring for why this isn't the same
# constant). v2 marks the Sexta auditoría's switch to the 5 literal
# eligibility criteria (Parte 6.2), replacing v1's 6-condition approximation.
# v3 (Auditoria del Radar, bloque H2): `entry_geometry` cambia de verdad qué
# se clasifica como viable - ya no rechaza por techo de riesgo adaptativo ni
# recorta el stop al techo duro de 2.0 ATR ("el stop no se mueve para caber;
# el tamaño sí"), y el peldaño de ruptura pasa de estructuralmente
# inalcanzable (comparaba el precio contra un nivel que, por construcción,
# seguía por encima) a uno real basado en `Level`/`BROKEN_CONFIRMED`. Un
# `GateResult`/`TradePlan` persistido con `gate_version="2026-09-levels-v2"`
# no es comparable a uno v3 para el mismo ticker/fecha.
GATE_VERSION = "2026-09-levels-v3"

# Parte 6.2: "en las próximas 10 sesiones" - aproximado en días naturales
# (~2 semanas de calendario para 10 sesiones de trading), mismo criterio que
# `trigger_performance_service.TAKEN_WINDOW_DAYS` ya usa para "10 días" sin
# un calendario de mercado exacto disponible en este nivel.
EVENT_RISK_WINDOW_DAYS = 14


@dataclass(frozen=True, slots=True)
class GateCondition:
    label: str
    passed: bool


@dataclass(frozen=True, slots=True)
class Eligibility:
    """Los 5 criterios eliminatorios literales de la Parte 6.2 - si uno
    falla, no hay disparador, sin importar cuán buena se vea la geometría.
    Cada uno se persiste individualmente (vía `GateResult.conditions`) para
    poder medir después cuál filtra y si filtra bien - el mismo motivo por
    el que el texto pide guardarlos en `eligibility` JSONB."""

    liquidity_ok: bool
    data_quality_ok: bool
    weekly_not_stage4: bool
    no_fast_bearish_cross: bool
    no_event_risk: bool

    @property
    def passes(self) -> bool:
        return (
            self.liquidity_ok
            and self.data_quality_ok
            and self.weekly_not_stage4
            and self.no_fast_bearish_cross
            and self.no_event_risk
        )

    @property
    def failing(self) -> list[str]:
        labels = {
            "liquidity_ok": "Liquidez insuficiente (volumen-dólar 20d < $20M o precio < $5)",
            "data_quality_ok": "Datos insuficientes (menos de 250 barras)",
            "weekly_not_stage4": "Semanal en Fase 4 de Weinstein (o desconocida)",
            "no_fast_bearish_cross": "Cruce bajista del par rápido EMA21/55 confirmado o proyectado",
            "no_event_risk": "Riesgo de evento (resultados conocidos) en los próximos 10 días",
        }
        return [label for field, label in labels.items() if not getattr(self, field)]


@dataclass(frozen=True, slots=True)
class GateResult:
    """`passes` is the single boolean the Radar/Screener (Fase 6) filters on;
    `conditions` is what makes that answer auditable instead of a black box -
    every condition evaluated, not just the one(s) that failed. `entry_trigger`
    can be `None` even when `passes` is `True`: a clean, tradeable setup with
    no support/resistance level close enough to define an imminent watch
    price yet is a real, if less actionable-today, state - see
    `trade_geometry.compute_entry_trigger`.

    `entry_geometry` (Parte 7, added once the literal brief text was back in
    hand) is the richer stop-cascade/adaptive-risk-ceiling/cost-net-target
    read from `trade_geometry.compute_entry_geometry` - `None` whenever the
    caller doesn't pass `ema21`/`ema55` (e.g. `replay_gate_at`'s point-in-time
    backtest replay, which doesn't compute them). `levels` (Auditoria del
    Radar, bloque H2) is optional too - without it the stop cascade's
    ruptura-confirmada and mínimo-de-20-sesiones rungs simply don't apply,
    same honest degradation as missing EMAs, never a fabricated anchor.
    Deliberately still no
    sizing on it (`shares_for_risk_budget`/`position_value`/`pct_of_portfolio`
    are always `None` here) - a ticker's own gate isn't scoped to any one
    portfolio's capital; call `trade_geometry.size_position` separately once
    a specific portfolio is in view. `stop_and_target` (the original, simpler
    read) is kept alongside, unchanged, for every existing consumer.
    `eligibility` is the Parte 6.2 read the 5 `conditions` above summarize -
    kept as its own typed object (not just the flattened list) so a caller
    can check a single criterion by name without matching on label text."""

    passes: bool
    conditions: list[GateCondition]
    entry_trigger: EntryTrigger | None
    stop_and_target: StopAndTarget | None
    eligibility: Eligibility
    entry_geometry: TradeGeometry | None = None


def _no_event_risk(
    next_earnings_date: date | None, as_of: date, window_days: int = EVENT_RISK_WINDOW_DAYS
) -> bool:
    """`None` significa "no se conoce ninguna fecha de resultados próxima" -
    una comprobación que tuvo éxito y no encontró nada, no una comprobación
    fallida - así que cuenta como sin riesgo, no como "no se pudo
    comprobar" (que sí cuenta como no cumplido, según la Parte 12.3: "si no
    se puede comprobar, cuenta como no cumplido" se refiere a que la propia
    llamada falle, no al resultado normal de "no hay nada programado
    todavía", que es el caso la mayoría de los días del año para la mayoría
    de los tickers). Una fecha ya pasada (calendario desactualizado) tampoco
    es riesgo - solo lo es una fecha futura dentro de la ventana."""
    if next_earnings_date is None:
        return True
    days_until = (next_earnings_date - as_of).days
    return not (0 <= days_until <= window_days)


def evaluate_gate(
    price: float,
    trend: TrendState,
    atr14: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    weekly_stage: Stage | None,
    liquidity_ok: bool,
    data_quality_ok: bool = True,
    fast_pair_bearish_signal: str | None = None,
    next_earnings_date: date | None = None,
    as_of: date | None = None,
    ema21: float | None = None,
    ema55: float | None = None,
    levels: list[Level] | None = None,
) -> GateResult:
    conditions: list[GateCondition] = []

    def add(label: str, passed: bool) -> None:
        conditions.append(GateCondition(label=label, passed=passed))

    weekly_not_stage4 = weekly_stage is not None and weekly_stage != Stage.STAGE_4
    no_fast_bearish_cross = fast_pair_bearish_signal is None
    no_event_risk = _no_event_risk(next_earnings_date, as_of if as_of is not None else date.today())

    eligibility = Eligibility(
        liquidity_ok=liquidity_ok,
        data_quality_ok=data_quality_ok,
        weekly_not_stage4=weekly_not_stage4,
        no_fast_bearish_cross=no_fast_bearish_cross,
        no_event_risk=no_event_risk,
    )

    add("Liquidez suficiente (volumen-dólar 20d >= $20M y precio >= $5)", liquidity_ok)
    add("Datos suficientes (>= 250 barras)", data_quality_ok)
    add("Semanal no en Fase 4 de Weinstein", weekly_not_stage4)
    add("Sin cruce bajista del par rápido (EMA21/55)", no_fast_bearish_cross)
    add("Sin riesgo de evento en los próximos 10 días", no_event_risk)

    stop_and_target = compute_stop_and_target(price, atr14, nearest_support, nearest_resistance)
    entry_trigger = compute_entry_trigger(price, nearest_support, nearest_resistance)

    # Parte 7: only computed when the caller has real EMA21/55 reads to give
    # it (e.g. not `replay_gate_at`'s point-in-time backtest replay, which
    # doesn't compute them - see that function's own docstring) - `None`
    # otherwise, never a guess built from `stop_and_target`'s simpler read.
    entry_geometry = None
    if ema21 is not None and ema55 is not None:
        entry_geometry = compute_entry_geometry(
            price, atr14, nearest_support, nearest_resistance, ema21, ema55, trend, levels
        )

    return GateResult(
        passes=eligibility.passes,
        conditions=conditions,
        entry_trigger=entry_trigger,
        stop_and_target=stop_and_target,
        eligibility=eligibility,
        entry_geometry=entry_geometry,
    )


def replay_gate_at(
    i: int,
    close: pd.Series,
    sma20: pd.Series,
    sma50: pd.Series,
    sma150: pd.Series,
    sma200: pd.Series,
    rsi14: pd.Series,
    adx14: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    atr14: pd.Series,
    volume: pd.Series | None = None,
) -> GateResult | None:
    """Reconstruction (2026-09), Fase 4, actualizado en la Sexta auditoría
    para los 5 criterios literales de la Parte 6.2: point-in-time replay of
    `evaluate_gate` at historical bar `i`, using only values knowable at that
    bar - either a scalar reading at `i`, or (for the functions that need a
    lookback window - `classify_stage`, `detect_fast_pair_bearish_veto`) a
    slice `[:i+1]`, still using no information beyond bar `i`.

    Replaces `walk_forward_backtest.replay_recommendation_at` (retired -
    see docs/quant_methodology.md) as `backtest_engine.find_triple_barrier_entries`'s
    "what would the system have proposed here" replay, now against the gate
    instead of the old weighted checklist. Point-in-time simplifications,
    documented and accepted rather than silently approximated:

    - **RS Rating**: needs a cross-sectional universe snapshot unavailable at
      arbitrary past dates. Actually moot here, not just worked around - RS
      Rating was deliberately never wired into the gate as a hard condition
      in the first place (see this module's own docstring), so there is
      nothing to omit.
    - **Nearest support/resistance**: the pivot scan is O(n) per call:
      re-running it at every historical bar in a backtest is prohibitively
      expensive. Passed as `None` to `evaluate_gate`, which (via
      `trade_geometry.compute_stop_and_target`) falls back gracefully to the
      ATR-ceiling stop and the fixed 2:1 target - the same graceful
      degradation the live gate already relies on for a ticker with no
      nearby level at all, not a special case invented for this replay.
    - **`weekly_not_stage4`**: a real weekly-bar Stage read
      (`multi_timeframe.py`, MA30 semanal genuina) would need re-resampling
      to weekly at every single historical bar - the same "prohibitively
      expensive per bar" problem as the pivot scan above, at an even worse
      multiplier (a full weekly resample, not a single pivot pass).
      `classify_stage`'s own daily-bar proxy (SMA150) stands in here - the
      same "proxy when weekly bars aren't available" role its own docstring
      already documents, just applied for cost instead of unavailability.
      Precomputing a real point-in-time weekly-Stage series once per ticker
      (Parte 8, reorienting the backtest properly) is future work, not a
      silent approximation - this paragraph is that disclosure.
    - **`liquidity_ok`/`no_event_risk`**: no historical dollar-volume-in-USD
      series or historical earnings-calendar data is threaded through this
      replay - both default to "pass" here (`True`), since a backtest that
      can't know either honestly should not fabricate a rejection for them.
      Only `weekly_not_stage4`/`no_fast_bearish_cross` are replayed for real.

    Returns `None` when there isn't yet enough history for the trend read
    itself (mirrors the retired function's own `None` case)."""
    price = close.iloc[i]
    s20, s50, s200 = sma20.iloc[i], sma50.iloc[i], sma200.iloc[i]
    if pd.isna(s20) or pd.isna(s50) or pd.isna(s200):
        return None

    trend = classify_trend(price, s20, s50, s200)
    s150 = sma150.iloc[i]
    daily_stage_proxy = classify_stage(price, sma150.iloc[: i + 1]) if not pd.isna(s150) else None

    atr_t = atr14.iloc[i]
    has_atr = not pd.isna(atr_t) and atr_t != 0

    return evaluate_gate(
        price=float(price),
        trend=trend,
        atr14=float(atr_t) if has_atr else None,
        nearest_support=None,
        nearest_resistance=None,
        weekly_stage=daily_stage_proxy,
        liquidity_ok=True,
        fast_pair_bearish_signal=detect_fast_pair_bearish_veto(close.iloc[: i + 1]),
        next_earnings_date=None,
    )


# --- Parte 5.3 (encargo literal, sexta auditoría): grados A/B/C -------------
#
# "No inventes una probabilidad; no la tienes. El grado es una lectura de la
# calidad de la oportunidad" - geometría, no un score de compra. Se calcula
# sobre un disparador (`trade_geometry.EntryTrigger` + `TradeGeometry`) ya
# viable - un gate que aprueba sin una entrada geométricamente viable no
# tiene grado, porque no hay ningún disparador que gradar (ver
# `levels_engine.py`'s propio docstring sobre por qué el R:R se separó del
# gate hacia la viabilidad de la geometría).


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"


_GRADE_ORDER = (Grade.C, Grade.B, Grade.A)  # peor a mejor


def upgrade_grade_one_step(grade: Grade) -> Grade:
    """Pública desde que `setups/context_modifiers.py` (Parte 6) también la
    necesita para su propio tope de un escalón - antes de eso, un detalle
    interno de este módulo (mismo criterio de "no repitas ninguna pieza" que
    ya promovió `technical_analysis.indexed_fractal_pivots`)."""
    idx = _GRADE_ORDER.index(grade)
    return _GRADE_ORDER[min(idx + 1, len(_GRADE_ORDER) - 1)]


def cap_grade_at_most(grade: Grade, ceiling: Grade) -> Grade:
    """Pública por el mismo motivo que `upgrade_grade_one_step`."""
    return ceiling if _GRADE_ORDER.index(grade) > _GRADE_ORDER.index(ceiling) else grade


# Umbrales de grado base, literales (Parte 5.3).
GRADE_A_MAX_DISTANCE_ATR = 1.0
GRADE_A_MAX_RISK_ATR = 1.5
GRADE_A_MIN_REWARD_RISK_NET = 2.5
GRADE_A_MIN_REL_VOLUME = 1.2
GRADE_B_MAX_DISTANCE_ATR = 1.5
GRADE_B_MAX_RISK_ATR = 2.0
GRADE_B_MIN_REWARD_RISK_NET = 2.0

# Umbrales de los modificadores, literales (Parte 5.3).
RS_PERCENTILE_UPGRADE_THRESHOLD = 70
RS_PERCENTILE_DOWNGRADE_THRESHOLD = 40
SECTOR_PERCENTILE_UPGRADE_THRESHOLD = 70
SECTOR_PERCENTILE_DOWNGRADE_THRESHOLD = 30


@dataclass(frozen=True, slots=True)
class GradeResult:
    """`grade=None` es el caso literal "si no llega a C, el disparador no se
    emite" - preferible una lista vacía a una lista de trades malos. `reasons`
    documenta cada modificador aplicado (no el grado base en sí, que ya se ve
    en los propios números de la geometría) - Parte 5.3: "cada uno registra
    su motivo en grade_reasons".

    `distance_atr`, añadido para la Parte 9.1 de la biblioteca de setups del
    Radar (`docs/quant_methodology.md` §28.x): la MISMA distancia que ya
    calculaba `compute_grade` internamente para el grado base - expuesta
    aquí en vez de recalculada aparte, para que la clave de ordenación del
    Radar ("distancia al gatillo en ATR, ascendente") no repita esa cuenta.
    `None` (no `inf`, que no serializa a JSON) cuando no hay ATR14 válido."""

    grade: Grade | None
    reasons: list[str]
    distance_atr: float | None = None


def _base_grade(
    distance_atr: float, risk_atr: float | None, reward_risk_net: float | None,
    relative_volume: float | None, weekly_bullish: bool,
) -> Grade:
    if (
        distance_atr <= GRADE_A_MAX_DISTANCE_ATR
        and risk_atr is not None and risk_atr <= GRADE_A_MAX_RISK_ATR
        and reward_risk_net is not None and reward_risk_net >= GRADE_A_MIN_REWARD_RISK_NET
        and relative_volume is not None and relative_volume >= GRADE_A_MIN_REL_VOLUME
        and weekly_bullish
    ):
        return Grade.A
    if (
        distance_atr <= GRADE_B_MAX_DISTANCE_ATR
        and risk_atr is not None and risk_atr <= GRADE_B_MAX_RISK_ATR
        and reward_risk_net is not None and reward_risk_net >= GRADE_B_MIN_REWARD_RISK_NET
    ):
        return Grade.B
    # "Pasa el mínimo de viabilidad pero con concesiones" - el llamador ya
    # garantiza `geometry.viable` antes de llegar aquí (ver `compute_grade`),
    # así que todo lo que no llega a A/B es C, nunca "sin grado" en esta capa.
    return Grade.C


def compute_grade(
    price: float,
    atr14: float | None,
    entry_trigger: EntryTrigger,
    geometry: TradeGeometry,
    weekly_bullish: bool,
    relative_volume: float | None = None,
    rs_percentile: int | None = None,
    sma200: float | None = None,
    sector_rs_percentile: int | None = None,
) -> GradeResult:
    """El grado base por geometría, más los modificadores de fuerza
    relativa/SMA200/sector (Parte 5.3) - sin cartera todavía, ver
    `apply_portfolio_grade_modifiers` para correlación/cartera llena, que sí
    la necesitan. `None` (sin grado) si la geometría no es viable - no hay
    ningún disparador real que gradar."""
    if not geometry.viable:
        return GradeResult(grade=None, reasons=[geometry.rejection_reason or "geometría no viable"])

    # "Distancia" (Parte 5.3) es al propio nivel del disparador, en ATR - un
    # número distinto de `geometry.risk_atr` (la distancia al *stop*, ver
    # `trade_geometry.TradeGeometry`).
    distance_atr = (
        abs(price - entry_trigger.trigger_price) / atr14 if atr14 and atr14 > 0 else float("inf")
    )

    grade = _base_grade(
        distance_atr, geometry.risk_atr, geometry.risk_reward_net, relative_volume, weekly_bullish
    )
    reasons: list[str] = []

    upgraded = False
    if rs_percentile is not None and rs_percentile >= RS_PERCENTILE_UPGRADE_THRESHOLD:
        reasons.append(f"Fuerza relativa alta (percentil {rs_percentile} >= {RS_PERCENTILE_UPGRADE_THRESHOLD})")
        upgraded = True
    if sector_rs_percentile is not None and sector_rs_percentile >= SECTOR_PERCENTILE_UPGRADE_THRESHOLD:
        reasons.append(
            f"Sector fuerte (percentil {sector_rs_percentile} >= {SECTOR_PERCENTILE_UPGRADE_THRESHOLD})"
        )
        upgraded = True
    # "Un grado nunca sube más de un escalón por el conjunto de modificadores"
    # - un único paso, sin importar cuántas razones de subida se cumplan a
    # la vez.
    if upgraded:
        grade = upgrade_grade_one_step(grade)

    capped_to_b = False
    if rs_percentile is not None and rs_percentile < RS_PERCENTILE_DOWNGRADE_THRESHOLD:
        reasons.append(
            f"Fuerza relativa baja (percentil {rs_percentile} < {RS_PERCENTILE_DOWNGRADE_THRESHOLD}) - limita a B"
        )
        capped_to_b = True
    if sma200 is not None and price < sma200:
        reasons.append("Precio bajo la SMA200 - limita a B")
        capped_to_b = True
    if sector_rs_percentile is not None and sector_rs_percentile <= SECTOR_PERCENTILE_DOWNGRADE_THRESHOLD:
        reasons.append(
            f"Sector débil (percentil {sector_rs_percentile} <= {SECTOR_PERCENTILE_DOWNGRADE_THRESHOLD}) - "
            "limita a B"
        )
        capped_to_b = True
    if capped_to_b:
        grade = cap_grade_at_most(grade, Grade.B)

    return GradeResult(
        grade=grade, reasons=reasons, distance_atr=distance_atr if atr14 and atr14 > 0 else None
    )


def apply_portfolio_grade_modifiers(
    grade_result: GradeResult,
    max_correlation_with_open_position: float | None = None,
    open_positions_count: int = 0,
    max_open_positions: int = MAX_OPEN_POSITIONS,
    high_correlation_threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> GradeResult:
    """Los dos modificadores de la Parte 5.3 que necesitan una cartera
    *específica* - correlación con una posición abierta y tope de posiciones
    - separados de `compute_grade` por el mismo motivo que
    `trade_geometry.size_position` está separado de `compute_entry_geometry`:
    un disparador del universo no pertenece a ninguna cartera en particular.
    No-op si `grade_result.grade` ya es `None` (nada que recortar)."""
    if grade_result.grade is None:
        return grade_result
    grade = grade_result.grade
    reasons = list(grade_result.reasons)
    correlation_too_high = (
        max_correlation_with_open_position is not None
        and max_correlation_with_open_position >= high_correlation_threshold
    )
    if correlation_too_high:
        reasons.append(
            f"Correlación alta ({max_correlation_with_open_position:.2f}) con una posición abierta - concentración"
        )
        grade = Grade.C
    if open_positions_count >= max_open_positions:
        reasons.append("Cartera llena - cada nueva entrada diluye el tamaño medio")
    return GradeResult(grade=grade, reasons=reasons, distance_atr=grade_result.distance_atr)
