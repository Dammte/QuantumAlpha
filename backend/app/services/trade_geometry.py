"""Reconstruction (2026-09), Fase 3: the "where" half of the levels/triggers
system the propietario commissioned to replace `recommendation_engine.py`'s
weighted checklist (see docs/quant_methodology.md and that module's own
docstring for the full history of what led here). Pure geometry, no verdict:
given a ticker's already-computed support/resistance levels and volatility,
at what exact price would a valid entry actually confirm, and what would it
cost to protect it once taken.

This module answers "what's about to trigger entry" (Parte 0, pregunta 2 of
the reconstruction brief) at the price level; `levels_engine.py` (same Fase)
adds the pass/fail gate on top of it (pregunta 3 - "is this specific entry
good"). `watchlist_service.py`'s own setup detectors - which *pattern*
(breakout, pullback, oversold bounce, trend continuation) a ticker matches -
were a separate concern this module never touched; that module retired
2026-09 (docs/quant_methodology.md §25), `GET /market/radar` now owns "which
pattern"/"is it good" together. This module still only answers "at what
price", never either of those.

Provenance note (2026-09-11): this file, `levels_engine.py`, and the tables/
jobs Fase 2 builds around them are the author's best-effort reconstruction of
Parte 3+ of the original brief, written after the literal spec text had
already scrolled out of context - the propietario explicitly signed off on
proceeding this way (see the session's own record) rather than re-pasting it.
Every threshold below is a first-pass value in the same spirit
`recommendation_engine.BUY_THRESHOLD`/`exit_engine.IMMINENT_CROSS_*` already
shipped as - not yet run through `scripts/factor_ablation_study.py` (Fase 8
retargets that script at trigger outcomes instead of the old score) - so
treat the exact numbers as adjustable, not as measured fact.

2026-09 (later pass, once the literal brief text was back in hand): the
above approximation shipped `compute_stop_and_target` - one fixed ATR
multiple, one fixed 2:1 target - never the real Parte 7 design (a stop
*cascade* by entry type, an ATR-percentile-adaptive risk ceiling, fixed-risk
sizing with three limits, cost-net reward:risk). `compute_entry_geometry`/
`size_position`/`TradeGeometry`/`EntryType` below are that real design,
added *alongside* the original function - not a replacement.
`compute_stop_and_target` keeps every existing caller
(`levels_engine.evaluate_gate`, `trade_plan_service.py`,
`scripts/factor_ablation_study.py`'s re-export) working unchanged; wiring
the richer functions into those call sites is deliberately left for its own
follow-up pass (`app.core.trading_params` already carries every constant
they need, so that migration is data plumbing, not new design).

The real design is deliberately split in two, not one `compute_trade_geometry`
call: `compute_entry_geometry` (stop/target/risk-ceiling, no capital input at
all) is what `evaluate_gate`/`daily_close.py` can call while scoring the
whole curated universe once a day, with no portfolio in scope - a ticker's
own gate/geometry doesn't belong to any one portfolio. `size_position`
(shares/position value/% of portfolio) only makes sense once a *specific*
portfolio's capital is known, e.g. rendering the Radar for one portfolio or
`trade_plan_service.py` at the moment a position is actually opened.
`compute_trade_geometry` is a thin convenience wrapper over both, for a
caller (tests, an on-demand single-ticker deep dive) that already has both
pieces of context up front.
"""

from dataclasses import dataclass, replace
from enum import Enum

from app.core.trading_params import (
    MAX_POSITION_PCT,
    MIN_POSITION_USD,
    MIN_RISK_REWARD_NET,
    RISK_CEILING_ATR_MULTIPLE,
    RISK_CEILING_MAX_PCT,
    RISK_CEILING_MIN_PCT,
    RISK_PER_TRADE_PCT,
    STOP_ATR_CEILING,
    STOP_CUSHION_ATR_CALM,
    STOP_CUSHION_ATR_NORMAL,
    STOP_CUSHION_ATR_VOLATILE,
    STOP_MIN_DISTANCE_ATR,
    TRANSACTION_COST_PCT,
    VOLATILITY_PROFILE_CALM_MAX_ATR_PCT,
    VOLATILITY_PROFILE_NORMAL_MAX_ATR_PCT,
    VOLATILITY_PROFILE_VOLATILE_MAX_ATR_PCT,
)
from app.services.technical_analysis import Level, LevelKind, LevelState, PriceLevel, TrendState

# A raw touch of the exact pivot price is noise, not confirmation - the
# breakout trigger sits a small buffer above the resistance itself. Smaller
# than support_resistance_levels' own 1.5% clustering tolerance on purpose:
# this buffer is about confirming a *break*, not about whether two pivots are
# "the same level".
BREAKOUT_BUFFER_PCT = 0.003  # 0.3% above the resistance pivot

# A resistance further than this isn't "about to" trigger - it's background
# context for the deep-dive view, not a Radar candidate. First-pass value,
# see module docstring.
BREAKOUT_TRIGGER_MAX_DISTANCE = 0.08  # 8%

# Same "close enough to matter" bar recommendation_engine.py's own
# near_support factor used before this reconstruction.
PULLBACK_PROXIMITY_PCT = 0.03

REWARD_RISK_RATIO = 2.0
MAX_RESISTANCE_TARGET_DISTANCE = 0.30


@dataclass(frozen=True, slots=True)
class EntryTrigger:
    """One concrete, watchable price level - not a verdict. `already_triggered`
    tells a caller whether *today's* price has already crossed it (a breakout
    already confirmed, a pullback already at support) versus still being
    something to wait and watch for."""

    trigger_type: str  # "breakout" | "pullback_bounce"
    trigger_price: float
    already_triggered: bool


@dataclass(frozen=True, slots=True)
class StopAndTarget:
    stop_loss: float | None
    take_profit: float | None
    take_profit_method: str | None
    risk_reward: float | None


def compute_entry_trigger(
    price: float,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
) -> EntryTrigger | None:
    """The single most relevant "what would make this a buy" price level for
    a ticker right now, or `None` when neither a nearby support nor a nearby
    resistance makes for a plausible near-term trigger.

    Pullback takes priority over breakout when both are technically present
    (rare, but possible on a wide multi-level chart): a price already sitting
    at support is an actionable trigger *today*, while a distant breakout is
    still just something to watch for - the more immediate answer wins.
    """
    if nearest_support is not None and abs(nearest_support.distance_pct) <= PULLBACK_PROXIMITY_PCT:
        return EntryTrigger(
            trigger_type="pullback_bounce",
            trigger_price=nearest_support.price,
            already_triggered=True,
        )
    if (
        nearest_resistance is not None
        and 0 < nearest_resistance.distance_pct <= BREAKOUT_TRIGGER_MAX_DISTANCE
    ):
        trigger_price = nearest_resistance.price * (1 + BREAKOUT_BUFFER_PCT)
        return EntryTrigger(
            trigger_type="breakout",
            trigger_price=trigger_price,
            already_triggered=price >= trigger_price,
        )
    return None


def compute_stop_and_target(
    price: float,
    atr14: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
) -> StopAndTarget:
    """The mechanical stop/target math - moved here unchanged from
    `recommendation_engine.compute_stop_and_target` (same function, same
    behavior, new home alongside the rest of the levels/triggers geometry).
    Place the stop just below the nearest support (a real technical level),
    but never let the risk exceed `ATR_STOP_MULTIPLE` x ATR (a
    volatility-aware ceiling) even if support is unusually far away; target
    the nearest resistance only if it still clears a minimum reward:risk,
    otherwise fall back to a fixed `REWARD_RISK_RATIO`:1 objective.

    Used by `levels_engine.evaluate_gate` below for the Radar's simple
    `stop_and_target` display. Auditoria del Radar, bloque H2: ya NO la usa
    `trade_plan_service.py` (que ahora reconstruye la geometría real de
    posiciones abiertas con `compute_entry_geometry`, la cascada unificada
    con anclaje persistido) - ver ese módulo. `STOP_ATR_CEILING`
    (`trading_params.py`) sustituye al `ATR_STOP_MULTIPLE=2,5` local que
    esta función usaba antes, que contradecía a ese mismo número sin que
    nadie lo hubiera notado - una sola fuente de verdad."""
    if not atr14:
        return StopAndTarget(None, None, None, None)

    candidate_stops = [price - STOP_ATR_CEILING * atr14]
    if nearest_support is not None:
        candidate_stops.append(nearest_support.price * 0.99)
    stop_loss = max(candidate_stops)  # the tighter of the two - never risk more than the ATR ceiling

    risk = price - stop_loss
    if risk <= 0:
        return StopAndTarget(stop_loss, None, None, None)

    resistance_target = None
    if nearest_resistance is not None and 0 < nearest_resistance.distance_pct <= MAX_RESISTANCE_TARGET_DISTANCE:
        resistance_target = nearest_resistance.price

    if resistance_target is not None and (resistance_target - price) / risk >= 1.0:
        take_profit = resistance_target
        take_profit_method = "resistencia más cercana"
    else:
        take_profit = price + REWARD_RISK_RATIO * risk
        take_profit_method = f"objetivo {REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"
    risk_reward = (take_profit - price) / risk

    return StopAndTarget(stop_loss, take_profit, take_profit_method, risk_reward)


# --- Parte 7 (real design): stop cascade + adaptive risk ceiling + sizing ----

class EntryType(str, Enum):
    """Which of Parte 7's four stop-cascade rungs actually produced this
    geometry's stop - kept on the result, not just used internally, so a
    caller/UI can show *why* the stop sits where it does (the same
    transparency principle `stop_basis`'s own prose text already gives, just
    as a stable enum instead of a string to match on)."""

    BREAKOUT = "breakout"
    PULLBACK_SUPPORT = "pullback_support"
    PULLBACK_EMA21 = "pullback_ema21"
    CONTINUATION_EMA55 = "continuation_ema55"


@dataclass(frozen=True, slots=True)
class TradeGeometry:
    """The full Parte 7 read for one potential entry: not just a stop/target
    (`StopAndTarget` above), but the complete chain from "where's the stop"
    to "how many shares" to "is this even worth taking" - every step visible,
    never collapsed to a bare yes/no. `viable=False` is a normal, frequent
    result (Parte 0's "empty list is valid"), never an error - `rejection_reason`
    says which specific check failed; every other field before that check is
    still populated where computable, so a caller can show *how close* a
    rejected setup came, not just that it was rejected."""

    entry_price: float
    stop_price: float | None
    stop_basis: str | None  # Spanish prose: qué peldaño de la cascada ancla este stop
    entry_type: EntryType | None
    level_kind: LevelKind | None  # Auditoria del Radar, bloque H2: el anclaje exacto, para persistir/mostrar
    risk_pct: float | None  # (entry - stop) / entry
    risk_atr: float | None  # (entry - stop) / atr14
    risk_ceiling_pct: float | None  # techo adaptativo INFORMATIVO (bloque H2, paso 4.1) - ya no rechaza
    target_price: float | None
    target_basis: str | None  # "resistencia más cercana" | "objetivo N:1 sobre el riesgo"
    reward_pct: float | None  # (target - entry) / entry
    risk_reward_gross: float | None
    risk_reward_net: float | None  # net of round-trip TRANSACTION_COST_PCT - what's actually shown
    shares_for_risk_budget: float | None
    position_value: float | None
    pct_of_portfolio: float | None  # position_value / capital_total
    viable: bool
    rejection_reason: str | None  # Spanish, only when viable is False
    # Auditoria del Radar, bloque H2/10: la clasificación de
    # `classify_volatility_profile` - "tranquilo"|"normal"|"volatil"|"extremo"
    # - expuesta en el resultado para que la UI (bloque H3,
    # `PositionDetailPanel.jsx`) pueda mostrar "el perfil de volatilidad del
    # valor y el techo de riesgo que le corresponde" sin reimplementar los
    # mismos umbrales en el cliente. `None` únicamente cuando no hay ATR con
    # el que clasificar nada (mismo caso que deja `risk_ceiling_pct` en `None`).
    volatility_profile: str | None = None


def _not_viable(entry_price: float, reason: str, **known: float | None) -> TradeGeometry:
    """Builds a rejected `TradeGeometry`, filling in whatever the caller
    already knows (Parte 0: showing *why* y *how close* beats a bare
    rejection) and `None`/`False` for everything downstream of the failure."""
    fields = dict(
        entry_price=entry_price, stop_price=None, stop_basis=None, entry_type=None, level_kind=None,
        risk_pct=None, risk_atr=None, risk_ceiling_pct=None,
        target_price=None, target_basis=None, reward_pct=None,
        risk_reward_gross=None, risk_reward_net=None,
        shares_for_risk_budget=None, position_value=None, pct_of_portfolio=None,
        volatility_profile=None,
    )
    fields.update(known)
    return TradeGeometry(**fields, viable=False, rejection_reason=reason)


def classify_volatility_profile(atr_pct: float) -> str:
    """Auditoria del Radar, bloque H2, paso 1: "la diferencia no está en el
    porcentaje que tolero, está en qué nivel del gráfico es lo bastante
    robusto para ese valor" - el perfil decide el colchón (`_cushion_for_profile`)
    y, indirectamente, qué tan fácil es que un anclaje cercano se descarte
    por "demasiado cerca" (`STOP_MIN_DISTANCE_ATR` es el mismo en ATR para
    los cuatro perfiles, pero un ATR mayor en términos de precio ya exige
    más margen en euros por sí solo)."""
    if atr_pct < VOLATILITY_PROFILE_CALM_MAX_ATR_PCT:
        return "tranquilo"
    if atr_pct < VOLATILITY_PROFILE_NORMAL_MAX_ATR_PCT:
        return "normal"
    if atr_pct < VOLATILITY_PROFILE_VOLATILE_MAX_ATR_PCT:
        return "volatil"
    return "extremo"


def _cushion_for_profile(profile: str) -> float:
    if profile == "tranquilo":
        return STOP_CUSHION_ATR_CALM
    if profile == "normal":
        return STOP_CUSHION_ATR_NORMAL
    return STOP_CUSHION_ATR_VOLATILE  # "volatil" y "extremo" comparten colchón - ver trading_params.py


def _broken_resistance_level(levels: list[Level] | None) -> Level | None:
    """Auditoria del Radar, bloque H1: el escalón de ruptura de la cascada
    vieja comparaba `price` contra `nearest_resistance.price` - una
    resistencia (`support_resistance_levels`, `PriceLevel`) es, POR
    CONSTRUCCIÓN, un pivote por ENCIMA del precio actual
    (`p > current_price` en su propio filtro), así que esa condición nunca
    podía cumplirse: pedía que el precio superara un nivel que, por
    definición, seguía por encima del precio. El mismo patrón de
    auto-referencia que ya apareció varias veces en la biblioteca de setups.
    El arreglo real: usar `ta.Level` (el tipo con estado que `detect_levels`
    ya calcula y que `breakout.py` ya usa para esto mismo) - un pivote de
    resistencia con `state == BROKEN_CONFIRMED` es, literalmente, "el
    precio ya lo rompió y lo confirmó", sin la contradicción."""
    if not levels:
        return None
    candidates = [
        lv for lv in levels if lv.kind == LevelKind.PIVOT_RESISTANCE and lv.state == LevelState.BROKEN_CONFIRMED
    ]
    return min(candidates, key=lambda lv: lv.distance_atr) if candidates else None


def _range_low_20_level(levels: list[Level] | None) -> Level | None:
    if not levels:
        return None
    return next((lv for lv in levels if lv.kind == LevelKind.RANGE_LOW_20), None)


def _stop_cascade_candidates(
    price: float,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    ema21: float | None,
    ema55: float | None,
    trend: TrendState,
    levels: list[Level] | None,
) -> list[tuple[float, EntryType, str, LevelKind]]:
    """Auditoria del Radar, bloque H2, paso 2: TODOS los anclajes que
    aplican hoy, en el mismo orden de prioridad que la cascada original de
    Parte 7 - `(level_price, entry_type, basis_label, level_kind)` por
    cada uno. Ya no "el primero que aplica y se acabó": `compute_entry_geometry`
    recorre esta lista completa y descarta los que resulten "demasiado
    cerca" (paso 4.3) probando el siguiente, en vez de rechazar la
    operación entera por un colchón que hizo que un anclaje válido quedara
    a menos de `STOP_MIN_DISTANCE_ATR`.

    1. Ruptura: una resistencia rota y confirmada (`_broken_resistance_level`).
    2. Rebote: precio sentado sobre `nearest_support` (misma proximidad que
       `compute_entry_trigger` usa para su propio disparador de pullback).
    3. Pullback a EMA21: solo en tendencia alcista, precio en o justo sobre
       la EMA21.
    4. Continuación sobre EMA55: solo en tendencia alcista, precio por
       encima de la EMA55.
    5. Respaldo, bloque H2 paso 2 literal ("sin estructura cercana válida
       → bajo el mínimo de 20 sesiones"): el único anclaje que no depende
       del tipo de entrada - siempre se ofrece como último recurso si
       `detect_levels` calculó uno."""
    candidates: list[tuple[float, EntryType, str, LevelKind]] = []

    broken_resistance = _broken_resistance_level(levels)
    if broken_resistance is not None:
        candidates.append(
            (
                broken_resistance.price,
                EntryType.BREAKOUT,
                f"bajo el nivel de resistencia roto en {broken_resistance.price:.2f}",
                LevelKind.PIVOT_RESISTANCE,
            )
        )
    if nearest_support is not None and abs(nearest_support.distance_pct) <= PULLBACK_PROXIMITY_PCT:
        candidates.append(
            (
                nearest_support.price,
                EntryType.PULLBACK_SUPPORT,
                f"bajo el soporte en {nearest_support.price:.2f}",
                LevelKind.PIVOT_SUPPORT,
            )
        )
    if trend == TrendState.UPTREND and ema21 is not None and price >= ema21:
        if (price - ema21) / price <= PULLBACK_PROXIMITY_PCT:
            candidates.append((ema21, EntryType.PULLBACK_EMA21, f"bajo la EMA21 ({ema21:.2f})", LevelKind.EMA21))
    if trend == TrendState.UPTREND and ema55 is not None and price > ema55:
        candidates.append(
            (ema55, EntryType.CONTINUATION_EMA55, f"bajo la EMA55 ({ema55:.2f})", LevelKind.EMA55)
        )
    range_low = _range_low_20_level(levels)
    if range_low is not None and range_low.price < price:
        candidates.append(
            (
                range_low.price,
                EntryType.PULLBACK_SUPPORT,
                f"bajo el mínimo de 20 sesiones en {range_low.price:.2f}",
                LevelKind.RANGE_LOW_20,
            )
        )
    return candidates


def compute_entry_geometry(
    price: float,
    atr14: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    ema21: float | None,
    ema55: float | None,
    trend: TrendState,
    levels: list[Level] | None = None,
) -> TradeGeometry:
    """The ticker-only half of the real Parte 7 pipeline - stop cascade,
    colchón por perfil de volatilidad, y objetivo neto de costes,
    deliberadamente sin capital ni tamaño de posición. `shares_for_risk_budget`/
    `position_value`/`pct_of_portfolio` son siempre `None` en el resultado de
    esta función; llama a `size_position` aparte una vez se conoce el capital
    de una cartera concreta.

    Auditoria del Radar, bloque H2 - rediseño completo tras el diagnóstico
    H1 (dos bugs reales, no solo el ajuste de colchón que pedía la
    especificación original): "el stop no se mueve para caber; el tamaño
    sí" (literal) - esta función YA NO rechaza por techo de riesgo ni
    encoge el stop con `STOP_ATR_CEILING`; ambos existían para "caber" en un
    presupuesto, y ese presupuesto lo absorbe `size_position` reduciendo
    acciones, no aquí. En orden:

    1. `_stop_cascade_candidates` da TODOS los anclajes que aplican hoy, en
       el mismo orden de prioridad de siempre (ruptura > rebote > EMA21 >
       EMA55 > mínimo de 20 sesiones).
    2. El colchón bajo el nivel ya no es fijo por tipo de entrada - escala
       con el perfil de volatilidad del propio valor
       (`classify_volatility_profile`/`_cushion_for_profile`).
    3. Se recorre la cascada en orden; un candidato cuyo stop resultante
       quede más cerca que `STOP_MIN_DISTANCE_ATR` ATR se descarta como
       ruido y se prueba el siguiente (bloque H2, paso 4.3) - nunca se
       acepta un stop demasiado ajustado, y nunca se rechaza la operación
       entera solo porque el PRIMER candidato resultó demasiado cerca.
    4. Si algo se acepta, el techo de riesgo adaptativo
       (`RISK_CEILING_ATR_MULTIPLE * atr_pct`, acotado a
       `[RISK_CEILING_MIN_PCT, RISK_CEILING_MAX_PCT]`) se calcula y se
       expone en `risk_ceiling_pct` de forma puramente INFORMATIVA - ya no
       decide viabilidad.
    5. El objetivo sigue igual: resistencia más cercana si su beneficio:riesgo
       *neto* de costes alcanza `MIN_RISK_REWARD_NET`, si no el objetivo fijo
       `REWARD_RISK_RATIO`:1, si ninguno alcanza, rechazo."""
    if not atr14 or atr14 <= 0:
        return _not_viable(price, "ATR no disponible - no se puede definir un stop con base de volatilidad")

    atr_pct = atr14 / price
    profile = classify_volatility_profile(atr_pct)
    cushion = _cushion_for_profile(profile)
    risk_ceiling_pct = min(max(RISK_CEILING_ATR_MULTIPLE * atr_pct, RISK_CEILING_MIN_PCT), RISK_CEILING_MAX_PCT)

    candidates = _stop_cascade_candidates(price, nearest_support, nearest_resistance, ema21, ema55, trend, levels)
    if not candidates:
        return _not_viable(
            price, "sin nivel de referencia (soporte/resistencia/EMA21/EMA55) para anclar el stop",
            risk_ceiling_pct=risk_ceiling_pct, volatility_profile=profile,
        )

    stop_price = stop_basis = entry_type = level_kind = None
    risk_per_share = risk_atr = None
    for level_price, candidate_entry_type, basis_label, candidate_level_kind in candidates:
        raw_stop = level_price - cushion * atr14
        raw_risk_per_share = price - raw_stop
        if raw_risk_per_share <= 0:
            continue  # el anclaje queda por encima (o igual a) el precio - no es un stop válido
        raw_risk_atr = raw_risk_per_share / atr14
        if raw_risk_atr < STOP_MIN_DISTANCE_ATR:
            continue  # demasiado cerca - ruido disfrazado de nivel (bloque H2, paso 4.3), prueba el siguiente
        stop_price, stop_basis = raw_stop, basis_label
        entry_type, level_kind = candidate_entry_type, candidate_level_kind
        risk_per_share, risk_atr = raw_risk_per_share, raw_risk_atr
        break

    if stop_price is None:
        return _not_viable(
            price,
            "todos los anclajes disponibles quedan demasiado cerca del precio (ruido) o por encima de él",
            risk_ceiling_pct=risk_ceiling_pct, volatility_profile=profile,
        )

    risk_pct = risk_per_share / price

    round_trip_cost_pct = 2 * TRANSACTION_COST_PCT

    def _net_reward_risk(target_price: float) -> tuple[float, float]:
        gross_reward = target_price - price
        gross = gross_reward / risk_per_share
        net_reward_pct = (gross_reward / price) - round_trip_cost_pct
        net = (net_reward_pct * price) / risk_per_share
        return gross, net

    target_price = target_basis = None
    risk_reward_gross = risk_reward_net = None
    if nearest_resistance is not None and nearest_resistance.price > price:
        gross, net = _net_reward_risk(nearest_resistance.price)
        if net >= MIN_RISK_REWARD_NET:
            target_price, target_basis = nearest_resistance.price, "resistencia más cercana"
            risk_reward_gross, risk_reward_net = gross, net

    if target_price is None:
        fixed_target = price + REWARD_RISK_RATIO * risk_per_share
        gross, net = _net_reward_risk(fixed_target)
        if net >= MIN_RISK_REWARD_NET:
            target_price, target_basis = fixed_target, f"objetivo {REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"
            risk_reward_gross, risk_reward_net = gross, net
        else:
            reason = (
                "resistencia demasiado cerca - ni ella ni el objetivo fijo alcanzan un "
                "beneficio:riesgo mínimo"
                if nearest_resistance is not None
                else "riesgo por acción demasiado pequeño - el objetivo fijo no compensa los costes"
            )
            return _not_viable(
                price, reason,
                stop_price=stop_price, stop_basis=stop_basis, entry_type=entry_type, level_kind=level_kind,
                risk_pct=risk_pct, risk_atr=risk_atr, risk_ceiling_pct=risk_ceiling_pct,
                volatility_profile=profile,
            )

    reward_pct = (target_price - price) / price

    return TradeGeometry(
        entry_price=price,
        stop_price=stop_price, stop_basis=stop_basis, entry_type=entry_type, level_kind=level_kind,
        risk_pct=risk_pct, risk_atr=risk_atr, risk_ceiling_pct=risk_ceiling_pct,
        target_price=target_price, target_basis=target_basis, reward_pct=reward_pct,
        risk_reward_gross=risk_reward_gross, risk_reward_net=risk_reward_net,
        shares_for_risk_budget=None, position_value=None, pct_of_portfolio=None,
        viable=True, rejection_reason=None, volatility_profile=profile,
    )


def size_position(
    geometry: TradeGeometry, capital_total: float, atr_percentile_252: float | None = None
) -> TradeGeometry:
    """The capital-dependent half of Parte 7, kept separate from
    `compute_entry_geometry` on purpose: `evaluate_gate`/`daily_close.py`
    score the whole curated universe once a day with no portfolio in scope
    at all (a ticker's own gate/geometry doesn't belong to any one
    portfolio) - `capital_total` only exists once a *specific* portfolio is
    being sized against a *specific* trigger, e.g. the Radar rendering for
    one portfolio, or `trade_plan_service.py` at the moment a position is
    actually opened. Calling this on a `geometry` that isn't `viable` (no
    stop/target to size against) returns it unchanged - sizing an already-
    rejected setup would fabricate numbers for a trade that was never on
    the table to begin with.

    Sizes for `RISK_PER_TRADE_PCT` of `capital_total`, halved once
    `atr_percentile_252 >= 0.85` (an unusually volatile stretch for this
    specific ticker), capped at `MAX_POSITION_PCT` of capital, rejected
    outright under `MIN_POSITION_USD` (a position that small lets
    transaction costs eat the trade). The 6% aggregate-risk-across-all-
    positions cap from the brief is deliberately NOT enforced here - that
    needs every other open position's own risk, which is
    `portfolio_construction_service.final_position_size`'s job, one layer up
    from a single ticker's own geometry."""
    if not geometry.viable or geometry.stop_price is None:
        return geometry

    risk_per_share = geometry.entry_price - geometry.stop_price
    shares_for_risk_budget = (capital_total * RISK_PER_TRADE_PCT) / risk_per_share
    if atr_percentile_252 is not None and atr_percentile_252 >= 0.85:
        shares_for_risk_budget /= 2

    position_value = shares_for_risk_budget * geometry.entry_price
    max_position_value = capital_total * MAX_POSITION_PCT
    if position_value > max_position_value:
        position_value = max_position_value
        shares_for_risk_budget = max_position_value / geometry.entry_price

    pct_of_portfolio = (position_value / capital_total) if capital_total > 0 else None
    sized = replace(
        geometry,
        shares_for_risk_budget=shares_for_risk_budget,
        position_value=position_value,
        pct_of_portfolio=pct_of_portfolio,
    )

    if position_value < MIN_POSITION_USD:
        reason = "posición demasiado pequeña - el coste de transacción se comería el resultado"
        return replace(sized, viable=False, rejection_reason=reason)

    return sized


def compute_trade_geometry(
    price: float,
    atr14: float | None,
    nearest_support: PriceLevel | None,
    nearest_resistance: PriceLevel | None,
    ema21: float | None,
    ema55: float | None,
    trend: TrendState,
    capital_total: float,
    atr_percentile_252: float | None = None,
    levels: list[Level] | None = None,
) -> TradeGeometry:
    """Convenience wrapper combining `compute_entry_geometry` and
    `size_position` in one call - for a caller that already knows a specific
    portfolio's capital up front (a test, or a one-off on-demand computation
    like `GET /tickers/{t}`'s deep dive). `evaluate_gate`/`daily_close.py`
    should call `compute_entry_geometry` alone (see its docstring for why)
    and size separately, per portfolio, only where a trigger is actually
    being acted on."""
    geometry = compute_entry_geometry(
        price, atr14, nearest_support, nearest_resistance, ema21, ema55, trend, levels
    )
    return size_position(geometry, capital_total, atr_percentile_252)


_GEOMETRY_FIELDS = (
    "entry_price", "stop_price", "stop_basis", "risk_pct", "risk_atr", "risk_ceiling_pct",
    "target_price", "target_basis", "reward_pct", "risk_reward_gross", "risk_reward_net",
    "shares_for_risk_budget", "position_value", "pct_of_portfolio", "viable", "rejection_reason",
)


def geometry_to_dict(geometry: TradeGeometry) -> dict:
    """Plain JSON-safe read of a `TradeGeometry` - same choice
    `TickerDailyState.gate_conditions` already made for `GateCondition`, used
    here so `scripts/daily_close.py` can persist the ticker-only half of
    Parte 7 (`compute_entry_geometry`, never `size_position` - no portfolio
    in scope in that job) on `TickerDailyState.entry_geometry` without a
    schema-specific mapper. `entry_type`/`level_kind` become their plain
    string value (`None` stays `None`) - the two fields that aren't already
    JSON-safe."""
    data = {field: getattr(geometry, field) for field in _GEOMETRY_FIELDS}
    data["entry_type"] = geometry.entry_type.value if geometry.entry_type is not None else None
    data["level_kind"] = geometry.level_kind.value if geometry.level_kind is not None else None
    data["volatility_profile"] = geometry.volatility_profile
    return data


def geometry_from_dict(data: dict) -> TradeGeometry:
    """Inverse of `geometry_to_dict` - reconstructs a real `TradeGeometry` (not
    just a display shape) so a caller with a specific portfolio's capital in
    view (e.g. `GET /market/radar?portfolio_id=`) can run the persisted,
    unsized geometry through `size_position` without recomputing the stop
    cascade from scratch. `level_kind` defaults to `None` on read for rows
    persisted before this field existed (Parte 4's own optional-field
    pattern) - a `NameError`-free, honest "no lo sabemos", not a fabricated
    value."""
    fields = {field: data[field] for field in _GEOMETRY_FIELDS}
    entry_type = data["entry_type"]
    level_kind = data.get("level_kind")
    return TradeGeometry(
        entry_type=EntryType(entry_type) if entry_type is not None else None,
        level_kind=LevelKind(level_kind) if level_kind is not None else None,
        volatility_profile=data.get("volatility_profile"),
        **fields,
    )
