"""Parte 10 del encargo: medición histórica de cada setup de la biblioteca
del Radar - "un setup sin medición es una opinión con nombre técnico"
(literal, y no es opcional). Extiende el mismo patrón de replay punto-en-
el-tiempo que `levels_engine.replay_gate_at`/`scripts/factor_ablation_study.py`
ya establecieron para el gate, aplicado ahora a `setups.registry.detect_all`
en vez de a `evaluate_gate` - misma rejilla no solapada, mismo etiquetado de
triple barrera (`backtest_engine.label_triple_barrier`), mismos costes
reales (`backtest_engine.ROUND_TRIP_COST_PCT`).

Este módulo es el MOTOR (funciones puras, sin red ni base de datos) - igual
que `backtest_engine.py` es el motor y `scripts/factor_ablation_study.py` es
el script que lo ejecuta contra el universo real y guarda el resultado.
`scripts/setup_replay_study.py` (fase posterior) hace ese papel aquí: baja
el histórico real, llama a `replay_setups_for_ticker` por ticker, agrega con
`aggregate_setup_performance` y persiste en la tabla `setup_performance`
(Parte 10.2, todavía sin construir).

**Dos etapas, no una** - "Cuando el setup alcanza READY, registra el gatillo
propuesto" y luego "etiqueta con triple barrera" son DOS eventos distintos,
no uno solo: la Parte 10.2 pide `trigger_rate` ("de los READY, ¿qué % llegó
a disparar?") como una métrica SEPARADA de `win_rate` ("de los disparados,
¿qué % tocó objetivo antes que stop?"). Si se etiquetara la barrera desde el
propio bar en que el setup llega a READY, esas dos preguntas colapsarían en
una sola y `trigger_rate` dejaría de tener sentido. Por eso
`_find_trigger_bar` busca la primera sesión, dentro de una ventana
razonable, en que el cierre confirma el nivel propuesto - solo esa
sub-muestra (la que sí confirmó) se etiqueta con triple barrera, con la
geometría real calculada en el momento en que el setup llegó a READY (no
recalculada de nuevo al disparar).

**Simplificaciones documentadas, mismo criterio que `replay_gate_at`** (ver
su propio docstring - "no aproximaciones silenciosas"):

- `ctx.levels` siempre `[]`: el escaneo de pivotes de
  `technical_analysis.detect_levels` es O(n) por llamada - repetirlo en
  cada punto de una rejilla histórica es el mismo coste "prohibitivo" que
  `replay_gate_at` ya documenta para soporte/resistencia más cercano.
  `breakout.py`/`pullback.py` (los dos únicos detectores que leen
  `ctx.levels`) estructuralmente nunca disparan en este replay - una
  limitación real y documentada, no un hueco silencioso. Como
  consecuencia, `trade_geometry.compute_entry_geometry` recibe
  `nearest_support=nearest_resistance=None` aquí siempre - la misma
  degradación ya aceptada para el replay del gate (cascada de stop por
  ATR, objetivo fijo 2:1), no un caso especial inventado para este módulo.
- `ctx.rs_percentile`/`ctx.sector_rs_percentile` siempre `None`: percentiles
  transversales sobre el universo completo en una fecha histórica
  arbitraria no son reconstruibles sin recalcular todo el universo en cada
  punto - mismo motivo exacto que `replay_gate_at` ya documenta para RS
  Rating. Ningún detector de esta biblioteca los lee para decidir (los usa
  `context_modifiers.py`, una capa posterior a la detección), así que no
  hay nada que omitir en la propia detección de setups.
- Semanal/`multi_timeframe`/`mansfield_rs_series` SÍ se reconstruyen de
  verdad en cada punto, a diferencia del gate (que usa la proxy diaria
  SMA150 por coste): `stage_transition.py` los necesita de verdad para ser
  replayable en absoluto, y remuestrear a semanal es barato (vectorizado
  en pandas), a diferencia del escaneo de pivotes de más arriba.

**Deduplicación en la rejilla**: un setup puede seguir en READY durante más
sesiones que el propio paso de la rejilla (`grid_stride_bars`) - sin
protección, el mismo READY se contaría una vez por cada punto de rejilla en
que sigue vigente, inflando `n_observations`. Se usa el propio
`SetupMatch.bars_in_stage` (que cada detector ya rellena) para quedarse
solo con la observación de un READY reciente (`bars_in_stage < grid_stride_bars`)
- aproximadamente "la primera vez que se ve", sin necesitar guardar estado
entre puntos de la rejilla."""

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from app.domain.models.setup_performance import SetupPerformance
from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services import trade_geometry as tg
from app.services.backtest_engine import ROUND_TRIP_COST_PCT, TripleBarrierLabel, label_triple_barrier
from app.services.levels_engine import compute_grade
from app.services.setups import registry as setups_registry
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupMatch, SetupStage

REPLAY_WARMUP_BARS = 260  # mismo que backtest_engine.WARMUP_BARS - SMA200 + su propio lookback de pendiente
REPLAY_GRID_STRIDE_BARS = 10  # una rejilla no solapada, mismo orden de magnitud que backtest_engine
# backtest_engine.VERTICAL_BARRIER_HORIZONS[1] - el horizonte largo de esta cartera
REPLAY_VERTICAL_BARRIER_BARS = 21
REPLAY_TRIGGER_WINDOW_BARS = 10  # cuánto se espera a que el precio confirme el nivel antes de darlo por caducado
REPLAY_FAILURE_WINDOW_BARS = 3  # Parte 10.2, "failure_rate_3d", literal
MIN_SAMPLE_FOR_STATS = 30  # Parte 10.3, literal


@dataclass(frozen=True, slots=True)
class SetupReplayObservation:
    """Un único READY histórico de un detector, con su desenlace (si lo
    hubo). `label=None` cuando el setup nunca llegó a confirmar el nivel
    dentro de `REPLAY_TRIGGER_WINDOW_BARS` - cuenta para `trigger_rate`,
    no para el resto de las métricas de la Parte 10.2."""

    ticker: str
    region: str
    setup_name: str
    family: str
    ready_date: date
    grade: str | None  # "A" | "B" | "C" | None - el grado en el momento del READY, sin cartera
    market_regime: str | None  # "market_above_sma200" | "market_below_sma200" | None
    triggered: bool
    # Riesgo inicial (Parte 7, `TradeGeometry.risk_pct`) en el momento del
    # READY - `TripleBarrierLabel` no lo lleva (es un resultado genérico,
    # reutilizado por `backtest_engine.py` para contextos sin sizing), pero
    # `expectancy_r` (Parte 10.2, "en múltiplos de R") lo necesita para
    # convertir el retorno neto en múltiplos de riesgo.
    risk_pct: float | None
    label: TripleBarrierLabel | None


def _at(series: pd.Series, i: int) -> float | None:
    value = series.iloc[i]
    return None if pd.isna(value) else float(value)


def _build_point_in_time_context(
    ticker: str,
    region: str,
    i: int,
    dates: pd.DatetimeIndex,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    open_: pd.Series,
    atr_series: pd.Series,
    ema21_series: pd.Series,
    ema55_series: pd.Series,
    sma20_series: pd.Series,
    sma50_series: pd.Series,
    sma150_series: pd.Series,
    sma200_series: pd.Series,
    rsi14_series: pd.Series,
    benchmark_close: pd.Series | None,
) -> SetupContext:
    """Reconstruye el `SetupContext` exacto que `daily_close.py` habría
    construido en la fecha `dates[i]`, usando solo información conocida
    hasta esa barra. Las series de indicadores (EMA/SMA/RSI/ATR) se
    calculan UNA SOLA VEZ sobre el histórico completo antes del bucle de
    reproducción (son causales por construcción - leerlas en la posición
    `i` da el mismo valor que recalcularlas sobre un recorte, el mismo
    principio que `levels_engine.replay_gate_at` ya usa), evitando repetir
    ese cómputo en cada punto de la rejilla."""
    close_i, high_i, low_i, volume_i, open_i = (
        close.iloc[: i + 1], high.iloc[: i + 1], low.iloc[: i + 1], volume.iloc[: i + 1], open_.iloc[: i + 1]
    )
    daily_df_i = pd.DataFrame(
        {"open": open_i, "high": high_i, "low": low_i, "close": close_i, "volume": volume_i}, index=dates[: i + 1]
    )
    multi_i = mtf.analyze_multi_timeframe(daily_df_i)
    weekly_df_i = ta.resample_ohlcv(daily_df_i, mtf.WEEKLY_RULE)
    has_weekly = len(weekly_df_i) >= 2

    mansfield_series = None
    if benchmark_close is not None:
        bench_i = benchmark_close.reindex(dates[: i + 1]).ffill().dropna()
        if len(bench_i) >= 21:
            mansfield_series = ta.mansfield_rs(close_i, bench_i.reindex(close_i.index).ffill(), window=20)

    return SetupContext(
        ticker=ticker,
        region=region,
        trade_date=dates[i].date(),
        close=close_i, high=high_i, low=low_i, volume=volume_i, open_=open_i,
        weekly_close=weekly_df_i["close"] if has_weekly else None,
        weekly_high=weekly_df_i["high"] if has_weekly else None,
        weekly_low=weekly_df_i["low"] if has_weekly else None,
        weekly_volume=weekly_df_i["volume"] if has_weekly else None,
        atr_series=atr_series.iloc[: i + 1],
        atr14=_at(atr_series, i),
        ema21=_at(ema21_series, i), ema55=_at(ema55_series, i),
        sma20=_at(sma20_series, i), sma50=_at(sma50_series, i),
        sma150=_at(sma150_series, i), sma200=_at(sma200_series, i),
        rsi14=_at(rsi14_series, i),
        levels=[],  # ver docstring del módulo
        multi_timeframe=multi_i,
        trend=multi_i.daily.trend,
        weekly_stage=multi_i.weekly.stage if multi_i.weekly is not None else None,
        relative_volume=ta.relative_volume(volume_i),
        rs_percentile=None, sector_rs_percentile=None,  # ver docstring del módulo
        mansfield_rs_series=mansfield_series,
    )


def _find_trigger_bar(close: pd.Series, ready_index: int, trigger_price: float, window_bars: int) -> int | None:
    """Primera sesión, tras `ready_index` y dentro de `window_bars`
    sesiones, cuyo cierre supera `trigger_price` - `None` si nunca ocurre
    en ese plazo (un READY que nunca llegó a confirmarse)."""
    end = min(ready_index + window_bars, len(close) - 1)
    for j in range(ready_index + 1, end + 1):
        if float(close.iloc[j]) > trigger_price:
            return j
    return None


def _market_regime_label(benchmark_close: pd.Series | None, i: int) -> str | None:
    if benchmark_close is None:
        return None
    bench_i = benchmark_close.iloc[: i + 1].dropna()
    market_trend, _ = ta.market_regime_inputs(bench_i if len(bench_i) > 0 else None, None)
    if market_trend is None:
        return None
    return "market_above_sma200" if market_trend == ta.TrendState.UPTREND else "market_below_sma200"


def replay_setups_for_ticker(
    daily_df: pd.DataFrame,
    ticker: str,
    region: str,
    benchmark_close: pd.Series | None = None,
    grid_stride_bars: int = REPLAY_GRID_STRIDE_BARS,
    vertical_barrier_bars: int = REPLAY_VERTICAL_BARRIER_BARS,
    trigger_window_bars: int = REPLAY_TRIGGER_WINDOW_BARS,
    warmup_bars: int = REPLAY_WARMUP_BARS,
) -> list[SetupReplayObservation]:
    """Recorre el histórico de un ticker con una rejilla no solapada
    (Parte 10.1), reproduce `setups.registry.detect_all` punto-en-el-tiempo
    en cada uno, y etiqueta con triple barrera cada READY que llegó a
    confirmarse. `[]` si no hay historial suficiente - mismo criterio que
    `backtest_engine.find_triple_barrier_entries`."""
    close, high, low, volume, open_ = (
        daily_df["close"], daily_df["high"], daily_df["low"], daily_df["volume"], daily_df["open"]
    )
    dates = daily_df.index
    n = len(close)
    last_valid_start = n - vertical_barrier_bars - trigger_window_bars - 1
    if last_valid_start <= warmup_bars:
        return []

    atr_series = ta.atr(high, low, close)
    ema21_series = ta.ema(close, mtf.FAST_MA_PERIOD)
    ema55_series = ta.ema(close, mtf.SLOW_MA_PERIOD)
    sma20_series = ta.sma(close, 20)
    sma50_series = ta.sma(close, 50)
    sma150_series = ta.sma(close, 150)
    sma200_series = ta.sma(close, 200)
    rsi14_series = ta.rsi(close)

    observations: list[SetupReplayObservation] = []
    for i in range(warmup_bars, last_valid_start, grid_stride_bars):
        ctx = _build_point_in_time_context(
            ticker, region, i, dates, close, high, low, volume, open_, atr_series, ema21_series, ema55_series,
            sma20_series, sma50_series, sma150_series, sma200_series, rsi14_series, benchmark_close,
        )
        for match in setups_registry.detect_all(ctx):
            if match.stage != SetupStage.READY or match.trigger_price is None:
                continue
            if match.bars_in_stage >= grid_stride_bars:
                continue  # ya contado en un punto de rejilla anterior - ver docstring del módulo

            geometry = tg.compute_entry_geometry(
                price=float(close.iloc[i]), atr14=ctx.atr14, nearest_support=None, nearest_resistance=None,
                ema21=ctx.ema21, ema55=ctx.ema55, trend=ctx.trend,
            )
            grade_value = None
            if geometry.viable:
                weekly_bullish = mtf.timeframe_bias(ctx.multi_timeframe.weekly) == "bullish"
                entry_trigger = tg.EntryTrigger(
                    trigger_type=geometry.entry_type.value, trigger_price=match.trigger_price,
                    already_triggered=False,
                )
                grade_result = compute_grade(
                    price=float(close.iloc[i]), atr14=ctx.atr14, entry_trigger=entry_trigger, geometry=geometry,
                    weekly_bullish=weekly_bullish, relative_volume=ctx.relative_volume, sma200=ctx.sma200,
                )
                grade_value = grade_result.grade.value if grade_result.grade is not None else None

            trigger_index = _find_trigger_bar(close, i, match.trigger_price, trigger_window_bars)
            label = None
            if trigger_index is not None and geometry.viable:
                label = label_triple_barrier(
                    close, high, low, trigger_index, geometry.stop_price, geometry.target_price,
                    vertical_barrier_bars, open_=open_,
                )

            observations.append(
                SetupReplayObservation(
                    ticker=ticker, region=region, setup_name=match.name, family=match.family.value,
                    ready_date=dates[i].date(), grade=grade_value,
                    market_regime=_market_regime_label(benchmark_close, i),
                    triggered=trigger_index is not None,
                    risk_pct=geometry.risk_pct if geometry.viable else None,
                    label=label,
                )
            )
    return observations


@dataclass(frozen=True, slots=True)
class SetupPerformanceStats:
    """Una fila de la tabla `setup_performance` (Parte 10.2) - `grade`/
    `market_regime` en `None` significan "todos los grados"/"ambos
    regímenes" (la fila sin segmentar), no "desconocido". Las siete
    métricas literales de la Parte 10.2, más `confidence`
    (`SetupConfidence`, Parte 10.3): `MEASURED` solo con
    `n_observations >= MIN_SAMPLE_FOR_STATS`, `THIN` por debajo con
    muestra > 0, nunca fabricado."""

    setup_name: str
    family: str
    grade: str | None
    market_regime: str | None
    n_observations: int
    trigger_rate: float | None
    win_rate: float | None
    expectancy_r: float | None
    median_bars_held: float | None
    mae_p80_pct: float | None
    failure_rate_3d: float | None
    confidence: SetupConfidence


def _confidence_for_sample(n: int) -> SetupConfidence:
    if n <= 0:
        return SetupConfidence.UNVALIDATED
    return SetupConfidence.MEASURED if n >= MIN_SAMPLE_FOR_STATS else SetupConfidence.THIN


def _stats_for_group(
    setup_name: str, family: str, grade: str | None, market_regime: str | None,
    observations: list[SetupReplayObservation],
) -> SetupPerformanceStats:
    n = len(observations)
    triggered = [o for o in observations if o.triggered and o.label is not None and o.risk_pct]
    trigger_rate = (len(triggered) / n) if n > 0 else None

    if not triggered:
        return SetupPerformanceStats(
            setup_name=setup_name, family=family, grade=grade, market_regime=market_regime,
            n_observations=n, trigger_rate=trigger_rate, win_rate=None, expectancy_r=None,
            median_bars_held=None, mae_p80_pct=None, failure_rate_3d=None,
            confidence=_confidence_for_sample(n),
        )

    # "Tocó objetivo antes que stop" (Parte 10.2, literal) - exit_reason
    # "target" exactamente, no una barrera vertical con retorno positivo.
    wins = [o for o in triggered if o.label.exit_reason == "target"]
    win_rate = len(wins) / len(triggered)

    r_multiples = [(o.label.return_pct - ROUND_TRIP_COST_PCT) / o.risk_pct for o in triggered]
    expectancy_r = float(np.mean(r_multiples))

    bars_held = [o.label.bars_held for o in triggered]
    median_bars_held = float(np.median(bars_held))

    # Percentil 80 de la MAGNITUD de la excursión adversa (Parte 10.2:
    # "dónde poner el stop de verdad") - se reporta en negativo, mismo
    # signo que TripleBarrierLabel.mae_pct.
    mae_p80_pct = -float(np.percentile([abs(o.label.mae_pct) for o in triggered], 80))

    quick_failures = [
        o for o in triggered if o.label.exit_reason == "stop" and o.label.bars_held <= REPLAY_FAILURE_WINDOW_BARS
    ]
    failure_rate_3d = len(quick_failures) / len(triggered)

    return SetupPerformanceStats(
        setup_name=setup_name, family=family, grade=grade, market_regime=market_regime,
        n_observations=n, trigger_rate=trigger_rate, win_rate=win_rate, expectancy_r=expectancy_r,
        median_bars_held=median_bars_held, mae_p80_pct=mae_p80_pct, failure_rate_3d=failure_rate_3d,
        confidence=_confidence_for_sample(n),
    )


def aggregate_setup_performance(observations: list[SetupReplayObservation]) -> list[SetupPerformanceStats]:
    """Agrupa por nombre de setup (la fila sin segmentar) y, "además"
    (Parte 10.2, literal - no en lugar de), por grado y por régimen de
    mercado por separado - no un cruce de ambos a la vez, que fragmentaría
    la muestra de casi cualquier setup muy por debajo de
    `MIN_SAMPLE_FOR_STATS` antes de que hubiera datos suficientes para
    decir nada de ninguna combinación."""
    by_name: dict[tuple[str, str], list[SetupReplayObservation]] = {}
    for obs in observations:
        by_name.setdefault((obs.setup_name, obs.family), []).append(obs)

    stats: list[SetupPerformanceStats] = []
    for (name, family), obs_list in by_name.items():
        stats.append(_stats_for_group(name, family, None, None, obs_list))

        grades = {o.grade for o in obs_list if o.grade is not None}
        for grade in sorted(grades):
            stats.append(_stats_for_group(name, family, grade, None, [o for o in obs_list if o.grade == grade]))

        regimes = {o.market_regime for o in obs_list if o.market_regime is not None}
        for regime in sorted(regimes):
            stats.append(
                _stats_for_group(name, family, None, regime, [o for o in obs_list if o.market_regime == regime])
            )
    return stats


def apply_measured_confidence(
    matches: list[SetupMatch], performance_by_name: dict[str, SetupPerformance]
) -> list[SetupMatch]:
    """Sustituye el `SetupConfidence.UNVALIDATED` con el que sale cada
    detector por la confianza medida de verdad en `setup_performance`
    (Parte 10.3), usando siempre la fila SIN segmentar de cada nombre de
    setup (`grade=None`, `market_regime=None`) - la pregunta de la 10.3
    ("¿hay muestra suficiente?") es sobre el propio setup, no sobre una
    combinación fina de grado/régimen; esa segmentación más fina es para
    el análisis y la interfaz (Parte 10.2: "información valiosa, no un
    defecto"), no para esta decisión binaria. Un nombre sin ninguna fila
    en `setup_performance` (el estudio nunca corrió, o el detector es
    nuevo) se queda tal cual - nunca se fabrica una medición."""
    if not performance_by_name:
        return matches
    result = []
    for match in matches:
        performance = performance_by_name.get(match.name)
        if performance is None:
            result.append(match)
            continue
        result.append(replace(match, confidence=SetupConfidence(performance.confidence)))
    return result
