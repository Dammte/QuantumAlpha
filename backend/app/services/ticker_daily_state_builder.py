"""Auditoria del Radar, bloque B: `build_ticker_daily_state` extraído de
`scripts/daily_close.py` a un módulo de `app/services/` - "el cómputo en
vivo reutiliza exactamente los mismos servicios que daily_close.py...
prohibido duplicar lógica: si hace falta, extrae la función común a un
módulo que llamen los dos" (literal). `scripts/daily_close.py` (el cron
nocturno) y `radar_fallback_service.py` (el fallback en vivo, bloque B) son
ahora los dos llamadores de la misma función, sin ninguna copia.

Un `script` no puede ser importado por un `service` sin invertir la
dirección de dependencias que CLAUDE.md establece (`domain` → `services` →
`infrastructure` → `api`, con `scripts/` como capa de entrada externa que
depende de todas las demás, nunca al revés) - de ahí que esta función viva
aquí, no que el fallback importe `scripts.daily_close`."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from app.domain.models.setup_performance import SetupPerformance
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import dynamic_universe_service as dus
from app.services import levels_engine as le
from app.services import multi_timeframe as mtf
from app.services import setup_replay
from app.services import technical_analysis as ta
from app.services.setups import arbitration as setups_arbitration
from app.services.setups import context_modifiers as setups_context_modifiers
from app.services.setups import horizon as setups_horizon
from app.services.setups import registry as setups_registry
from app.services.setups.context import SetupContext
from app.services.setups.types import setup_match_to_dict
from app.services.ticker_analysis_service import MIN_BARS_REQUIRED
from app.services.trade_geometry import geometry_to_dict


def _nearest_level(levels: list[ta.PriceLevel], kind: str) -> ta.PriceLevel | None:
    candidates = [lv for lv in levels if lv.kind == kind]
    return min(candidates, key=lambda lv: abs(lv.distance_pct)) if candidates else None


def build_ticker_daily_state(
    snapshot: TickerSnapshot,
    region: str,
    df: pd.DataFrame,
    trade_date: date,
    computed_at: datetime,
    next_earnings_date: date | None = None,
    benchmark_close: pd.Series | None = None,
    setup_performance_by_name: dict[str, SetupPerformance] | None = None,
) -> TickerDailyState | None:
    """Pure function (`next_earnings_date` is the one exception - a value
    the caller already paid the network cost for, never fetched here):
    everything `levels_engine.evaluate_gate` needs beyond what `TickerSnapshot`
    already carries (raw ATR, support/resistance, the fast-pair veto, the
    weekly Weinstein Stage, the liquidity floor) is derived here from the
    same OHLCV frame the screener already downloaded - no new network call,
    no re-fetch. `None` when there isn't enough history to say anything (same
    bar `ticker_analysis_service.compute_core_signals` uses).

    Sexta auditoría (Parte 6.2, texto literal completo): the gate's 5
    eligibility criteria replace the previous 6-condition approximation -
    see `levels_engine.py`'s own module docstring for the full reasoning."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    if len(close) < MIN_BARS_REQUIRED:
        return None

    atr_series = ta.atr(high, low, close)
    atr14 = float(atr_series.iloc[-1]) if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else None
    levels = ta.support_resistance_levels(high, low, close)
    nearest_support = _nearest_level(levels, "support")
    nearest_resistance = _nearest_level(levels, "resistance")
    fast_pair_veto = ta.detect_fast_pair_bearish_veto(close)

    # Parte 5.5: la MA30 semanal real (no la proxy diaria de `snapshot.stage`,
    # que sigue siendo el Stage *diario* mostrado en el Radar/screener como
    # contexto, sin cambios) - `weekly_not_stage4` del gate exige el
    # semanal genuino.
    multi_timeframe = mtf.analyze_multi_timeframe(df)
    weekly_stage = multi_timeframe.weekly.stage if multi_timeframe.weekly is not None else None
    # Parte 7 (§28.x): puramente informativa - `evaluate_gate`/`compute_grade`
    # de aquí abajo no reciben `timeframe_strip` como argumento, ni lo van a
    # recibir nunca (ver el test de aislamiento en test_levels_engine.py).
    timeframe_strip_dict = mtf.timeframe_strip_to_dict(mtf.build_timeframe_strip(df, multi_timeframe))

    # Divisa nativa, sin conversión a USD - misma simplificación documentada
    # en `ticker_analysis_service.compute_core_signals` (el universo dinámico
    # mensual, Job C, ya filtra por liquidez en USD real al entrar; esta es
    # la comprobación diaria adicional del propio gate, Parte 6.2).
    dollar_volume_20d = float((close.iloc[-20:] * volume.iloc[-20:]).mean()) if len(close) >= 20 else None
    liquidity_ok = dus.passes_liquidity_floor(snapshot.price, dollar_volume_20d)

    # Parte 7 (later pass): the same real EMA21/55 read
    # `ticker_analysis_service.compute_core_signals` already passes to
    # `evaluate_gate` for "Analizar activo"/`/risk` - cheap off the same
    # `close` series already in memory here, no new network call. Without
    # these two, `gate.entry_geometry` comes back `None` (see that
    # function's own docstring) and this job would keep persisting nothing
    # for Parte 7's real design, same gap CLAUDE.md used to flag.
    ema21_raw = ta.ema(close, mtf.FAST_MA_PERIOD).iloc[-1] if len(close) else None
    ema55_raw = ta.ema(close, mtf.SLOW_MA_PERIOD).iloc[-1] if len(close) else None
    ema21 = None if ema21_raw is None or pd.isna(ema21_raw) else float(ema21_raw)
    ema55 = None if ema55_raw is None or pd.isna(ema55_raw) else float(ema55_raw)

    # Biblioteca de setups del Radar (en curso, ver quant_methodology.md §28):
    # el mismo remuestreo semanal en memoria que `ticker_analysis_service.compute_core_signals`
    # ya paga por separado (§27.11) para darle a `detect_levels` un
    # `weekly_close` real - un segundo `resample_ohlcv` aquí, sobre el mismo
    # `df` ya en memoria, nunca una llamada de red nueva.
    weekly_df = ta.resample_ohlcv(df, mtf.WEEKLY_RULE)
    weekly_close = weekly_df["close"] if len(weekly_df) >= 2 else None
    weekly_high = weekly_df["high"] if len(weekly_df) >= 2 else None
    weekly_low = weekly_df["low"] if len(weekly_df) >= 2 else None
    weekly_volume = weekly_df["volume"] if len(weekly_df) >= 2 else None
    # `detect_levels` (Parte 5.1/§27.5) es aditivo junto a `support_resistance_levels`
    # de arriba, no un reemplazo - el gate sigue leyendo `nearest_support`/
    # `nearest_resistance` (`PriceLevel`, el tipo simple); esta lista rica
    # (`Level`, con estado FAR/APPROACHING/TESTING/BREAKING/...) es solo para
    # que los detectores de setups la lean sin recalcular nada.
    setup_levels = ta.detect_levels(high, low, close, volume, weekly_close=weekly_close)
    # Parte 2.2 (`stage1_rs_turning`): RS de Mansfield a 20 sesiones, sobre
    # el benchmark real de la región - `None` sin uno disponible (el
    # subestado simplemente no se evalúa, nunca se fabrica).
    mansfield_rs_series = (
        ta.mansfield_rs(close, benchmark_close, window=20) if benchmark_close is not None else None
    )

    setup_ctx = SetupContext(
        ticker=snapshot.ticker,
        region=region,
        trade_date=trade_date,
        close=close,
        high=high,
        low=low,
        volume=volume,
        open_=df["open"],
        weekly_close=weekly_close,
        weekly_high=weekly_high,
        weekly_low=weekly_low,
        weekly_volume=weekly_volume,
        atr_series=atr_series,
        atr14=atr14,
        ema21=ema21,
        ema55=ema55,
        sma20=snapshot.sma20,
        sma50=snapshot.sma50,
        sma150=snapshot.sma150,
        sma200=snapshot.sma200,
        rsi14=snapshot.rsi14,
        levels=setup_levels,
        multi_timeframe=multi_timeframe,
        trend=snapshot.trend,
        weekly_stage=weekly_stage,
        relative_volume=snapshot.relative_volume,
        rs_percentile=snapshot.rs_rating,
        sector_rs_percentile=snapshot.sector_rs_percentile,
        mansfield_rs_series=mansfield_rs_series,
    )
    # `stage_transition.py` (Fase 3) es, por ahora, el único detector
    # registrado en `SETUP_DETECTORS` - esta llamada devolvía siempre `[]`
    # hasta entonces (§28.2). Cablearla desde antes de tener ningún detector
    # real dejó que los tests de este job cubrieran la construcción de
    # `SetupContext` contra datos de verdad, no solo el stub sintético de
    # `test_setups_registry.py`.
    # `[]`, no `None`: "sin coincidencias hoy" es un resultado real y
    # esperado (la mayoría de tickers la mayoría de días no cumplen ningún
    # setup), mismo criterio que `gate_conditions` - `None` queda reservado
    # para una fila anterior a esta columna, no para "no se encontró nada".
    # `order_by_rank` (Parte 0/6, §28.8) deja el setup titular en el índice
    # 0 - "el mejor gana, los demás se muestran como contexto" - sin
    # descartar ninguno.
    ordered_setups = setups_arbitration.order_by_rank(setups_registry.detect_all(setup_ctx))
    # Parte 10.3 (§28.x): sustituye el UNVALIDATED con el que sale cada
    # detector por la confianza medida de verdad en `setup_performance`,
    # cuando el estudio (`scripts/setup_replay_study.py`) ya corrió para
    # ese nombre de setup - `{}` (el valor por defecto) dejaría a todos en
    # UNVALIDATED, el mismo comportamiento honesto de antes de que esta
    # tabla existiera, nunca un error.
    ordered_setups = setup_replay.apply_measured_confidence(ordered_setups, setup_performance_by_name or {})
    # Auditoria del Radar, bloque D: horizonte corto/medio plazo - paso
    # posterior a la detección, igual que los dos de arriba, ver
    # `setups/horizon.py` para el criterio completo.
    ordered_setups = setups_horizon.assign_horizon(ordered_setups, setup_ctx)
    setups_list = [setup_match_to_dict(m) for m in ordered_setups]

    gate = le.evaluate_gate(
        price=snapshot.price,
        trend=snapshot.trend,
        atr14=atr14,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        weekly_stage=weekly_stage,
        liquidity_ok=liquidity_ok,
        fast_pair_bearish_signal=fast_pair_veto,
        next_earnings_date=next_earnings_date,
        as_of=trade_date,
        ema21=ema21,
        ema55=ema55,
        levels=setup_levels,
    )
    trigger = gate.entry_trigger
    stop_target = gate.stop_and_target

    # Parte 5.3 (later pass): the exact same "is there even a viable
    # geometry to grade" precondition `ticker_analysis_service.compute_core_signals`
    # uses for "Analizar activo" - see levels_engine.compute_grade's own
    # docstring for why `grade=None` (never a fabricated grade on top of a
    # non-viable trade). `snapshot.relative_volume`/`sma200`/`rs_rating`/
    # `sector_rs_percentile` are already computed by the screener for this
    # exact ticker - no re-derivation, no new network call.
    grade_dict: dict | None = None
    if gate.entry_trigger is not None and gate.entry_geometry is not None and gate.entry_geometry.viable:
        weekly_bullish = mtf.timeframe_bias(multi_timeframe.weekly) == "bullish"
        grade_result = le.compute_grade(
            price=snapshot.price,
            atr14=atr14,
            entry_trigger=gate.entry_trigger,
            geometry=gate.entry_geometry,
            weekly_bullish=weekly_bullish,
            relative_volume=snapshot.relative_volume,
            rs_percentile=snapshot.rs_rating,
            sma200=snapshot.sma200,
            sector_rs_percentile=snapshot.sector_rs_percentile,
        )
        # Parte 6 (§28.x): los modificadores de contexto de la biblioteca de
        # setups ajustan este MISMO grado - no un concepto paralelo - con la
        # lista de setups que este ticker ya calculó arriba
        # (`ordered_setups`). No necesitan cartera, así que corren aquí, no
        # en el endpoint de lectura (a diferencia de
        # `apply_portfolio_grade_modifiers`).
        grade_result = setups_context_modifiers.apply_context_modifiers(grade_result, setup_ctx, ordered_setups)
        grade_dict = {
            "grade": grade_result.grade.value if grade_result.grade is not None else None,
            "reasons": grade_result.reasons,
            "distance_atr": grade_result.distance_atr,
        }

    return TickerDailyState(
        id=None,
        region=region,
        ticker=snapshot.ticker,
        trade_date=trade_date,
        computed_at=computed_at,
        price=snapshot.price,
        currency=snapshot.currency,
        trend=snapshot.trend.value,
        stage=snapshot.stage.value if snapshot.stage else None,
        rs_rating=snapshot.rs_rating,
        adx14=snapshot.adx14,
        atr_multiple=snapshot.atr_multiple,
        rsi14=snapshot.rsi14,
        gate_passes=gate.passes,
        gate_conditions=[{"label": c.label, "passed": c.passed} for c in gate.conditions],
        gate_version=le.GATE_VERSION,
        entry_trigger_type=trigger.trigger_type if trigger else None,
        entry_trigger_price=trigger.trigger_price if trigger else None,
        entry_already_triggered=trigger.already_triggered if trigger else False,
        stop_loss=stop_target.stop_loss,
        take_profit=stop_target.take_profit,
        take_profit_method=stop_target.take_profit_method,
        risk_reward=stop_target.risk_reward,
        entry_geometry=geometry_to_dict(gate.entry_geometry) if gate.entry_geometry is not None else None,
        grade=grade_dict,
        setups=setups_list,
        timeframe_strip=timeframe_strip_dict,
        sector=snapshot.sector,
        sector_rs_percentile=snapshot.sector_rs_percentile,
        relative_volume=snapshot.relative_volume,
        next_earnings_date=next_earnings_date,
        atr_pct=(atr14 / snapshot.price) if atr14 and snapshot.price else None,
    )
