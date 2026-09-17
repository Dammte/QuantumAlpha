"""`SetupContext` - todo lo que un detector de `setups/*.py` puede necesitar,
ya calculado por quien construye el contexto (Fase 2: `daily_close.py`).

Regla de diseño no negociable: ningún detector recibe una serie OHLCV cruda
y la vuelve a procesar por su cuenta. Si un detector necesita el remuestreo
semanal, el ATR, los niveles (`technical_analysis.detect_levels`) o el sesgo
multi-timeframe, los lee de aquí - nunca llama `resample_ohlcv`/`atr`/
`detect_levels`/`analyze_multi_timeframe` por su cuenta. Esto es lo que
mantiene el presupuesto de 40 ms/ticker (ver `setups/__init__.py`): el coste
real de esas funciones ya está pagado por el gate/la geometría/los niveles
existentes antes de que el primer detector se ejecute.

`SetupContext` es `kw_only` a propósito - tiene más de una docena de campos
y una llamada posicional sería ilegible y frágil ante reordenaciones."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta


@dataclass(frozen=True, slots=True, kw_only=True)
class SetupContext:
    ticker: str
    region: str
    trade_date: date

    # OHLCV diario, ya recortado a barras cerradas por el llamador (el mismo
    # frame que `daily_close.py::build_ticker_daily_state` ya recibe - ese
    # job corre después del cierre, así que "diario" aquí ya es "cerrado").
    close: pd.Series
    high: pd.Series
    low: pd.Series
    volume: pd.Series
    open_: pd.Series

    # Semanal, remuestreado una sola vez (`ta.resample_ohlcv`) por quien
    # construye el contexto - `None` solo cuando no hay ni una semana cerrada
    # todavía (mismo criterio que `MultiTimeframeRead.weekly`).
    weekly_close: pd.Series | None
    weekly_high: pd.Series | None
    weekly_low: pd.Series | None
    weekly_volume: pd.Series | None

    atr_series: pd.Series
    atr14: float | None
    ema21: float | None
    ema55: float | None
    sma20: float | None
    sma50: float | None
    sma150: float | None
    sma200: float | None
    rsi14: float | None

    # El motor de niveles real (Parte 5.1) - ya trae estado/duración
    # (FAR/APPROACHING/TESTING/BREAKING/BROKEN_CONFIRMED/LOST_CONFIRMED) por
    # nivel, así que `breakout.py`/`pullback.py` lo leen directamente en vez
    # de recalcular distancias a mano.
    levels: list[ta.Level]

    multi_timeframe: mtf.MultiTimeframeRead
    trend: ta.TrendState
    weekly_stage: ta.Stage | None

    relative_volume: float | None
    rs_percentile: int | None
    sector_rs_percentile: int | None

    # Serie diaria de RS de Mansfield a 20 sesiones (`ta.mansfield_rs(close,
    # benchmark_close, window=20)` - Parte 2.2's "RS de Mansfield a 20
    # sesiones"), no el escalar de 200 sesiones que ya muestra "Analizar
    # activo". `None` sin un cierre de benchmark disponible para la región -
    # `stage_transition.py`'s subestado `stage1_rs_turning` simplemente no
    # se evalúa en ese caso, nunca se fabrica.
    mansfield_rs_series: pd.Series | None
