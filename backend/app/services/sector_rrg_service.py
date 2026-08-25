"""Cuarta auditoría independiente, Bloque E (nuevo módulo, explícitamente
pedido - "reconstrucción del cuadrante RRG de rotación sectorial", señalado
como pendiente desde `watchlist_service.py`/`relationship_map_service.py` en
rondas anteriores).

Un Relative Rotation Graph (RRG, Julius de Kempenaer) sitúa cada sector en un
plano de dos ejes frente a un benchmark:

- **RS-Ratio**: fuerza relativa normalizada (>100 = el sector supera al
  benchmark ahora mismo, <100 = lo hace peor).
- **RS-Momentum**: el *ritmo de cambio* de esa fuerza relativa (>100 =
  ganando impulso, <100 = perdiéndolo).

Cruzando ambos ejes salen cuatro cuadrantes - y, a diferencia de
`sector_rotation_service.py` (que solo mira el ranking de fuerza relativa de
hoy contra un patrón histórico de ciclo económico), el RRG añade el eje de
*momentum*, distinguiendo un sector que lidera y sigue acelerando de uno que
lidera pero ya está perdiendo fuelle - la vieja información de
`sector_rotation_service.py` sigue siendo útil (y no se toca ni se
sustituye), esto es un eje nuevo que esa función nunca tuvo:

- **Leading** (RS-Ratio ≥ 100, RS-Momentum ≥ 100): liderando y acelerando.
- **Weakening** (RS-Ratio ≥ 100, RS-Momentum < 100): todavía líder, pero ya
  frenando - la rotación clásica sale de aquí hacia "lagging".
- **Lagging** (RS-Ratio < 100, RS-Momentum < 100): rezagado y sin mejorar.
- **Improving** (RS-Ratio < 100, RS-Momentum ≥ 100): rezagado pero ganando
  impulso - la rotación clásica entra a "leading" pasando por aquí.

Un RRG real rota en el sentido horario a lo largo de un ciclo completo
(leading → weakening → lagging → improving → leading) - de ahí que cada
lectura incluya una "cola" (`tail`) de puntos recientes: la posición actual
sin la trayectoria no dice si un sector "weakening" lleva ahí semanas
(genuina rotación en marcha) o acaba de entrar (todavía podría revertir).

**Nota de honestidad**: la normalización exacta que StockCharts/JdK usa en su
implementación comercial no es pública. Esta es una normalización estándar,
ampliamente usada en implementaciones abiertas del concepto (z-score móvil
del ratio de fuerza relativa, y de su propio ritmo de cambio, centrados en
100) - reproduce el comportamiento cualitativo del RRG (los cuatro
cuadrantes, la rotación en el tiempo), no pretende ser un clon numérico
exacto del producto comercial.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

RRG_HISTORY_YEARS = 2  # ~500 sesiones - suficiente para el warmup doble que compute_rs_ratio_and_momentum necesita
RRG_SMOOTHING_SPAN = 10  # EMA ligera sobre el ratio crudo, reduce ruido día a día sin ocultar el movimiento real
RRG_NORMALIZATION_WINDOW = 100  # ventana móvil para el z-score de RS-Ratio y, después, el de RS-Momentum
RRG_MOMENTUM_WINDOW = 10  # sesiones sobre las que se mide el "ritmo de cambio" del RS-Ratio ya normalizado
RRG_SCALE = 10.0  # escala el z-score a un rango visualmente legible (~90-110), como un RRG publicado de verdad
RRG_TAIL_LENGTH = 10  # puntos de cola mostrados - suficiente para ver la trayectoria sin saturar el gráfico
MIN_OBSERVATIONS = RRG_NORMALIZATION_WINDOW * 2 + RRG_MOMENTUM_WINDOW  # el warmup doble real, con margen

QUADRANT_LEADING = "leading"
QUADRANT_WEAKENING = "weakening"
QUADRANT_LAGGING = "lagging"
QUADRANT_IMPROVING = "improving"


def _quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100:
        return QUADRANT_LEADING if rs_momentum >= 100 else QUADRANT_WEAKENING
    return QUADRANT_IMPROVING if rs_momentum >= 100 else QUADRANT_LAGGING


def compute_rs_ratio_and_momentum(
    sector_close: pd.Series, benchmark_close: pd.Series
) -> tuple[pd.Series, pd.Series] | None:
    """The full RS-Ratio/RS-Momentum series (not just the latest point) - a
    caller wanting the "tail" trajectory needs the history, not one number.
    `None` when there isn't enough overlapping history to clear the double
    rolling-window warmup (RS-Ratio's own 100-session z-score, then
    RS-Momentum's 100-session z-score *of that already-normalized series*).

    A rolling standard deviation of exactly 0 (a genuinely flat ratio over
    the whole window - degenerate, but not impossible for a thinly-traded
    ETF pair) is replaced with NaN before dividing, so it propagates as an
    honest "undefined" data point instead of a raw `inf` that `.dropna()`
    would silently let through (same discipline as
    `relationship_map_service.compute_statistical_relations`'s own NaN
    guard)."""
    aligned = pd.DataFrame({"sector": sector_close, "benchmark": benchmark_close}).dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        return None

    raw_rs = aligned["sector"] / aligned["benchmark"]
    smoothed_rs = raw_rs.ewm(span=RRG_SMOOTHING_SPAN, adjust=False).mean()

    rolling_mean = smoothed_rs.rolling(RRG_NORMALIZATION_WINDOW).mean()
    rolling_std = smoothed_rs.rolling(RRG_NORMALIZATION_WINDOW).std().replace(0, np.nan)
    rs_ratio = 100 + (smoothed_rs - rolling_mean) / rolling_std * RRG_SCALE

    momentum_raw = rs_ratio.diff(RRG_MOMENTUM_WINDOW)
    momentum_mean = momentum_raw.rolling(RRG_NORMALIZATION_WINDOW).mean()
    momentum_std = momentum_raw.rolling(RRG_NORMALIZATION_WINDOW).std().replace(0, np.nan)
    rs_momentum = 100 + (momentum_raw - momentum_mean) / momentum_std * RRG_SCALE

    combined = pd.DataFrame({"rs_ratio": rs_ratio, "rs_momentum": rs_momentum}).dropna()
    if combined.empty:
        return None
    return combined["rs_ratio"], combined["rs_momentum"]


@dataclass(frozen=True, slots=True)
class RrgPoint:
    as_of: date
    rs_ratio: float
    rs_momentum: float


@dataclass(frozen=True, slots=True)
class SectorRrgReading:
    sector: str
    etf: str
    quadrant: str  # "leading" | "weakening" | "lagging" | "improving"
    rs_ratio: float
    rs_momentum: float
    # Trailing RRG_TAIL_LENGTH points, oldest first - the trajectory a real
    # RRG chart plots as a tail behind each sector's current dot.
    tail: list[RrgPoint]


def compute_sector_rrg(
    sector_etfs: dict[str, str], ohlcv_by_ticker: dict[str, pd.DataFrame], benchmark_ticker: str
) -> list[SectorRrgReading]:
    """One reading per sector whose ETF has enough history against
    `benchmark_ticker` - sectors without a liquid ETF, or too little shared
    history, are simply absent (never a fabricated placeholder reading).
    Sorted by RS-Ratio descending, so the strongest-right-now sector leads
    the list regardless of which quadrant it's in."""
    benchmark_df = ohlcv_by_ticker.get(benchmark_ticker)
    if benchmark_df is None or benchmark_df.empty:
        return []
    benchmark_close = benchmark_df["close"]

    readings = []
    for sector, etf in sector_etfs.items():
        df = ohlcv_by_ticker.get(etf)
        if df is None or df.empty:
            continue
        result = compute_rs_ratio_and_momentum(df["close"], benchmark_close)
        if result is None:
            continue
        rs_ratio, rs_momentum = result

        tail_ratio = rs_ratio.iloc[-RRG_TAIL_LENGTH:]
        tail_momentum = rs_momentum.iloc[-RRG_TAIL_LENGTH:]
        tail = [
            RrgPoint(as_of=idx.date(), rs_ratio=float(r), rs_momentum=float(m))
            for idx, r, m in zip(tail_ratio.index, tail_ratio, tail_momentum, strict=True)
        ]
        latest = tail[-1]
        readings.append(
            SectorRrgReading(
                sector=sector, etf=etf, quadrant=_quadrant(latest.rs_ratio, latest.rs_momentum),
                rs_ratio=latest.rs_ratio, rs_momentum=latest.rs_momentum, tail=tail,
            )
        )

    readings.sort(key=lambda r: r.rs_ratio, reverse=True)
    return readings
