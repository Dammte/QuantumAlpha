"""Parte 4.4 del encargo: canales por regresión lineal.

Advertencia de diseño que el propio encargo hace explícita: cualquier
paseo aleatorio ajusta razonablemente bien a un canal sin un filtro
estadístico real detrás - `technical_analysis.linear_regression_fit`
(compartida con `classic_patterns.py`, fase posterior) exige tanto un
t-estadístico de la pendiente (|t| >= `CHANNEL_MIN_ABS_T_STAT`, rechaza el
ruido puro) como un R² mínimo (rechaza una recta con pendiente "real" pero
que apenas explica el precio) - la combinación hace más trabajo que
cualquiera de los dos criterios solo.

**Mismo patrón de "no midas una ventana con la propia barra que estás
juzgando" que ya corrigió un bug real en `stage_transition.py` (§28.3) y
`breakout.py` (§28.6)**: el canal se ajusta sobre las
`CHANNEL_LOOKBACK` sesiones ANTERIORES a hoy, nunca incluyendo la barra de
hoy - las bandas se proyectan un paso más allá del ajuste, y el precio de
hoy se compara contra esa proyección. Si el ajuste incluyera hoy, un
movimiento fuerte de hoy simplemente "ensancharía las bandas" del propio
ajuste que debía juzgarlo, haciendolo estructuralmente indetectable el mismo
día en que ocurre."""

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

CHANNEL_LOOKBACK = 60
CHANNEL_MIN_ABS_T_STAT = 2.0
CHANNEL_MIN_R2 = 0.55
CHANNEL_BAND_STD_MULTIPLE = 2.0
CHANNEL_LOWER_BAND_FRACTION = 0.20  # "banda inferior" - percentil 20 del canal


def _fit_channel(close) -> ta.RegressionFit | None:
    if len(close) < CHANNEL_LOOKBACK + 1:
        return None
    window = close.iloc[-CHANNEL_LOOKBACK - 1 : -1]  # excluye la barra de hoy
    fit = ta.linear_regression_fit(window)
    if fit is None:
        return None
    if abs(fit.t_stat) < CHANNEL_MIN_ABS_T_STAT:
        return None
    if fit.r_squared < CHANNEL_MIN_R2:
        return None
    return fit


def _projected_band(fit: ta.RegressionFit) -> tuple[float, float]:
    """Las bandas, proyectadas UN paso más allá de la ventana ajustada - la
    posición de "hoy", que nunca formó parte del propio ajuste."""
    today_x = float(CHANNEL_LOOKBACK)
    predicted = fit.slope * today_x + fit.intercept
    half_width = CHANNEL_BAND_STD_MULTIPLE * fit.residual_std
    return predicted - half_width, predicted + half_width


def _check_rising_channel_pullback(ctx: SetupContext, fit: ta.RegressionFit) -> SetupMatch | None:
    if fit.slope <= 0:
        return None
    lower, upper = _projected_band(fit)
    if upper <= lower:
        return None
    price = float(ctx.close.iloc[-1])
    position_frac = (price - lower) / (upper - lower)
    if position_frac > CHANNEL_LOWER_BAND_FRACTION:
        return None
    if position_frac < 0:
        return None  # ya rompió por debajo de la banda inferior - canal invalidado, no un retroceso sano

    evidence = {
        "r_squared": round(fit.r_squared, 3),
        "t_stat": round(fit.t_stat, 2),
        "position_in_channel_pct": round(position_frac * 100, 1),
        "lower_band": round(lower, 4),
        "upper_band": round(upper, 4),
    }
    return SetupMatch(
        family=SetupFamily.CHANNEL,
        name="canal_alcista_banda_inferior",
        label_es="Retroceso en canal alcista",
        stage=SetupStage.READY,
        bars_in_stage=1,
        timeframe="daily",
        trigger_price=None,
        trigger_condition="giro al alza dentro del canal, tras tocar la banda inferior",
        invalidation_price=lower,
        invalidation_condition=f"cierre por debajo de la banda inferior ({lower:.2f}) invalida el canal",
        evidence=evidence,
        narrative_es=(
            f"Canal alcista limpio (R²={fit.r_squared:.2f}, t={fit.t_stat:.1f}) - el precio ha retrocedido "
            "hasta la banda inferior. Geometría de retroceso dentro de tendencia."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def _check_falling_channel_breakout(ctx: SetupContext, fit: ta.RegressionFit) -> SetupMatch | None:
    if fit.slope >= 0:
        return None
    # "solo se emite si además el sesgo semanal ha dejado de ser bajista" -
    # Parte 4.4, literal. Sin esta condición sería contratendencia pura.
    if mtf.timeframe_bias(ctx.multi_timeframe.weekly) == "bearish":
        return None
    lower, upper = _projected_band(fit)
    price = float(ctx.close.iloc[-1])
    if price <= upper:
        return None

    evidence = {
        "r_squared": round(fit.r_squared, 3),
        "t_stat": round(fit.t_stat, 2),
        "upper_band": round(upper, 4),
    }
    return SetupMatch(
        family=SetupFamily.CHANNEL,
        name="ruptura_canal_bajista",
        label_es="Ruptura de canal bajista",
        stage=SetupStage.TRIGGERED,
        bars_in_stage=1,
        timeframe="daily",
        trigger_price=upper,
        trigger_condition=f"cierre por encima de la banda superior del canal bajista ({upper:.2f})",
        invalidation_price=lower,
        invalidation_condition=f"cierre de vuelta por debajo de {lower:.2f} devuelve al canal bajista",
        evidence=evidence,
        narrative_es=(
            f"Ruptura de la banda superior de un canal bajista (R²={fit.r_squared:.2f}) con el sesgo "
            "semanal ya no bajista - posible cambio de tendencia."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def detect(ctx: SetupContext) -> list[SetupMatch]:
    fit = _fit_channel(ctx.close)
    if fit is None:
        return []
    match = _check_rising_channel_pullback(ctx, fit) or _check_falling_channel_breakout(ctx, fit)
    return [match] if match is not None else []
