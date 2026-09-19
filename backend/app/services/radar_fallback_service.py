"""Auditoria del Radar, bloque B: fallback de cómputo en vivo para que el
Radar nunca vuelva a estar vacío sin explicación. Excepción DELIBERADA y
documentada a la regla de CLAUDE.md "sin cómputo en el propio request para
Radar/Hoy" - ver docs/quant_methodology.md §29.1 para el porqué completo
(la misma regla que existe porque `PortfolioRiskService` ya sufrió un
incidente de latencia real por saltársela) y las salvaguardas que hacen
defendible esta excepción concreta:

1. Acotado a `DEFAULT_MAX_TICKERS` tickers, elegidos por liquidez (dollar
   volume medio 20d) - nunca el universo completo.
2. Reutiliza `ticker_daily_state_builder.build_ticker_daily_state` -
   EXACTAMENTE los mismos servicios que `daily_close.py`, cero lógica
   duplicada.
3. Aislado por ticker (mismo B2/B3 que `daily_close.py`) - un ticker malo
   no tumba el resto.
4. Con timeout: si supera `DEFAULT_TIMEOUT_SECONDS`, devuelve lo que tenga
   con `partial=True`, nunca deja al usuario mirando un spinner eterno.
5. Cacheado `CACHE_TTL` minutos por región - refrescar la página no dispara
   el cómputo otra vez.
6. NUNCA persiste nada en `ticker_daily_states` - un resultado en memoria,
   solo para esta respuesta. Escribirlo confundiría la señal de "¿corrió
   daily_close.py de verdad?" que esa tabla existe para responder.

**Limitación deliberada y documentada, no un hueco silencioso**: no llama
`MarketDataService.get_next_earnings_date` por ticker - eso SÍ sería la
llamada de red por ticker en el camino caliente que CLAUDE.md prohíbe sin
excepción (a diferencia de la descarga de OHLCV, que ya es una sola llamada
por lotes vía `get_universe_snapshot`/`get_cached_ohlcv`, pagada una vez
para todo el universo). Con 120 tickers, 120 llamadas de red secuenciales
agotarían el presupuesto de 25s por sí solas. Consecuencia real, no
oculta: `no_event_risk` no se evalúa para los candidatos del fallback (pasa
por defecto, `next_earnings_date=None`), y la penalización de earnings del
score compuesto tampoco puede aplicar - un valor que informa resultados
mañana puede aparecer en el fallback sin esa señal, algo que
`daily_close.py` sí captura de verdad."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from app.domain.models.setup_performance import SetupPerformance
from app.domain.models.ticker_daily_state import TickerDailyState
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import benchmark_for_region
from app.services.ticker_daily_state_builder import build_ticker_daily_state

logger = logging.getLogger(__name__)

CACHE_TTL = timedelta(minutes=15)
# "Hace más de 36 horas hábiles" (literal) se aproxima aquí a 36 horas de
# reloj sin más - cubre holgadamente un ciclo normal de refresco (~24h) y
# también un fin de semana largo sin que el cron haya corrido, que es
# justamente cuando esta red de seguridad debe activarse.
STALE_AFTER = timedelta(hours=36)
DEFAULT_MAX_TICKERS = 120
DEFAULT_TIMEOUT_SECONDS = 25.0
LIQUIDITY_LOOKBACK_DAYS = 20


def is_stale(computed_at: datetime | None, now: datetime, stale_after: timedelta = STALE_AFTER) -> bool:
    """`TickerDailyStateORM.computed_at` no declara `DateTime(timezone=True)`
    - tanto Postgres (producción) como SQLite (tests) devuelven un
    `datetime` *naive* al leerlo de vuelta, aunque se escribió con
    `datetime.now(UTC)` (aware). Todo el resto del código asume UTC
    implícitamente para estos campos - normalizar aquí explícitamente en
    vez de dejar que la resta falle (`TypeError: can't subtract
    offset-naive and offset-aware datetimes`) es más seguro que exigirle a
    cada llamador que lo sepa."""
    if computed_at is None:
        return True
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (now - computed_at) > stale_after


@dataclass(frozen=True, slots=True)
class LiveRadarSnapshot:
    """Resultado de una corrida del fallback - nunca persistido, ver el
    docstring del módulo."""

    states: list[TickerDailyState]
    analyzed: int  # cuántos tickers se procesaron de verdad (puede ser < universe por el tope o el timeout)
    universe: int  # tamaño del universo completo de la región, para el "coverage" honesto
    partial: bool  # `True` si el timeout cortó la corrida antes de terminar
    computed_at: datetime


class RadarFallbackService:
    """Cacheado como singleton (ver `api/deps.py`, mismo patrón que
    `MarketScreenerService`) - el caché de `CACHE_TTL` solo sirve de algo si
    la misma instancia sobrevive entre requests."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[datetime, LiveRadarSnapshot]] = {}

    def get_or_compute(
        self,
        region: str,
        market_data: MarketDataService,
        screener: MarketScreenerService,
        setup_performance_by_name: dict[str, SetupPerformance],
        max_tickers: int = DEFAULT_MAX_TICKERS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> LiveRadarSnapshot:
        cached = self._cache.get(region)
        now = datetime.now(UTC)
        if cached is not None and (now - cached[0]) < CACHE_TTL:
            return cached[1]
        snapshot = self._compute(
            region, market_data, screener, setup_performance_by_name, max_tickers, timeout_seconds
        )
        self._cache[region] = (now, snapshot)
        return snapshot

    def _compute(
        self,
        region: str,
        market_data: MarketDataService,
        screener: MarketScreenerService,
        setup_performance_by_name: dict[str, SetupPerformance],
        max_tickers: int,
        timeout_seconds: float,
    ) -> LiveRadarSnapshot:
        started = time.monotonic()
        now = datetime.now(UTC)
        trade_date = date.today()

        # Sin `force_refresh`: la caché durable (hasta 3h) es aceptable aquí
        # - a diferencia de `daily_close.py`, que SÍ necesita el cierre
        # exacto (bloque B1), este es un cálculo de emergencia, no "el
        # cierre oficial del día".
        universe = screener.get_universe_snapshot(region)
        ohlcv_by_ticker = screener.get_cached_ohlcv(region)

        def liquidity(ticker: str) -> float:
            df = ohlcv_by_ticker.get(ticker)
            if df is None or len(df) < LIQUIDITY_LOOKBACK_DAYS:
                return 0.0
            recent = df.iloc[-LIQUIDITY_LOOKBACK_DAYS:]
            return float((recent["close"] * recent["volume"]).mean())

        ranked = sorted(universe, key=lambda ts: liquidity(ts.ticker), reverse=True)
        top = ranked[:max_tickers]

        benchmark_df = ohlcv_by_ticker.get(benchmark_for_region(region))
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None

        states: list[TickerDailyState] = []
        partial = False
        for ts in top:
            if (time.monotonic() - started) > timeout_seconds:
                partial = True
                break
            df = ohlcv_by_ticker.get(ts.ticker)
            if df is None:
                continue
            try:
                # Sin `next_earnings_date` - ver el docstring del módulo
                # para el porqué (nunca una llamada de red por ticker aquí).
                state = build_ticker_daily_state(
                    ts, region, df, trade_date, now, None, benchmark_close, setup_performance_by_name
                )
                if state is not None:
                    states.append(state)
            except Exception:
                logger.exception("radar_fallback: fallo procesando %s (%s)", ts.ticker, region)

        return LiveRadarSnapshot(
            states=states, analyzed=len(states), universe=len(universe), partial=partial, computed_at=now
        )
