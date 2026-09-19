"""Auditoria del Radar, bloque B (docs/quant_methodology.md §29): el
fallback de cómputo en vivo. Integración real (mismo `FakeMarketDataProvider`
determinista que `test_daily_close_job.py` ya usa) porque este servicio
depende de `MarketScreenerService`/`build_ticker_daily_state` de verdad, no
solo de su propia lógica de caché/tope/timeout."""

from datetime import UTC, datetime, timedelta

import app.services.radar_fallback_service as rfs
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from tests.integration.conftest import FakeMarketDataProvider


def _service_and_screener() -> tuple[MarketDataService, MarketScreenerService]:
    market_data = MarketDataService(FakeMarketDataProvider())
    screener = MarketScreenerService(market_data)
    return market_data, screener


def test_is_stale_true_when_never_computed():
    assert rfs.is_stale(None, datetime.now(UTC)) is True


def test_is_stale_false_within_the_window():
    now = datetime.now(UTC)
    assert rfs.is_stale(now - timedelta(hours=1), now) is False


def test_is_stale_true_beyond_the_window():
    now = datetime.now(UTC)
    assert rfs.is_stale(now - timedelta(hours=40), now) is True


def test_compute_produces_states_for_at_least_some_of_the_universe():
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    snapshot = service.get_or_compute("us", market_data, screener, {})

    assert snapshot.universe > 0
    assert snapshot.analyzed > 0
    assert len(snapshot.states) == snapshot.analyzed
    assert not snapshot.partial


def test_compute_never_calls_next_earnings_date_per_ticker():
    # Bloque B: nunca una llamada de red por ticker en el propio request -
    # cada estado del fallback sale sin fecha de earnings.
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    snapshot = service.get_or_compute("us", market_data, screener, {})

    assert all(s.next_earnings_date is None for s in snapshot.states)


def test_compute_respects_the_max_tickers_bound():
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    snapshot = service.get_or_compute("us", market_data, screener, {}, max_tickers=3)

    assert snapshot.analyzed <= 3
    assert snapshot.universe > 3  # el universo real de prueba tiene más de 3 tickers


def test_compute_marks_partial_when_the_timeout_is_exhausted_immediately():
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    snapshot = service.get_or_compute("us", market_data, screener, {}, timeout_seconds=0.0)

    assert snapshot.partial is True
    assert snapshot.analyzed == 0


def test_get_or_compute_caches_within_the_ttl(monkeypatch):
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    first = service.get_or_compute("us", market_data, screener, {})
    second = service.get_or_compute("us", market_data, screener, {})

    assert first is second  # mismo objeto - no se recalculó


def test_get_or_compute_recomputes_after_the_ttl_expires():
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    first = service.get_or_compute("us", market_data, screener, {})
    # Simula que el caché ya caducó, adelantando su marca de tiempo.
    cached_at, snapshot = service._cache["us"]
    service._cache["us"] = (cached_at - rfs.CACHE_TTL - timedelta(seconds=1), snapshot)

    second = service.get_or_compute("us", market_data, screener, {})

    assert second is not first


def test_get_or_compute_caches_separately_per_region():
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    us_snapshot = service.get_or_compute("us", market_data, screener, {})
    europe_snapshot = service.get_or_compute("europe", market_data, screener, {})

    assert us_snapshot is not europe_snapshot


def test_compute_isolates_a_ticker_that_raises(monkeypatch):
    market_data, screener = _service_and_screener()
    service = rfs.RadarFallbackService()

    real_build = rfs.build_ticker_daily_state
    call_count = {"n": 0}

    def _flaky_build(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("dato corrupto simulado")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(rfs, "build_ticker_daily_state", _flaky_build)

    snapshot = service.get_or_compute("us", market_data, screener, {})

    assert call_count["n"] > 1  # el resto del universo se siguió procesando
    assert snapshot.analyzed >= 0  # no propagó la excepción
