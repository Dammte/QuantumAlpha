from unittest.mock import Mock

from app.services.macro_data_service import CPI_SERIES, UNEMPLOYMENT_SERIES, YIELD_CURVE_SERIES, MacroDataService


def _client(is_configured=True, observations=None):
    client = Mock()
    client.is_configured = is_configured
    client.latest_observation.side_effect = lambda series_id, units="lin": (observations or {}).get(
        (series_id, units)
    )
    return client


def test_returns_none_when_client_not_configured():
    client = _client(is_configured=False)
    service = MacroDataService(client)

    assert service.get_macro_snapshot() is None
    client.latest_observation.assert_not_called()


def test_returns_none_when_every_series_fails():
    client = _client(observations={})
    service = MacroDataService(client)

    assert service.get_macro_snapshot() is None


def test_full_snapshot_with_all_series_available():
    client = _client(
        observations={
            (YIELD_CURVE_SERIES, "lin"): ("2026-08-01", 0.35),
            (UNEMPLOYMENT_SERIES, "lin"): ("2026-07-01", 4.2),
            (CPI_SERIES, "pc1"): ("2026-07-01", 2.9),
        }
    )
    service = MacroDataService(client)

    snapshot = service.get_macro_snapshot()

    assert snapshot is not None
    assert snapshot.yield_curve_spread == 0.35
    assert snapshot.yield_curve_date == "2026-08-01"
    assert snapshot.yield_curve_inverted is False
    assert snapshot.unemployment_rate == 4.2
    assert snapshot.cpi_yoy_change == 2.9


def test_yield_curve_inverted_when_spread_negative():
    client = _client(observations={(YIELD_CURVE_SERIES, "lin"): ("2026-08-01", -0.2)})
    service = MacroDataService(client)

    snapshot = service.get_macro_snapshot()

    assert snapshot is not None
    assert snapshot.yield_curve_inverted is True


def test_partial_snapshot_when_only_some_series_available():
    client = _client(observations={(YIELD_CURVE_SERIES, "lin"): ("2026-08-01", 0.1)})
    service = MacroDataService(client)

    snapshot = service.get_macro_snapshot()

    assert snapshot is not None
    assert snapshot.yield_curve_spread == 0.1
    assert snapshot.unemployment_rate is None
    assert snapshot.cpi_yoy_change is None
