from unittest.mock import Mock, patch

from app.infrastructure.macro_data.fred_client import FredClient


def test_is_configured_reflects_api_key_presence():
    assert FredClient(api_key="abc123").is_configured is True
    assert FredClient(api_key=None).is_configured is False
    assert FredClient(api_key="").is_configured is False


def test_latest_observation_returns_none_without_api_key():
    client = FredClient(api_key=None)
    assert client.latest_observation("T10Y2Y") is None


@patch("app.infrastructure.macro_data.fred_client.requests.get")
def test_latest_observation_parses_the_first_valid_value(mock_get):
    mock_get.return_value = Mock(
        json=lambda: {"observations": [{"date": "2026-08-01", "value": "0.45"}]},
        raise_for_status=lambda: None,
    )
    client = FredClient(api_key="key")

    result = client.latest_observation("T10Y2Y")

    assert result == ("2026-08-01", 0.45)


@patch("app.infrastructure.macro_data.fred_client.requests.get")
def test_latest_observation_skips_missing_markers(mock_get):
    mock_get.return_value = Mock(
        json=lambda: {
            "observations": [
                {"date": "2026-08-01", "value": "."},
                {"date": "2026-07-31", "value": "."},
                {"date": "2026-07-30", "value": "-0.12"},
            ]
        },
        raise_for_status=lambda: None,
    )
    client = FredClient(api_key="key")

    result = client.latest_observation("T10Y2Y")

    assert result == ("2026-07-30", -0.12)


@patch("app.infrastructure.macro_data.fred_client.requests.get")
def test_latest_observation_none_when_all_values_missing(mock_get):
    mock_get.return_value = Mock(
        json=lambda: {"observations": [{"date": "2026-08-01", "value": "."}]},
        raise_for_status=lambda: None,
    )
    client = FredClient(api_key="key")

    assert client.latest_observation("T10Y2Y") is None


@patch("app.infrastructure.macro_data.fred_client.requests.get")
def test_latest_observation_none_on_request_exception(mock_get):
    mock_get.side_effect = RuntimeError("network down")
    client = FredClient(api_key="key")

    assert client.latest_observation("T10Y2Y") is None


@patch("app.infrastructure.macro_data.fred_client.requests.get")
def test_latest_observation_none_on_bad_http_status(mock_get):
    response = Mock()
    response.raise_for_status.side_effect = RuntimeError("401 Unauthorized")
    mock_get.return_value = response
    client = FredClient(api_key="bad-key")

    assert client.latest_observation("T10Y2Y") is None


@patch("app.infrastructure.macro_data.fred_client.requests.get")
def test_latest_observation_passes_units_param(mock_get):
    mock_get.return_value = Mock(
        json=lambda: {"observations": [{"date": "2026-08-01", "value": "3.1"}]},
        raise_for_status=lambda: None,
    )
    client = FredClient(api_key="key")

    client.latest_observation("CPIAUCSL", units="pc1")

    assert mock_get.call_args.kwargs["params"]["units"] == "pc1"
