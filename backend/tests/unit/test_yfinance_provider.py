from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from app.infrastructure.market_data.yfinance_provider import YFinanceProvider


def _multiindex_frame(tickers: list[str]) -> pd.DataFrame:
    """Mirrors the real shape `yf.download(..., group_by="ticker")` returns -
    a (Ticker, Price) MultiIndex, even for a single ticker (verified against the
    real yfinance API; regression-tested here because a wrong assumption about
    "single ticker means flat columns" previously broke this method in production)."""
    dates = pd.date_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    data = {}
    for i, ticker in enumerate(tickers):
        base = 100.0 + i * 10
        data[(ticker, "Open")] = [base, base + 1, base + 2]
        data[(ticker, "High")] = [base + 1, base + 2, base + 3]
        data[(ticker, "Low")] = [base - 1, base, base + 1]
        data[(ticker, "Close")] = [base + 0.5, base + 1.5, base + 2.5]
        data[(ticker, "Volume")] = [1000, 1100, 1200]
    return pd.DataFrame(data, index=dates, columns=columns)


@patch("app.infrastructure.market_data.yfinance_provider.yf.download")
def test_get_bulk_price_history_single_ticker(mock_download) -> None:
    mock_download.return_value = _multiindex_frame(["AAPL"])
    provider = YFinanceProvider()

    result = provider.get_bulk_price_history(["AAPL"], date(2024, 1, 1), date(2024, 1, 3))

    assert list(result.keys()) == ["AAPL"]
    assert len(result["AAPL"]) == 3
    assert result["AAPL"][0].close == pytest.approx(100.5)


@patch("app.infrastructure.market_data.yfinance_provider.yf.download")
def test_get_bulk_price_history_multiple_tickers(mock_download) -> None:
    mock_download.return_value = _multiindex_frame(["AAPL", "MSFT"])
    provider = YFinanceProvider()

    result = provider.get_bulk_price_history(["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 3))

    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert result["MSFT"][0].close == pytest.approx(110.5)


@patch("app.infrastructure.market_data.yfinance_provider.yf.download")
def test_get_bulk_price_history_empty_frame_returns_empty_dict(mock_download) -> None:
    mock_download.return_value = pd.DataFrame()
    provider = YFinanceProvider()

    assert provider.get_bulk_price_history(["AAPL"], date(2024, 1, 1), date(2024, 1, 3)) == {}


def test_get_bulk_price_history_no_tickers_returns_empty_dict() -> None:
    provider = YFinanceProvider()
    assert provider.get_bulk_price_history([], date(2024, 1, 1), date(2024, 1, 3)) == {}


def _major_holders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Value": [0.0165, 0.6595, 0.6705, 7661.0]},
        index=[
            "insidersPercentHeld",
            "institutionsPercentHeld",
            "institutionsFloatPercentHeld",
            "institutionsCount",
        ],
    )


def _institutional_holders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date Reported": ["2026-03-31", "2026-03-31"],
            "Holder": ["Blackrock Inc.", "Vanguard Group Inc"],
            "pctHeld": [0.081, 0.068],
            "Shares": [1_000_000.0, 900_000.0],
            "Value": [356_000_277_175.0, 296_646_618_528.0],
            "pctChange": [-0.0086, 1.0],
        }
    )


@patch("app.infrastructure.market_data.yfinance_provider.yf.Ticker")
def test_get_holders_parses_real_shaped_data(mock_ticker_cls) -> None:
    mock_ticker_cls.return_value.major_holders = _major_holders_frame()
    mock_ticker_cls.return_value.institutional_holders = _institutional_holders_frame()
    provider = YFinanceProvider()

    result = provider.get_holders("AAPL")

    assert result is not None
    assert result.pct_held_by_institutions == pytest.approx(0.6595)
    assert result.pct_held_by_insiders == pytest.approx(0.0165)
    assert len(result.top_institutional_holders) == 2
    top = result.top_institutional_holders[0]
    assert top.holder == "Blackrock Inc."
    assert top.shares == pytest.approx(1_000_000.0)
    assert top.pct_held == pytest.approx(0.081)
    assert top.date_reported == "2026-03-31"


@patch("app.infrastructure.market_data.yfinance_provider.yf.Ticker")
def test_get_holders_caps_at_max_holders(mock_ticker_cls) -> None:
    many_holders = pd.DataFrame(
        {
            "Date Reported": ["2026-03-31"] * 15,
            "Holder": [f"Holder {i}" for i in range(15)],
            "pctHeld": [0.01] * 15,
            "Shares": [1000.0] * 15,
            "Value": [1000.0] * 15,
            "pctChange": [0.0] * 15,
        }
    )
    mock_ticker_cls.return_value.major_holders = _major_holders_frame()
    mock_ticker_cls.return_value.institutional_holders = many_holders
    provider = YFinanceProvider()

    result = provider.get_holders("AAPL")

    assert result is not None
    assert len(result.top_institutional_holders) == 10


@patch("app.infrastructure.market_data.yfinance_provider.yf.Ticker")
def test_get_holders_none_when_both_frames_empty(mock_ticker_cls) -> None:
    mock_ticker_cls.return_value.major_holders = pd.DataFrame()
    mock_ticker_cls.return_value.institutional_holders = pd.DataFrame()
    provider = YFinanceProvider()

    assert provider.get_holders("UNKNOWN") is None


@patch("app.infrastructure.market_data.yfinance_provider.yf.Ticker")
def test_get_holders_none_on_exception(mock_ticker_cls) -> None:
    mock_ticker_cls.side_effect = RuntimeError("boom")
    provider = YFinanceProvider()

    assert provider.get_holders("AAPL") is None


@patch("app.infrastructure.market_data.yfinance_provider.yf.Ticker")
def test_get_holders_skips_rows_with_missing_holder_name(mock_ticker_cls) -> None:
    frame = _institutional_holders_frame()
    frame.loc[0, "Holder"] = None
    mock_ticker_cls.return_value.major_holders = _major_holders_frame()
    mock_ticker_cls.return_value.institutional_holders = frame
    provider = YFinanceProvider()

    result = provider.get_holders("AAPL")

    assert result is not None
    assert len(result.top_institutional_holders) == 1
    assert result.top_institutional_holders[0].holder == "Vanguard Group Inc"
