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
