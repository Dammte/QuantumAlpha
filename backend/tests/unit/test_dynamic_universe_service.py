from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from app.domain.models.ticker_info import TickerInfo
from app.domain.models.universe_membership import UniverseMember
from app.services import dynamic_universe_service as dus

_SP500_HTML = """
<table class="wikitable">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
</table>
"""

_STOXX_HTML = """
<table><tr><th>Year</th><th>Points</th></tr><tr><td>2020</td><td>100</td></tr></table>
<table>
<tr><th>Ticker</th><th>Company</th><th>ICB Sector</th><th>Country</th><th>Headquarters</th></tr>
<tr><td>SAF</td><td>Safran</td><td>Industrials</td><td>France</td><td>Paris</td></tr>
<tr><td>ZURN</td><td>Zurich Insurance</td><td>Financials</td><td>Switzerland</td><td>Zurich</td></tr>
<tr><td>XYZ</td><td>Unknown Co</td><td>Other</td><td>Nowhereland</td><td>Nowhere</td></tr>
</table>
"""


def test_parse_us_constituents_reads_symbol_and_sector():
    result = dus.parse_us_constituents(_SP500_HTML)
    assert {c.ticker for c in result} == {"MMM", "BRK-B"}  # dot -> dash for Yahoo Finance
    by_ticker = {c.ticker: c for c in result}
    assert by_ticker["MMM"].sector == "Industrials"


def test_parse_stoxx600_constituents_maps_country_to_yahoo_suffix():
    result = dus.parse_stoxx600_constituents(_STOXX_HTML)
    tickers = {c.ticker for c in result}
    assert tickers == {"SAF.PA", "ZURN.SW"}  # "Nowhereland" has no suffix mapping - skipped


def test_parse_stoxx600_constituents_picks_the_table_with_ticker_and_country_columns():
    # The decoy "Year"/"Points" table (same shape as several real navboxes on
    # that page) must never be mistaken for the constituents table.
    result = dus.parse_stoxx600_constituents(_STOXX_HTML)
    assert len(result) == 2


_STOXX_HTML_WITH_DUPLICATE = """
<table>
<tr><th>Ticker</th><th>Company</th><th>ICB Sector</th><th>Country</th><th>Headquarters</th></tr>
<tr><td>SHEL</td><td>Shell</td><td>Energy</td><td>United Kingdom</td><td>London</td></tr>
<tr><td>SHEL</td><td>Shell</td><td>Energy</td><td>United Kingdom</td><td>London</td></tr>
</table>
"""


def test_fetch_live_constituents_dedupes_a_duplicate_row_on_the_source_page(monkeypatch):
    # Real production bug found running scripts/refresh_universe_membership.py
    # for real: "SHEL.L" appears twice on the actual STOXX 600 Wikipedia
    # page, which violated universe_memberships' own uniqueness constraint at
    # save time before this fix.
    monkeypatch.setattr(dus, "_fetch_html", lambda url: _STOXX_HTML_WITH_DUPLICATE)
    result = dus.fetch_live_constituents("europe")
    assert [c.ticker for c in result] == ["SHEL.L"]


def test_fetch_live_constituents_dedupes_us_overlap_between_sp500_and_sp400(monkeypatch):
    monkeypatch.setattr(dus, "_fetch_html", lambda url: _SP500_HTML)  # same fixture for both S&P 500 and S&P 400
    result = dus.fetch_live_constituents("us")
    assert sorted(c.ticker for c in result) == ["BRK-B", "MMM"]


def _ohlc_df(n: int, close: float, volume: float) -> pd.DataFrame:
    dates = pd.bdate_range(end=date.today() - timedelta(days=1), periods=n)
    closes = pd.Series([close] * n, index=dates)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [volume] * n}
    )


def _ticker_info(market_cap: float | None) -> TickerInfo:
    return TickerInfo(
        ticker="X", name=None, sector=None, industry=None, market_cap=market_cap, currency="USD",
        trailing_pe=None, forward_pe=None, dividend_yield=None, beta=None, average_volume=None,
        analyst_recommendation=None, analyst_target_mean_price=None, analyst_opinion_count=None,
        revenue_growth=None, profit_margins=None, debt_to_equity=None,
    )


class _FakeMarketData:
    """Duck-typed stand-in for MarketDataService - only the two methods
    apply_liquidity_filter actually calls."""

    def __init__(self, ohlcv: dict[str, pd.DataFrame], market_caps: dict[str, float | None]):
        self._ohlcv = ohlcv
        self._market_caps = market_caps

    def get_bulk_ohlcv(self, tickers, start, end):
        return {t: self._ohlcv[t] for t in tickers if t in self._ohlcv}

    def get_ticker_info(self, ticker):
        if ticker not in self._market_caps:
            return None
        return _ticker_info(self._market_caps[ticker])


def test_apply_liquidity_filter_keeps_names_clearing_every_bar():
    constituents = [dus.RawConstituent(ticker="LIQUID", sector="Tech")]
    market_data = _FakeMarketData(
        ohlcv={"LIQUID": _ohlc_df(25, close=50.0, volume=1_000_000.0)},  # $50 x 1M = $50M/day, well above $20M
        market_caps={"LIQUID": 5_000_000_000.0},
    )
    result = dus.apply_liquidity_filter(constituents, market_data)
    assert [m.ticker for m in result] == ["LIQUID"]
    assert result[0].sector == "Tech"


def test_apply_liquidity_filter_rejects_a_penny_stock():
    constituents = [dus.RawConstituent(ticker="PENNY", sector=None)]
    market_data = _FakeMarketData(
        ohlcv={"PENNY": _ohlc_df(25, close=1.0, volume=100_000_000.0)},  # huge volume, but price < $5
        market_caps={"PENNY": 2_000_000_000.0},
    )
    assert dus.apply_liquidity_filter(constituents, market_data) == []


def test_apply_liquidity_filter_rejects_thin_dollar_volume():
    constituents = [dus.RawConstituent(ticker="THIN", sector=None)]
    market_data = _FakeMarketData(
        ohlcv={"THIN": _ohlc_df(25, close=50.0, volume=1_000.0)},  # $50k/day, well under $20M
        market_caps={"THIN": 2_000_000_000.0},
    )
    assert dus.apply_liquidity_filter(constituents, market_data) == []


def test_apply_liquidity_filter_rejects_a_small_cap():
    constituents = [dus.RawConstituent(ticker="SMALLCAP", sector=None)]
    market_data = _FakeMarketData(
        ohlcv={"SMALLCAP": _ohlc_df(25, close=50.0, volume=1_000_000.0)},
        market_caps={"SMALLCAP": 500_000_000.0},  # below the $1B floor
    )
    assert dus.apply_liquidity_filter(constituents, market_data) == []


def test_apply_liquidity_filter_rejects_a_ticker_with_no_data_at_all():
    # Delisted since the Wikipedia snapshot, or a ticker Yahoo Finance simply
    # doesn't recognize - fails the same way an illiquid one does.
    constituents = [dus.RawConstituent(ticker="GHOST", sector=None)]
    market_data = _FakeMarketData(ohlcv={}, market_caps={})
    assert dus.apply_liquidity_filter(constituents, market_data) == []


def test_apply_liquidity_filter_empty_input_returns_empty():
    assert dus.apply_liquidity_filter([], _FakeMarketData({}, {})) == []


class _FakeUniverseMembershipRepo:
    def __init__(self):
        self._snapshots: dict[tuple[str, date], list[UniverseMember]] = {}

    def save_snapshot(self, region, as_of_date, source, members):
        self._snapshots[(region, as_of_date)] = [replace(m, source=source) for m in members]

    def latest_as_of_date(self, region):
        dates = [d for (r, d) in self._snapshots if r == region]
        return max(dates) if dates else None

    def members_as_of(self, region, as_of_date=None):
        target = as_of_date if as_of_date is not None else self.latest_as_of_date(region)
        if target is None:
            return []
        applicable = [d for (r, d) in self._snapshots if r == region and d <= target]
        if not applicable:
            return []
        return self._snapshots[(region, max(applicable))]

    def all_as_of_dates(self, region):
        return sorted(d for (r, d) in self._snapshots if r == region)


def test_is_refresh_due_when_never_refreshed():
    assert dus.is_refresh_due(_FakeUniverseMembershipRepo(), "us") is True


def test_is_refresh_due_false_within_the_interval():
    repo = _FakeUniverseMembershipRepo()
    today = date(2026, 6, 15)
    repo.save_snapshot("us", today - timedelta(days=10), "live", [])
    assert dus.is_refresh_due(repo, "us", today=today) is False


def test_is_refresh_due_true_past_the_interval():
    repo = _FakeUniverseMembershipRepo()
    today = date(2026, 6, 15)
    repo.save_snapshot("us", today - timedelta(days=31), "live", [])
    assert dus.is_refresh_due(repo, "us", today=today) is True


def test_read_dynamic_universe_none_when_nothing_on_file():
    assert dus.read_dynamic_universe(_FakeUniverseMembershipRepo(), "us") is None


def test_read_dynamic_universe_returns_ticker_to_sector_mapping():
    repo = _FakeUniverseMembershipRepo()
    today = date(2026, 6, 15)
    repo.save_snapshot(
        "us", today, "live",
        [
            UniverseMember("AAPL", "us", "Technology", today, "live"),
            UniverseMember("XOM", "us", "Energy", today, "live"),
        ],
    )
    result = dus.read_dynamic_universe(repo, "us")
    assert result == {"AAPL": "Technology", "XOM": "Energy"}


def test_refresh_universe_membership_uses_live_source_when_fetch_succeeds(monkeypatch):
    monkeypatch.setattr(
        dus, "fetch_live_constituents", lambda region: [dus.RawConstituent(ticker="LIQUID", sector="Tech")]
    )
    market_data = _FakeMarketData(
        ohlcv={"LIQUID": _ohlc_df(25, close=50.0, volume=1_000_000.0)}, market_caps={"LIQUID": 5_000_000_000.0}
    )
    repo = _FakeUniverseMembershipRepo()
    count, source = dus.refresh_universe_membership("us", market_data, repo)
    assert (count, source) == (1, "live")
    assert repo.members_as_of("us")[0].ticker == "LIQUID"


def test_refresh_universe_membership_falls_back_to_curated_when_live_fetch_fails(monkeypatch):
    monkeypatch.setattr(dus, "fetch_live_constituents", lambda region: None)
    repo = _FakeUniverseMembershipRepo()
    count, source = dus.refresh_universe_membership("us", _FakeMarketData({}, {}), repo)
    assert source == "curated_fallback"
    assert count > 0  # the real curated dict in market_universe.py is non-empty
    assert all(m.source == "curated_fallback" for m in repo.members_as_of("us"))


def test_refresh_universe_membership_falls_back_when_liquidity_filter_leaves_nothing(monkeypatch):
    # A live fetch that technically "succeeds" but the filter rejects every
    # single name must not persist an empty snapshot - falls back instead.
    monkeypatch.setattr(
        dus, "fetch_live_constituents", lambda region: [dus.RawConstituent(ticker="PENNY", sector=None)]
    )
    market_data = _FakeMarketData(
        ohlcv={"PENNY": _ohlc_df(25, close=1.0, volume=100.0)}, market_caps={"PENNY": 2_000_000_000.0}
    )
    repo = _FakeUniverseMembershipRepo()
    count, source = dus.refresh_universe_membership("us", market_data, repo)
    assert source == "curated_fallback"
    assert count > 0
