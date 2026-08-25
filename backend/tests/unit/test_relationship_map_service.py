"""Tercera auditoría, Bloque G: relationship_map_service.py - three layers,
tested independently. Layer 1 (statistical) and Layer 2 (sector peers) are
pure functions over hand-built inputs; Layer 3 (EDGAR) is tested against a
mocked HTTP response, never the real network.
"""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import relationship_map_service as rms
from app.services import technical_analysis as ta
from app.services.market_universe import Industry, RegionConfig


def _snapshot(ticker: str, sector: str = "Tecnología", rs_rating: int | None = 50) -> TickerSnapshot:
    return TickerSnapshot(
        ticker=ticker, sector=sector, industry=None, cap_tier="large", price=100.0, change_1d=0.0,
        change_1w=0.0, change_1m=0.0, change_3m=0.0, change_6m=0.0, change_1y=0.0, volume=1_000_000.0,
        relative_volume=1.0, rsi14=50.0, sma20=95.0, sma50=90.0, sma150=85.0, sma200=80.0,
        dist_52w_high=-0.1, dist_52w_low=0.2, atr_multiple=1.0, adx14=15.0, plus_di=20.0, minus_di=20.0,
        mansfield_rs=0.0, trend=ta.TrendState.UPTREND, stage=None, ma_cross=None, minervini_score=0,
        minervini_pass=False, rs_rating=rs_rating,
    )


def _df(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=len(closes))
    close = pd.Series(closes, index=dates)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": [1_000_000.0] * len(closes)}
    )


# --- _lead_lag / _relative_beta / _extreme_day_comovement -------------------


def test_lead_lag_detects_a_when_leads_b_by_a_fixed_number_of_sessions():
    n = 120
    rng = np.random.RandomState(0)
    a = pd.Series(rng.normal(0, 1, n))
    lag = 3
    b = a.shift(lag).fillna(0)  # b[t] == a[t-lag] -> a leads b by `lag` sessions
    aligned = pd.DataFrame({"a": a, "b": b})
    best_lag, best_corr = rms._lead_lag(aligned)
    assert best_lag == lag
    assert best_corr > 0.9


def test_lead_lag_none_with_no_correlation_at_all():
    n = 120
    rng = np.random.RandomState(0)
    aligned = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    lag, corr = rms._lead_lag(aligned)
    # Some spurious correlation is expected by chance, but should be weak.
    assert corr is None or abs(corr) < 0.5


def test_relative_beta_reflects_the_true_scaling_factor():
    n = 100
    rng = np.random.RandomState(1)
    a = pd.Series(rng.normal(0, 1, n))
    b = a * 2.0  # b moves exactly 2x for every 1-unit move in a
    aligned = pd.DataFrame({"a": a, "b": b})
    beta = rms._relative_beta(aligned)
    assert beta == pytest.approx(2.0, abs=0.01)


def test_relative_beta_none_with_zero_variance():
    aligned = pd.DataFrame({"a": [1.0] * 30, "b": [2.0] * 30})
    assert rms._relative_beta(aligned) is None


def test_extreme_day_comovement_all_same_direction():
    a = pd.Series([0.001] * 20 + [0.05, -0.05, 0.06, -0.06])  # 4 extreme days
    b = pd.Series([0.001] * 20 + [0.04, -0.03, 0.05, -0.04])  # same sign every time
    aligned = pd.DataFrame({"a": a, "b": b})
    result = rms._extreme_day_comovement(aligned)
    assert result == pytest.approx(1.0)


def test_extreme_day_comovement_opposite_direction():
    a = pd.Series([0.001] * 20 + [0.05, -0.05, 0.06, -0.06])
    b = pd.Series([0.001] * 20 + [-0.04, 0.03, -0.05, 0.04])  # always opposite sign
    aligned = pd.DataFrame({"a": a, "b": b})
    result = rms._extreme_day_comovement(aligned)
    assert result == pytest.approx(0.0)


def test_extreme_day_comovement_none_with_zero_variance():
    aligned = pd.DataFrame({"a": [0.0] * 30, "b": [0.01] * 30})
    assert rms._extreme_day_comovement(aligned) is None


# --- compute_statistical_relations ------------------------------------------


def test_compute_statistical_relations_ranks_by_correlation_descending():
    n = 300
    rng = np.random.RandomState(2)
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    target_close = pd.Series(base)
    twin_close = target_close * 1.01  # near-identical returns -> correlation ~1.0
    independent_close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))

    ohlcv = {
        "TARGET": _df(list(target_close)),
        "TWIN": _df(list(twin_close)),
        "OTHER": _df(list(independent_close)),
    }
    snapshots = [_snapshot("TARGET"), _snapshot("TWIN"), _snapshot("OTHER")]
    relations = rms.compute_statistical_relations("TARGET", ohlcv, snapshots)
    assert relations[0].ticker == "TWIN"
    assert relations[0].correlation_60d == pytest.approx(1.0, abs=0.01)


def test_compute_statistical_relations_empty_when_ticker_missing_from_ohlcv():
    assert rms.compute_statistical_relations("GHOST", {}, []) == []


def test_compute_statistical_relations_flags_divergence():
    # First 250 bars: A and B move in lockstep. Last 60 bars: B moves on its
    # own, independent noise instead - a real historical correlation that's
    # recently vanished (not literally flat - a flat window makes the 60d
    # correlation itself undefined (NaN, zero variance), which is a
    # different, honestly-reported "can't tell" case, not "diverging").
    n_history = 250
    rng = np.random.RandomState(3)
    shared_moves = rng.normal(0, 1, n_history)
    a_history = 100 + np.cumsum(shared_moves)
    b_history = 100 + np.cumsum(shared_moves)  # identical history -> corr_250d ~= 1.0

    recent_a = a_history[-1] + np.cumsum(rng.normal(0, 1, 60))
    recent_b = b_history[-1] + np.cumsum(rng.normal(0, 1, 60))  # independent noise - decoupled, still moving

    a_close = list(a_history) + list(recent_a)
    b_close = list(b_history) + list(recent_b)
    ohlcv = {"TARGET": _df(a_close), "DECOUPLED": _df(b_close)}
    snapshots = [_snapshot("TARGET"), _snapshot("DECOUPLED")]
    relations = rms.compute_statistical_relations("TARGET", ohlcv, snapshots)
    decoupled = next(r for r in relations if r.ticker == "DECOUPLED")
    assert decoupled.is_diverging is True


def test_compute_statistical_relations_attaches_setup_and_percentile(monkeypatch):
    n = 100
    rng = np.random.RandomState(4)
    base = list(100 + np.cumsum(rng.normal(0, 1, n)))
    ohlcv = {"TARGET": _df(base), "RELATED": _df(base)}
    snapshots = [_snapshot("TARGET"), _snapshot("RELATED")]

    fake_item = type("Item", (), {"ticker": "RELATED", "setup": "oversold_bounce", "percentile_score": 88.0})()
    monkeypatch.setattr(rms.wl, "build_watchlist", lambda snapshots: [fake_item])

    relations = rms.compute_statistical_relations("TARGET", ohlcv, snapshots)
    related = next(r for r in relations if r.ticker == "RELATED")
    assert related.setup == "oversold_bounce"
    assert related.percentile_score == 88.0


# --- compute_sector_peers ----------------------------------------------------


def _fake_region_config():
    industry = Industry(name="Software", sector="Tecnología", etf="XSW", tickers=("AAPL", "MSFT", "ORCL"))
    other_industry = Industry(name="Banking", sector="Financiero", etf="XLF", tickers=("JPM", "BAC"))
    return RegionConfig(key="us", label="Estados Unidos", industries=(industry, other_industry),
                        sector_etfs={}, benchmark_ticker="SPY")


def test_compute_sector_peers_returns_same_industry_tickers_only(monkeypatch):
    monkeypatch.setattr(rms, "region_config", lambda region: _fake_region_config())
    snapshots = [_snapshot("AAPL", rs_rating=90), _snapshot("MSFT", rs_rating=70), _snapshot("JPM", rs_rating=99)]
    peers = rms.compute_sector_peers("AAPL", "us", snapshots)
    tickers = {p.ticker for p in peers}
    assert tickers == {"MSFT", "ORCL"}  # same industry, excluding AAPL itself; JPM (different industry) absent


def test_compute_sector_peers_sorted_by_rs_rating_descending(monkeypatch):
    monkeypatch.setattr(rms, "region_config", lambda region: _fake_region_config())
    snapshots = [_snapshot("AAPL", rs_rating=50), _snapshot("MSFT", rs_rating=95), _snapshot("ORCL", rs_rating=10)]
    peers = rms.compute_sector_peers("AAPL", "us", snapshots)
    assert [p.ticker for p in peers] == ["MSFT", "ORCL"]


def test_compute_sector_peers_empty_when_ticker_has_no_industry(monkeypatch):
    monkeypatch.setattr(rms, "region_config", lambda region: _fake_region_config())
    assert rms.compute_sector_peers("UNKNOWN_TICKER", "us", []) == []


# --- EDGAR (Layer 3): mocked HTTP, never real network -----------------------


_EDGAR_RESPONSE = {
    "hits": {
        "hits": [
            {
                "_source": {
                    "display_names": ["SUPPLIER CORP (0001234567)"],
                    "file_date": "2025-03-01",
                    "form": "10-K",
                }
            },
            {
                "_source": {
                    "display_names": ["CUSTOMER INC (0007654321)"],
                    "file_date": "2025-06-15",
                    "form": "10-Q",
                }
            },
        ]
    }
}


def test_fetch_edgar_mentions_parses_real_shaped_response(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = _EDGAR_RESPONSE
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr(rms.requests, "get", lambda *a, **k: mock_response)

    relations = rms._fetch_edgar_mentions("Analyzed Company Inc")
    assert relations is not None
    assert len(relations) == 2
    assert relations[0].filer_name == "SUPPLIER CORP (0001234567)"
    assert relations[0].form == "10-K"
    assert relations[0].filing_date == date(2025, 3, 1)


def test_fetch_edgar_mentions_none_on_network_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(rms.requests, "get", _raise)
    assert rms._fetch_edgar_mentions("Anything") is None


def test_fetch_edgar_mentions_none_on_malformed_json(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("not json")
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr(rms.requests, "get", lambda *a, **k: mock_response)
    assert rms._fetch_edgar_mentions("Anything") is None


def test_fetch_edgar_mentions_empty_list_when_no_hits(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {"hits": {"hits": []}}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr(rms.requests, "get", lambda *a, **k: mock_response)
    assert rms._fetch_edgar_mentions("Anything") == []


# --- get_disclosed_relations: availability semantics + caching -------------


def test_get_disclosed_relations_unavailable_with_no_company_name():
    relations, available = rms.get_disclosed_relations("XYZ", None, db=None)
    assert relations is None
    assert available is False


def test_get_disclosed_relations_available_true_with_empty_list_when_genuinely_nothing_found(monkeypatch):
    monkeypatch.setattr(rms, "_fetch_edgar_mentions", lambda name: [])
    relations, available = rms.get_disclosed_relations("XYZ", "XYZ Corp", db=None)
    assert relations == []
    assert available is True


def test_get_disclosed_relations_unavailable_when_edgar_fetch_fails(monkeypatch):
    monkeypatch.setattr(rms, "_fetch_edgar_mentions", lambda name: None)
    relations, available = rms.get_disclosed_relations("XYZ", "XYZ Corp", db=None)
    assert relations is None
    assert available is False


def test_get_disclosed_relations_reads_from_durable_cache_without_refetching(monkeypatch):
    calls = []
    monkeypatch.setattr(rms, "_fetch_edgar_mentions", lambda name: calls.append(name) or [])
    cached_payload = [
        {"filer_name": "CACHED CORP", "filer_ticker": None, "form": "10-K", "filing_date": "2025-01-01"}
    ]
    monkeypatch.setattr(
        rms.durable_cache, "load_fresh_as", lambda db, key, max_age, reconstruct: reconstruct(cached_payload)
    )
    relations, available = rms.get_disclosed_relations("XYZ", "XYZ Corp", db=MagicMock())
    assert available is True
    assert relations[0].filer_name == "CACHED CORP"
    assert calls == []  # never hit the network - served entirely from the durable cache
