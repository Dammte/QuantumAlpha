from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import technical_analysis as ta
from app.services import watchlist_service as wl


def _snap(**overrides) -> TickerSnapshot:
    defaults = dict(
        ticker="TEST",
        sector="Tecnología",
        industry="Software empresarial",
        cap_tier="large",
        price=100.0,
        change_1d=0.0,
        change_1w=0.0,
        change_1m=0.0,
        change_3m=0.0,
        change_6m=0.0,
        change_1y=0.0,
        volume=1_000_000.0,
        relative_volume=1.0,
        rsi14=50.0,
        sma20=95.0,
        sma50=90.0,
        sma150=85.0,
        sma200=80.0,
        dist_52w_high=-0.2,
        dist_52w_low=0.3,
        atr_multiple=1.0,
        adx14=15.0,
        plus_di=20.0,
        minus_di=20.0,
        mansfield_rs=0.0,
        trend=ta.TrendState.SIDEWAYS,
        stage=None,
        ma_cross=None,
        minervini_score=0,
        minervini_pass=False,
        rs_rating=50,
    )
    defaults.update(overrides)
    return TickerSnapshot(**defaults)


def test_short_term_breakout_with_volume():
    snap = _snap(dist_52w_high=-0.01, relative_volume=1.5)
    items = wl.build_watchlist([snap], horizon=wl.SHORT_TERM)
    assert len(items) == 1
    assert "Ruptura" in items[0].reasons[0]


def test_short_term_oversold_bounce():
    snap = _snap(rsi14=30.0, change_1d=0.02)
    items = wl.build_watchlist([snap], horizon=wl.SHORT_TERM)
    assert len(items) == 1
    assert "sobreventa" in items[0].reasons[0]


def test_short_term_strong_confirmed_trend():
    snap = _snap(adx14=30.0, plus_di=25.0, minus_di=10.0, change_1w=0.05)
    items = wl.build_watchlist([snap], horizon=wl.SHORT_TERM)
    assert len(items) == 1


def test_no_short_term_match_when_nothing_qualifies():
    snap = _snap()
    assert wl.build_watchlist([snap], horizon=wl.SHORT_TERM) == []


def test_medium_term_minervini_pass():
    snap = _snap(minervini_pass=True)
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert len(items) == 1
    assert "Minervini" in items[0].reasons[0]


def test_medium_term_stage2_with_high_rs():
    snap = _snap(stage=ta.Stage.STAGE_2, rs_rating=85)
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert len(items) == 1


def test_medium_term_golden_cross():
    snap = _snap(ma_cross="golden")
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert len(items) == 1


def test_long_term_sustained_rs_leader():
    snap = _snap(rs_rating=95, trend=ta.TrendState.UPTREND, dist_52w_high=-0.05)
    items = wl.build_watchlist([snap], horizon=wl.LONG_TERM)
    assert len(items) == 1


def test_long_term_mansfield_positive_in_stage2():
    snap = _snap(mansfield_rs=2.5, stage=ta.Stage.STAGE_2)
    items = wl.build_watchlist([snap], horizon=wl.LONG_TERM)
    assert len(items) == 1


def test_ticker_can_appear_in_multiple_horizons():
    snap = _snap(minervini_pass=True, rsi14=30.0, change_1d=0.02)
    items = wl.build_watchlist([snap])  # all horizons
    horizons_matched = {item.horizon for item in items}
    assert wl.SHORT_TERM in horizons_matched
    assert wl.MEDIUM_TERM in horizons_matched


def test_sorted_by_rs_rating_descending():
    weak = _snap(ticker="WEAK", minervini_pass=True, rs_rating=20)
    strong = _snap(ticker="STRONG", minervini_pass=True, rs_rating=95)
    items = wl.build_watchlist([weak, strong], horizon=wl.MEDIUM_TERM)
    assert [i.ticker for i in items] == ["STRONG", "WEAK"]
