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
        atr_ratio_50d=1.0,
        atr_multiple_sma21=0.5,
        range_position_20d=0.5,
        mansfield_rs_4w=0.0,
        relative_volume_trend=0.0,
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


# --- Segunda auditoría, Bloque 3: setup separation + percentile scoring --------


def test_short_term_pullback_to_support():
    # Price just above a rising SMA50 (itself above SMA200 - confirmed
    # intermediate uptrend), RSI not oversold - a shallow, orderly dip, not a
    # deep bounce setup.
    snap = _snap(price=100.0, sma50=98.0, sma200=90.0, rsi14=55.0)
    items = wl.build_watchlist([snap], horizon=wl.SHORT_TERM)
    assert len(items) == 1
    assert items[0].setup == wl.PULLBACK_TO_SUPPORT
    assert "Retroceso" in items[0].reasons[0]


def test_short_term_no_pullback_when_price_already_below_sma50():
    snap = _snap(price=95.0, sma50=98.0, sma200=90.0, rsi14=55.0)
    assert wl.build_watchlist([snap], horizon=wl.SHORT_TERM) == []


def test_short_term_no_pullback_without_a_confirmed_intermediate_uptrend():
    snap = _snap(price=100.0, sma50=90.0, sma200=98.0, rsi14=55.0)  # SMA50 below SMA200
    assert wl.build_watchlist([snap], horizon=wl.SHORT_TERM) == []


def test_short_term_no_pullback_when_already_oversold():
    # RSI this low is oversold_bounce's territory, not pullback_to_support's -
    # but change_1d isn't positive here, so oversold_bounce doesn't fire either.
    snap = _snap(price=100.0, sma50=98.0, sma200=90.0, rsi14=35.0, change_1d=-0.01)
    assert wl.build_watchlist([snap], horizon=wl.SHORT_TERM) == []


def test_ticker_matching_two_setups_appears_as_two_separate_items():
    # Oversold *and* a volume breakout on the same day - genuinely possible,
    # and each setup type gets its own card, not one blended entry.
    snap = _snap(
        rsi14=30.0, change_1d=0.02,  # oversold_bounce
        dist_52w_high=-0.01, relative_volume=1.5,  # breakout_volume
    )
    items = wl.build_watchlist([snap], horizon=wl.SHORT_TERM)
    assert {i.setup for i in items} == {wl.OVERSOLD_BOUNCE, wl.BREAKOUT_VOLUME}
    assert len(items) == 2


def test_short_term_items_carry_a_setup_and_percentile_score():
    snap = _snap(rsi14=30.0, change_1d=0.02)
    items = wl.build_watchlist([snap], horizon=wl.SHORT_TERM)
    assert items[0].setup == wl.OVERSOLD_BOUNCE
    assert items[0].percentile_score is not None


def test_medium_and_long_term_items_carry_no_setup_or_percentile_score():
    snap = _snap(minervini_pass=True)
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert items[0].setup is None
    assert items[0].percentile_score is None


def test_setup_percentile_scores_ranks_within_the_given_snapshots():
    weak = _snap(ticker="WEAK", change_1w=-0.05, relative_volume=0.8)
    strong = _snap(ticker="STRONG", change_1w=0.08, relative_volume=2.0)
    scores = wl.setup_percentile_scores([weak, strong], wl.BREAKOUT_VOLUME)
    assert scores["STRONG"] > scores["WEAK"]


def test_setup_percentile_scores_inverts_change_1w_for_oversold_bounce():
    # A deeper recent drop sets up a bigger bounce - scores *higher* for
    # oversold_bounce specifically, the opposite of every other setup.
    dropped_more = _snap(ticker="DROPPED", change_1w=-0.10, relative_volume=1.0)
    dropped_less = _snap(ticker="FLAT", change_1w=-0.01, relative_volume=1.0)
    oversold_scores = wl.setup_percentile_scores([dropped_more, dropped_less], wl.OVERSOLD_BOUNCE)
    breakout_scores = wl.setup_percentile_scores([dropped_more, dropped_less], wl.BREAKOUT_VOLUME)
    assert oversold_scores["DROPPED"] > oversold_scores["FLAT"]
    assert breakout_scores["DROPPED"] < breakout_scores["FLAT"]


def test_setup_percentile_scores_skips_missing_fields_without_crashing():
    thin_data = _snap(
        ticker="THIN", change_1w=None, relative_volume=None, relative_volume_trend=None,
        atr_ratio_50d=None, atr_multiple_sma21=None, range_position_20d=None, mansfield_rs_4w=None,
    )
    assert wl.setup_percentile_scores([thin_data], wl.OVERSOLD_BOUNCE) == {}
