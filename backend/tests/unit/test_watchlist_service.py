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


# Tercera auditoría, Bloque F-2: the "weekly" tier used to run on Minervini
# 8/8 + Stage 2/RS>=80 + a golden cross on SMA50/SMA200 - all months-scale
# signals on a tier meant for a weekly review cadence. Rebuilt on the short
# pair (ma_cross_short/imminent_cross_short_term), split by setup like the
# daily tier - see MEDIUM_TERM_SETUPS.


def test_medium_term_fast_golden_cross():
    snap = _snap(ma_cross_short="golden")
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert len(items) == 1
    assert items[0].setup == wl.FAST_GOLDEN_CROSS
    assert "corto plazo" in items[0].reasons[0] or "corto" in items[0].reasons[0]


def test_medium_term_fast_cross_imminent():
    snap = _snap(imminent_cross_short_term=ta.ImminentCross(direction="golden", bars_until=3, r_squared=0.75))
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert len(items) == 1
    assert items[0].setup == wl.FAST_CROSS_IMMINENT
    assert "3" in items[0].reasons[0]


def test_medium_term_no_fast_cross_imminent_when_direction_is_bearish():
    snap = _snap(imminent_cross_short_term=ta.ImminentCross(direction="death", bars_until=3, r_squared=0.75))
    assert wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM) == []


def test_medium_term_stage2_with_high_rs():
    snap = _snap(stage=ta.Stage.STAGE_2, rs_rating=85)
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert len(items) == 1
    assert items[0].setup == wl.STAGE2_LEADER


def test_medium_term_no_match_when_nothing_qualifies():
    snap = _snap()
    assert wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM) == []


def test_medium_term_ticker_matching_two_setups_appears_twice():
    snap = _snap(ma_cross_short="golden", stage=ta.Stage.STAGE_2, rs_rating=85)
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert {i.setup for i in items} == {wl.FAST_GOLDEN_CROSS, wl.STAGE2_LEADER}


def test_long_term_sustained_rs_leader():
    snap = _snap(rs_rating=95, trend=ta.TrendState.UPTREND, dist_52w_high=-0.05)
    items = wl.build_watchlist([snap], horizon=wl.LONG_TERM)
    assert len(items) == 1


def test_long_term_mansfield_positive_in_stage2():
    snap = _snap(mansfield_rs=2.5, stage=ta.Stage.STAGE_2)
    items = wl.build_watchlist([snap], horizon=wl.LONG_TERM)
    assert len(items) == 1


def test_ticker_can_appear_in_multiple_horizons():
    snap = _snap(ma_cross_short="golden", rsi14=30.0, change_1d=0.02)
    items = wl.build_watchlist([snap])  # all horizons
    horizons_matched = {item.horizon for item in items}
    assert wl.SHORT_TERM in horizons_matched
    assert wl.MEDIUM_TERM in horizons_matched


def test_medium_term_sorted_by_percentile_score_descending():
    # Tercera auditoría, Bloque F-2: medium-term items now carry a real
    # percentile_score (same architecture as the daily tier) - _sort_key
    # uses it in preference to rs_rating, so ordering is driven by the
    # cross-sectional percentile fields, not rs_rating directly.
    weak = _snap(ticker="WEAK", ma_cross_short="golden", mansfield_rs_4w=-5.0)
    strong = _snap(ticker="STRONG", ma_cross_short="golden", mansfield_rs_4w=5.0)
    items = wl.build_watchlist([weak, strong], horizon=wl.MEDIUM_TERM)
    assert [i.ticker for i in items] == ["STRONG", "WEAK"]


def test_medium_term_items_with_no_score_at_all_sort_last_not_first():
    # Tercera auditoría, Bloque A-4: `sorted(..., reverse=True)` flips a
    # tuple's leading bool too, so a ticker with neither percentile_score nor
    # rs_rating used to land at the *top* of the watchlist - rewarding
    # missing data over a real, if weak, score.
    no_score = _snap(
        ticker="NO_SCORE", ma_cross_short="golden", rs_rating=None, change_1w=None, relative_volume=None,
        relative_volume_trend=None, atr_ratio_50d=None, atr_multiple_sma21=None, range_position_20d=None,
        mansfield_rs_4w=None, adx14=None,
    )
    weak = _snap(ticker="WEAK", ma_cross_short="golden", mansfield_rs_4w=-5.0)
    strong = _snap(ticker="STRONG", ma_cross_short="golden", mansfield_rs_4w=5.0)
    items = wl.build_watchlist([no_score, weak, strong], horizon=wl.MEDIUM_TERM)
    assert [i.ticker for i in items] == ["STRONG", "WEAK", "NO_SCORE"]


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


def test_medium_term_items_carry_a_setup_and_percentile_score():
    # Tercera auditoría, Bloque F-2: rebuilt on the same setup-type
    # architecture as the daily tier - no longer the "no setup, no
    # percentile_score" blended-reasons item medium/long term items used to
    # both be.
    snap = _snap(ma_cross_short="golden")
    items = wl.build_watchlist([snap], horizon=wl.MEDIUM_TERM)
    assert items[0].setup == wl.FAST_GOLDEN_CROSS
    assert items[0].percentile_score is not None


def test_long_term_items_carry_no_setup_or_percentile_score():
    snap = _snap(rs_rating=95, trend=ta.TrendState.UPTREND, dist_52w_high=-0.05)
    items = wl.build_watchlist([snap], horizon=wl.LONG_TERM)
    assert items[0].setup is None
    assert items[0].percentile_score is None


def test_percentile_rank_by_ticker_ranks_the_given_field():
    weak = _snap(ticker="WEAK", mansfield_rs_4w=-5.0)
    strong = _snap(ticker="STRONG", mansfield_rs_4w=5.0)
    ranks = wl.percentile_rank_by_ticker([weak, strong], "mansfield_rs_4w")
    assert ranks["STRONG"] > ranks["WEAK"]


def test_percentile_rank_by_ticker_skips_tickers_with_a_none_field():
    has_value = _snap(ticker="HAS_VALUE", mansfield_rs_4w=5.0)
    missing = _snap(ticker="MISSING", mansfield_rs_4w=None)
    ranks = wl.percentile_rank_by_ticker([has_value, missing], "mansfield_rs_4w")
    assert set(ranks) == {"HAS_VALUE"}


def test_setup_percentile_scores_ranks_within_the_given_snapshots():
    weak = _snap(ticker="WEAK", change_1w=-0.05, relative_volume=0.8)
    strong = _snap(ticker="STRONG", change_1w=0.08, relative_volume=2.0)
    scores = wl.setup_percentile_scores([weak, strong], wl.BREAKOUT_VOLUME)
    assert scores["STRONG"] > scores["WEAK"]


def test_setup_percentile_scores_inverts_change_1w_for_oversold_bounce():
    # A deeper recent drop sets up a bigger bounce - scores *higher* for
    # oversold_bounce specifically, unlike a plain (non-inverted) percentile.
    dropped_more = _snap(ticker="DROPPED", change_1w=-0.10, relative_volume=1.0)
    dropped_less = _snap(ticker="FLAT", change_1w=-0.01, relative_volume=1.0)
    oversold_scores = wl.setup_percentile_scores([dropped_more, dropped_less], wl.OVERSOLD_BOUNCE)
    assert oversold_scores["DROPPED"] > oversold_scores["FLAT"]


# --- Tercera auditoría, Bloque F-4: each setup its own field combo/sign ------


def test_setup_percentile_scores_breakout_volume_and_trend_continuation_differ():
    # The exact bug: before this, every setup but oversold_bounce shared the
    # same 7-field composite, so a ticker matching two of them scored
    # identically in both (measured: AAPL at 82.14 in both). Built from
    # genuinely distinct fields now (relative_volume_trend/range_position_20d
    # for breakout_volume vs. adx14/atr_multiple_sma21/mansfield_rs_4w for
    # trend_continuation), so differing on fields one setup cares about but
    # the other doesn't must produce different scores.
    snap = _snap(ticker="T", relative_volume_trend=2.0, range_position_20d=0.95, adx14=10.0,
                 atr_multiple_sma21=3.0, mansfield_rs_4w=-5.0)
    other = _snap(ticker="OTHER")
    breakout = wl.setup_percentile_scores([snap, other], wl.BREAKOUT_VOLUME)
    trend = wl.setup_percentile_scores([snap, other], wl.TREND_CONTINUATION)
    assert breakout["T"] != trend["T"]


def test_setup_percentile_scores_trend_continuation_rewards_adx_and_penalizes_overextension():
    strong_trend = _snap(ticker="STRONG", adx14=35.0, atr_multiple_sma21=0.5)
    weak_overextended = _snap(ticker="WEAK", adx14=15.0, atr_multiple_sma21=4.0)
    scores = wl.setup_percentile_scores([strong_trend, weak_overextended], wl.TREND_CONTINUATION)
    assert scores["STRONG"] > scores["WEAK"]


def test_setup_percentile_scores_pullback_rewards_shallow_dip_and_relative_strength():
    orderly = _snap(ticker="ORDERLY", atr_multiple_sma21=0.3, mansfield_rs_4w=8.0)
    blown_out = _snap(ticker="BLOWN_OUT", atr_multiple_sma21=5.0, mansfield_rs_4w=-8.0)
    scores = wl.setup_percentile_scores([orderly, blown_out], wl.PULLBACK_TO_SUPPORT)
    assert scores["ORDERLY"] > scores["BLOWN_OUT"]


def test_setup_percentile_scores_skips_missing_fields_without_crashing():
    thin_data = _snap(
        ticker="THIN", change_1w=None, relative_volume=None, relative_volume_trend=None,
        atr_ratio_50d=None, atr_multiple_sma21=None, range_position_20d=None, mansfield_rs_4w=None,
    )
    assert wl.setup_percentile_scores([thin_data], wl.OVERSOLD_BOUNCE) == {}


# --- WatchlistItem.setup_label: cuarta auditoría, recomendación FE-1 - un
# único origen de verdad para la etiqueta en español, para que el frontend
# nunca vuelva a mantener su propia copia desincronizada de este mapa.


def test_watchlist_item_setup_label_matches_setup_labels_dict():
    item = wl.WatchlistItem(
        ticker="T", sector="Tecnología", industry=None, cap_tier="large", horizon="short",
        reasons=[], snapshot=_snap(), setup=wl.OVERSOLD_BOUNCE,
    )
    assert item.setup_label == wl.SETUP_LABELS[wl.OVERSOLD_BOUNCE]


def test_watchlist_item_setup_label_none_when_no_setup():
    item = wl.WatchlistItem(
        ticker="T", sector="Tecnología", industry=None, cap_tier="large", horizon="long",
        reasons=[], snapshot=_snap(), setup=None,
    )
    assert item.setup_label is None


def test_watchlist_item_setup_label_covers_every_setup_constant():
    # A setup with no entry in SETUP_LABELS would silently render as raw
    # snake_case in the UI (exactly the bug this property exists to prevent) -
    # this test fails loudly instead if a future setup constant is added
    # without also adding its label.
    for setup in (*wl.SHORT_TERM_SETUPS, *wl.MEDIUM_TERM_SETUPS):
        item = wl.WatchlistItem(
            ticker="T", sector="Tecnología", industry=None, cap_tier="large", horizon="short",
            reasons=[], snapshot=_snap(), setup=setup,
        )
        assert item.setup_label is not None, f"{setup} has no entry in SETUP_LABELS"
