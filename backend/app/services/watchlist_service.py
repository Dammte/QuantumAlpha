"""Builds the "acciones a revisar" watchlist: scans the universe snapshot against
rule bundles grouped by investment horizon (short/medium/long), so a name only
shows up once its technicals actually line up - the user reviews the list and
decides whether to act, rather than hunting through the whole screener by hand.

Segunda auditoría, Bloque 3: the short-term horizon used to OR three unrelated
setups (a 52-week breakout with volume, an oversold bounce, a confirmed-trend
continuation) into one blended reasons list - a name could match for any of
the three and there was no way to tell which, or to score them differently
even though the ablation study (docs/quant_methodology.md §12) measures wildly
different edges for each (oversold_bounce +0.723pp @5d, p<0.0001;
trend_continuation -0.301pp, IC-IR -0.461 @5d). Each is now its own setup
type, with its own card and its own cross-sectional percentile score - a
ticker matching two setups at once shows up as two separate items, one per
setup, each scored independently.
"""

from dataclasses import dataclass

import pandas as pd

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import technical_analysis as ta

SHORT_TERM = "short"
MEDIUM_TERM = "medium"
LONG_TERM = "long"

# The four short-term setup types (Segunda auditoría, Bloque 3) - see the
# module docstring. Only oversold_bounce has been individually ablation-
# validated so far; the other three are exposed distinctly *so that* they can
# be measured separately in a future study, not because all four are already
# known-good.
OVERSOLD_BOUNCE = "oversold_bounce"
BREAKOUT_VOLUME = "breakout_volume"
TREND_CONTINUATION = "trend_continuation"
PULLBACK_TO_SUPPORT = "pullback_to_support"
SHORT_TERM_SETUPS = (OVERSOLD_BOUNCE, BREAKOUT_VOLUME, TREND_CONTINUATION, PULLBACK_TO_SUPPORT)

SETUP_LABELS = {
    OVERSOLD_BOUNCE: "Rebote desde sobreventa",
    BREAKOUT_VOLUME: "Ruptura con volumen",
    TREND_CONTINUATION: "Continuación de tendencia",
    PULLBACK_TO_SUPPORT: "Retroceso a soporte",
}


@dataclass(frozen=True, slots=True)
class WatchlistItem:
    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    horizon: str
    reasons: list[str]
    snapshot: TickerSnapshot
    # `None` for medium/long-term items (they aren't split into setup types -
    # see the module docstring) and for a short-term item if, for some reason,
    # the cross-sectional score couldn't be computed for it.
    setup: str | None = None
    percentile_score: float | None = None


def _oversold_bounce_reason(s: TickerSnapshot) -> str | None:
    if s.rsi14 is not None and s.rsi14 <= 35 and s.change_1d is not None and s.change_1d > 0:
        return "Rebote desde zona de sobreventa (RSI ≤ 35 y hoy en positivo)"
    return None


def _breakout_volume_reason(s: TickerSnapshot) -> str | None:
    if s.dist_52w_high is not None and s.dist_52w_high >= -0.02 and s.relative_volume and s.relative_volume >= 1.3:
        return "Ruptura o cercanía a máximos de 52 semanas con volumen por encima de lo normal"
    return None


def _trend_continuation_reason(s: TickerSnapshot) -> str | None:
    if (
        s.adx14 is not None
        and s.adx14 >= 25
        and s.plus_di is not None
        and s.minus_di is not None
        and s.plus_di > s.minus_di
        and s.change_1w is not None
        and s.change_1w > 0
    ):
        return "Tendencia alcista fuerte y confirmada (ADX ≥ 25, +DI > -DI)"
    return None


PULLBACK_MAX_DISTANCE_ABOVE_SMA50 = 0.04  # within 4% above the 50-day average counts as "at" it, not far above
PULLBACK_MIN_RSI = 40.0  # a genuinely different setup from oversold_bounce (RSI <= 35) - a shallow, orderly dip


def _pullback_to_support_reason(s: TickerSnapshot) -> str | None:
    """The classic "orderly pullback to a rising 50-day average, inside a
    confirmed intermediate uptrend" setup - approximated from the moving
    averages every snapshot already carries rather than a fresh
    support/resistance-level computation across the whole universe (that
    would need a second full pass; this is the same information a
    pullback-to-the-50-day-line read is usually built on in practice)."""
    if s.price is None or s.sma50 is None or s.sma200 is None or s.rsi14 is None:
        return None
    if s.sma50 <= s.sma200:  # not even in a confirmed intermediate uptrend
        return None
    if s.price < s.sma50:  # already broke the average, not just pulling back to it
        return None
    if (s.price - s.sma50) / s.sma50 > PULLBACK_MAX_DISTANCE_ABOVE_SMA50:
        return None  # too far above the average to still call this "at" support
    if s.rsi14 <= PULLBACK_MIN_RSI:  # that's oversold_bounce's territory, not this setup's
        return None
    return "Retroceso ordenado hasta la media de 50 sesiones dentro de una tendencia alcista confirmada"


_SHORT_TERM_SETUP_DETECTORS = {
    OVERSOLD_BOUNCE: _oversold_bounce_reason,
    BREAKOUT_VOLUME: _breakout_volume_reason,
    TREND_CONTINUATION: _trend_continuation_reason,
    PULLBACK_TO_SUPPORT: _pullback_to_support_reason,
}


def _medium_term_reasons(s: TickerSnapshot) -> list[str]:
    reasons = []
    if s.minervini_pass:
        reasons.append("Cumple las 8 condiciones del Trend Template de Minervini")
    if s.stage == ta.Stage.STAGE_2 and s.rs_rating is not None and s.rs_rating >= 80:
        reasons.append("Fase 2 de Weinstein (avance) con RS Rating ≥ 80 - líder de mercado")
    if s.ma_cross == "golden":
        reasons.append("Golden cross reciente entre MA50 y MA200")
    return reasons


def _long_term_reasons(s: TickerSnapshot) -> list[str]:
    reasons = []
    if (
        s.rs_rating is not None
        and s.rs_rating >= 90
        and s.trend == ta.TrendState.UPTREND
        and s.dist_52w_high is not None
        and s.dist_52w_high >= -0.10
    ):
        reasons.append("RS Rating ≥ 90 sostenido, en tendencia alcista y cerca de máximos de 52 semanas")
    if s.mansfield_rs is not None and s.mansfield_rs > 0 and s.stage == ta.Stage.STAGE_2:
        reasons.append("Fuerza relativa positiva frente al S&P 500 (Mansfield RS) en Fase 2")
    return reasons


_MEDIUM_LONG_RULES = {
    MEDIUM_TERM: _medium_term_reasons,
    LONG_TERM: _long_term_reasons,
}

# Which of TickerSnapshot's cross-sectional inputs feed the percentile score
# (Segunda auditoría, Bloque 3) - an equal-weighted average of percentile
# ranks *within the day's own universe snapshot*, never a fixed baseline.
# None of these per-field weights are ablation-measured yet (unlike
# recommendation_engine.py's own factors) - this is deliberately a
# transparent, unweighted composite rather than an invented weighting scheme,
# consistent with CLAUDE.md's "no factor/weight without measured evidence".
PERCENTILE_SCORE_FIELDS: tuple[str, ...] = (
    "change_1w",
    "relative_volume",
    "relative_volume_trend",
    "atr_ratio_50d",
    "atr_multiple_sma21",
    "range_position_20d",
    "mansfield_rs_4w",
)


def _percentile_ranks(values: list[float | None]) -> list[float | None]:
    """0 (weakest) - 100 (strongest) percentile rank per value, skipping
    `None` entries (they stay `None` in the output, contributing nothing
    rather than a guessed midpoint). Tied values get the same (average)
    rank via `pandas.Series.rank` - a naive "first one in gets the lower
    rank" would silently favor whichever ticker happened to come first in
    the input list for every tie, which is exactly the kind of arbitrary,
    unearned tie-break this checklist is supposed to never have."""
    ranks = pd.Series(values, dtype="float64").rank(pct=True, na_option="keep") * 100.0
    return [None if pd.isna(r) else float(r) for r in ranks]


def setup_percentile_scores(snapshots: list[TickerSnapshot], setup: str) -> dict[str, float]:
    """Cross-sectional, setup-specific percentile score (0-100, higher =
    stronger), computed against *every* snapshot passed in (the day's whole
    universe), not just the subset that already matches `setup`'s own
    trigger rule - the trigger decides who qualifies, this decides ordering
    among qualifiers. Replaces RS Rating (12-month momentum) as the ordering
    criterion for the short-term tiers - RS Rating stays the criterion for
    the monthly tier, where a 12-month momentum read is the right question.

    The brief's own one explicit inversion: a 5-day return that's *more
    negative* scores higher for `oversold_bounce` (a deeper drop sets up a
    bigger bounce) - every other field/setup combination reads the same
    direction, since no ablation evidence exists yet for any other
    setup-specific inversion (see this module's docstring)."""
    field_ranks = {
        field: _percentile_ranks([getattr(s, field) for s in snapshots]) for field in PERCENTILE_SCORE_FIELDS
    }

    scores: dict[str, float] = {}
    for idx, s in enumerate(snapshots):
        component_ranks = []
        for field in PERCENTILE_SCORE_FIELDS:
            rank = field_ranks[field][idx]
            if rank is None:
                continue
            if setup == OVERSOLD_BOUNCE and field == "change_1w":
                rank = 100.0 - rank
            component_ranks.append(rank)
        if component_ranks:
            scores[s.ticker] = sum(component_ranks) / len(component_ranks)
    return scores


def _build_short_term_items(snapshots: list[TickerSnapshot]) -> list[WatchlistItem]:
    scores_by_setup = {setup: setup_percentile_scores(snapshots, setup) for setup in SHORT_TERM_SETUPS}
    items = []
    for snapshot in snapshots:
        for setup, detector in _SHORT_TERM_SETUP_DETECTORS.items():
            reason = detector(snapshot)
            if reason is None:
                continue
            items.append(
                WatchlistItem(
                    ticker=snapshot.ticker,
                    sector=snapshot.sector,
                    industry=snapshot.industry,
                    cap_tier=snapshot.cap_tier,
                    horizon=SHORT_TERM,
                    reasons=[reason],
                    snapshot=snapshot,
                    setup=setup,
                    percentile_score=scores_by_setup[setup].get(snapshot.ticker),
                )
            )
    return items


def _build_medium_long_items(snapshots: list[TickerSnapshot], horizon: str) -> list[WatchlistItem]:
    rule = _MEDIUM_LONG_RULES[horizon]
    items = []
    for snapshot in snapshots:
        reasons = rule(snapshot)
        if reasons:
            items.append(
                WatchlistItem(
                    ticker=snapshot.ticker,
                    sector=snapshot.sector,
                    industry=snapshot.industry,
                    cap_tier=snapshot.cap_tier,
                    horizon=horizon,
                    reasons=reasons,
                    snapshot=snapshot,
                )
            )
    return items


def _sort_key(item: WatchlistItem) -> tuple[bool, float]:
    primary = item.percentile_score if item.percentile_score is not None else item.snapshot.rs_rating
    return (primary is None, primary if primary is not None else 0.0)


def build_watchlist(snapshots: list[TickerSnapshot], horizon: str | None = None) -> list[WatchlistItem]:
    horizons = [horizon] if horizon else [SHORT_TERM, MEDIUM_TERM, LONG_TERM]
    items: list[WatchlistItem] = []
    if SHORT_TERM in horizons:
        items.extend(_build_short_term_items(snapshots))
    for h in (MEDIUM_TERM, LONG_TERM):
        if h in horizons:
            items.extend(_build_medium_long_items(snapshots, h))
    return sorted(items, key=_sort_key, reverse=True)
