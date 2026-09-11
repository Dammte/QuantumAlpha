"""Builds the "acciones a revisar" watchlist: scans the universe snapshot against
rule bundles grouped by investment horizon (short/medium/long), so a name only
shows up once its technicals actually line up - the user reviews the list and
decides whether to act, rather than hunting through the whole screener by hand.

Segunda auditoría, Bloque 3: the short-term horizon used to OR three unrelated
setups (a 52-week breakout with volume, an oversold bounce, a confirmed-trend
continuation) into one blended reasons list - a name could match for any of
the three and there was no way to tell which, or to score them differently.
Each is now its own setup type, with its own card and its own cross-sectional
percentile score - a ticker matching two setups at once shows up as two
separate items, one per setup, each scored independently.

**Corrección (Tercera auditoría, Bloque F-5) - atribución de evidencia falsa,
encontrada y corregida, no repetida aquí**: la justificación original de este
cambio citaba "oversold_bounce +0,723pp @5d, p<0,0001; trend_continuation
-0,301pp, IC-IR -0,461 @5d" como si esos números validaran los *setups* de
este módulo. No lo hacen - son el resultado medido, en el estudio de
ablación, para dos factores de `recommendation_engine.py`
(`rsi_oversold_bounce`: RSI ≤ 30 y tendencia ≠ bajista; `adx_strong_trend`:
ADX ≥ 25 y +DI > -DI), reglas distintas de estos setups (`oversold_bounce`
aquí es RSI ≤ 35 y `change_1d > 0`; `trend_continuation` añade además
`change_1w > 0`). Los setups de este módulo **nunca se han medido de
verdad** - citar la evidencia de una regla distinta como si midiera el
diseño propio es exactamente lo que el resto de este proyecto prohíbe (ver
CLAUDE.md). Ahora sí se miden: `scripts/factor_ablation_study.py`'s
`segment_by_setup_type` + los cuatro campos de `backtest_engine.TripleBarrierLabel`
(win rate, expectancy en R, duración mediana, MAE p80 - ver `FactorSample` y
`docs/quant_methodology.md` para el detalle) - la evidencia real de estos
setups vive ahí, no en un número prestado de otra regla.
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

# Tercera auditoría, Bloque F-2: the "weekly" (MEDIUM_TERM) tier's three
# setup types - see _MEDIUM_TERM_SETUP_DETECTORS below for why these replace
# the old single OR-blob of Minervini 8/8 + Stage 2/RS≥80 + golden cross
# SMA50/SMA200 (months-scale signals on a tier meant for a *weekly*, not
# daily, review cadence).
FAST_GOLDEN_CROSS = "fast_golden_cross"
FAST_CROSS_IMMINENT = "fast_cross_imminent"
STAGE2_LEADER = "stage2_leader"
MEDIUM_TERM_SETUPS = (FAST_GOLDEN_CROSS, FAST_CROSS_IMMINENT, STAGE2_LEADER)

SETUP_LABELS.update(
    {
        FAST_GOLDEN_CROSS: "Golden cross confirmado (corto plazo)",
        FAST_CROSS_IMMINENT: "Cruce alcista próximo (corto plazo)",
        STAGE2_LEADER: "Fase 2 con liderazgo (RS ≥ 80)",
    }
)


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

    @property
    def setup_label(self) -> str | None:
        """Root-cause fix for a real drift bug (Tercera auditoría, recomendación
        FE-1 de la cuarta auditoría independiente): the frontend used to keep
        its own copy of this Spanish label in `format.js`'s `SETUP_LABELS`,
        completely disconnected from this module's own `SETUP_LABELS` - it
        silently fell behind by three setups (Bloque F-2's weekly tier) until
        someone happened to notice raw snake_case in the UI. Computing the
        label server-side, from this single source of truth, and sending it
        over the wire makes that whole class of bug structurally impossible -
        the frontend's own map becomes a defensive fallback, never the source
        of truth."""
        return SETUP_LABELS.get(self.setup) if self.setup else None


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


def _fast_golden_cross_reason(s: TickerSnapshot) -> str | None:
    if s.ma_cross_short == "golden":
        return "Golden cross confirmado en el par corto (media rápida sobre la de 50 sesiones)"
    return None


def _fast_cross_imminent_reason(s: TickerSnapshot) -> str | None:
    imminent = s.imminent_cross_short_term
    if imminent is not None and imminent.direction == "golden":
        return f"Cruce alcista de medias (corto plazo) proyectado en ~{imminent.bars_until} sesiones"
    return None


def _stage2_leader_reason(s: TickerSnapshot) -> str | None:
    if s.stage == ta.Stage.STAGE_2 and s.rs_rating is not None and s.rs_rating >= 80:
        return "Fase 2 de Weinstein (avance) con RS Rating ≥ 80 - líder de mercado"
    return None


_MEDIUM_TERM_SETUP_DETECTORS = {
    FAST_GOLDEN_CROSS: _fast_golden_cross_reason,
    FAST_CROSS_IMMINENT: _fast_cross_imminent_reason,
    STAGE2_LEADER: _stage2_leader_reason,
}


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


# Which of TickerSnapshot's cross-sectional inputs feed the percentile score
# (Segunda auditoría, Bloque 3) - an equal-weighted average of percentile
# ranks *within the day's own universe snapshot*, never a fixed baseline.
# None of these per-field weights are ablation-measured yet (unlike
# recommendation_engine.py's own factors) - this is deliberately a
# transparent, unweighted composite rather than an invented weighting scheme,
# consistent with CLAUDE.md's "no factor/weight without measured evidence".
# Kept as the fallback composite for any setup without its own entry in
# SETUP_PERCENTILE_FIELDS below (Tercera auditoría, Bloque F-4) - should
# never actually be needed since every real setup has one, but a missing key
# falls back to this instead of raising.
PERCENTILE_SCORE_FIELDS: tuple[str, ...] = (
    "change_1w",
    "relative_volume",
    "relative_volume_trend",
    "atr_ratio_50d",
    "atr_multiple_sma21",
    "range_position_20d",
    "mansfield_rs_4w",
)

# Tercera auditoría, Bloque F-4: each setup's own field combination and sign
# - before this, every setup shared the exact same 7-field composite (only
# oversold_bounce inverting change_1w as a single hardcoded special case), so
# breakout_volume/trend_continuation/pullback_to_support produced
# mathematically identical scores for any ticker matching more than one of
# them (measured: AAPL scored 82.14 in both setups it triggered - the same
# number, computed 4 times to get 2 distinct answers). `True` means a higher
# raw value scores higher for that setup; `False` inverts it (same idea as
# oversold_bounce's existing change_1w inversion, now explicit and
# setup-specific instead of one special case). None of these per-field
# choices are ablation-measured yet either - same transparency-over-invented-
# precision standard as PERCENTILE_SCORE_FIELDS above, just no longer
# pretending four different setups measure the same thing.
SETUP_PERCENTILE_FIELDS: dict[str, dict[str, bool]] = {
    OVERSOLD_BOUNCE: {
        "change_1w": False,  # a deeper recent drop sets up a bigger bounce
        "range_position_20d": False,  # closer to its recent low, not its high
        "relative_volume": True,  # the bounce itself drawing real participation
    },
    BREAKOUT_VOLUME: {
        "relative_volume_trend": True,  # volume building into the breakout, not a one-off spike
        "relative_volume": True,
        "range_position_20d": True,  # near the top of its recent range
    },
    TREND_CONTINUATION: {
        "adx14": True,  # a stronger, more clearly confirmed trend
        "atr_multiple_sma21": False,  # penalize overextension above the 21-day average
        "mansfield_rs_4w": True,
    },
    PULLBACK_TO_SUPPORT: {
        "atr_multiple_sma21": False,  # a shallow, orderly pullback - not already blown far past the average
        "mansfield_rs_4w": True,  # still outperforming despite the dip
        "range_position_20d": False,  # nearer support than resistance
    },
    # The three weekly (MEDIUM_TERM) setups (Bloque F-2) - distinct combos
    # for the same reason, not left on the generic fallback.
    FAST_GOLDEN_CROSS: {
        "mansfield_rs_4w": True,
        "adx14": True,  # the trend the cross is confirming should itself be strengthening
        "relative_volume_trend": True,
    },
    FAST_CROSS_IMMINENT: {
        "mansfield_rs_4w": True,
        "atr_multiple_sma21": False,  # not overextended yet - the cross hasn't even confirmed
        "relative_volume_trend": True,
    },
    STAGE2_LEADER: {
        "mansfield_rs_4w": True,
        "range_position_20d": True,  # near highs, consistent with genuine leadership
        "relative_volume_trend": True,
    },
}


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


def percentile_rank_by_ticker(snapshots: list[TickerSnapshot], field: str) -> dict[str, float]:
    """Cross-sectional 0-100 percentile rank of one `TickerSnapshot` field,
    keyed by ticker (skipping tickers where the field is `None`) - built from
    the same `_percentile_ranks` primitive as `setup_percentile_scores`
    below, exposed directly for callers that need a single field's
    percentile rather than a blended composite. Tercera auditoría, Bloque
    F-3: a same-scale (0-100, cross-sectional) substitute for `rs_rating`
    (which is itself only ever an IBD-style 1-99 percentile) - a raw
    3-12-month RS Rating shouldn't score a 5-21 day trade the same way the
    5-21-day setup percentile already does. 2026-09: its original caller
    (`premium_watchlist_service.py`) was retired (see
    docs/quant_methodology.md); kept as a small, independently tested
    building block for whatever surface (Fase 5's Radar) needs this next."""
    values = [getattr(s, field) for s in snapshots]
    ranks = _percentile_ranks(values)
    return {s.ticker: r for s, r in zip(snapshots, ranks, strict=True) if r is not None}


def setup_percentile_scores(snapshots: list[TickerSnapshot], setup: str) -> dict[str, float]:
    """Cross-sectional, setup-specific percentile score (0-100, higher =
    stronger), computed against *every* snapshot passed in (the day's whole
    universe), not just the subset that already matches `setup`'s own
    trigger rule - the trigger decides who qualifies, this decides ordering
    among qualifiers. Replaces RS Rating (12-month momentum) as the ordering
    criterion for the short-term tiers - RS Rating stays the criterion for
    the monthly tier, where a 12-month momentum read is the right question.

    Tercera auditoría, Bloque F-4: each setup uses its own field combination
    and sign (`SETUP_PERCENTILE_FIELDS`) instead of one shared 7-field
    composite - see that dict for the per-setup reasoning and the real bug
    this replaces (every setup but oversold_bounce scored identically). A
    setup with no entry there (shouldn't happen - every real setup key has
    one) falls back to `PERCENTILE_SCORE_FIELDS`, unweighted, no inversion,
    rather than raising."""
    field_signs = SETUP_PERCENTILE_FIELDS.get(setup) or dict.fromkeys(PERCENTILE_SCORE_FIELDS, True)
    field_ranks = {field: _percentile_ranks([getattr(s, field) for s in snapshots]) for field in field_signs}

    scores: dict[str, float] = {}
    for idx, s in enumerate(snapshots):
        component_ranks = []
        for field, higher_is_better in field_signs.items():
            rank = field_ranks[field][idx]
            if rank is None:
                continue
            component_ranks.append(rank if higher_is_better else 100.0 - rank)
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


def _build_medium_term_items(snapshots: list[TickerSnapshot]) -> list[WatchlistItem]:
    """Tercera auditoría, Bloque F-2: split by setup type, same architecture
    as `_build_short_term_items` - a ticker matching more than one weekly
    setup shows up once per setup, each with its own cross-sectional
    percentile score, instead of one blended reasons list with no way to
    tell which condition actually fired or to rank candidates against each
    other."""
    scores_by_setup = {setup: setup_percentile_scores(snapshots, setup) for setup in MEDIUM_TERM_SETUPS}
    items = []
    for snapshot in snapshots:
        for setup, detector in _MEDIUM_TERM_SETUP_DETECTORS.items():
            reason = detector(snapshot)
            if reason is None:
                continue
            items.append(
                WatchlistItem(
                    ticker=snapshot.ticker,
                    sector=snapshot.sector,
                    industry=snapshot.industry,
                    cap_tier=snapshot.cap_tier,
                    horizon=MEDIUM_TERM,
                    reasons=[reason],
                    snapshot=snapshot,
                    setup=setup,
                    percentile_score=scores_by_setup[setup].get(snapshot.ticker),
                )
            )
    return items


def _build_long_term_items(snapshots: list[TickerSnapshot]) -> list[WatchlistItem]:
    items = []
    for snapshot in snapshots:
        reasons = _long_term_reasons(snapshot)
        if reasons:
            items.append(
                WatchlistItem(
                    ticker=snapshot.ticker,
                    sector=snapshot.sector,
                    industry=snapshot.industry,
                    cap_tier=snapshot.cap_tier,
                    horizon=LONG_TERM,
                    reasons=reasons,
                    snapshot=snapshot,
                )
            )
    return items


def _sort_key(item: WatchlistItem) -> tuple[bool, float]:
    """Ascending key: score present (False) sorts before score missing (True),
    and higher scores sort first *within* the present group via negation.
    Deliberately never paired with `sorted(..., reverse=True)` at the call
    site - reverse=True flips a tuple's leading bool too, which is exactly
    how this used to put unscored items (no percentile_score, no rs_rating -
    a ticker with 200-252 bars of history, past the screener's own minimum
    but short of what rs_rating needs) at the *top* of the watchlist instead
    of the bottom (Tercera auditoría, Bloque A-4) - rewarding the absence of
    data instead of its presence."""
    primary = item.percentile_score if item.percentile_score is not None else item.snapshot.rs_rating
    if primary is None:
        return (True, 0.0)
    return (False, -primary)


def build_watchlist(snapshots: list[TickerSnapshot], horizon: str | None = None) -> list[WatchlistItem]:
    horizons = [horizon] if horizon else [SHORT_TERM, MEDIUM_TERM, LONG_TERM]
    items: list[WatchlistItem] = []
    if SHORT_TERM in horizons:
        items.extend(_build_short_term_items(snapshots))
    if MEDIUM_TERM in horizons:
        items.extend(_build_medium_term_items(snapshots))
    if LONG_TERM in horizons:
        items.extend(_build_long_term_items(snapshots))
    return sorted(items, key=_sort_key)
