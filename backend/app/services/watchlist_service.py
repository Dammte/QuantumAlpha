"""Builds the "acciones a revisar" watchlist: scans the universe snapshot against
rule bundles grouped by investment horizon (short/medium/long), so a name only
shows up once its technicals actually line up - the user reviews the list and
decides whether to act, rather than hunting through the whole screener by hand.
"""

from dataclasses import dataclass

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import technical_analysis as ta

SHORT_TERM = "short"
MEDIUM_TERM = "medium"
LONG_TERM = "long"


@dataclass(frozen=True, slots=True)
class WatchlistItem:
    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    horizon: str
    reasons: list[str]
    snapshot: TickerSnapshot


def _short_term_reasons(s: TickerSnapshot) -> list[str]:
    reasons = []
    if s.dist_52w_high is not None and s.dist_52w_high >= -0.02 and s.relative_volume and s.relative_volume >= 1.3:
        reasons.append("Ruptura o cercanía a máximos de 52 semanas con volumen por encima de lo normal")
    if s.rsi14 is not None and s.rsi14 <= 35 and s.change_1d is not None and s.change_1d > 0:
        reasons.append("Rebote desde zona de sobreventa (RSI ≤ 35 y hoy en positivo)")
    if (
        s.adx14 is not None
        and s.adx14 >= 25
        and s.plus_di is not None
        and s.minus_di is not None
        and s.plus_di > s.minus_di
        and s.change_1w is not None
        and s.change_1w > 0
    ):
        reasons.append("Tendencia alcista fuerte y confirmada (ADX ≥ 25, +DI > -DI)")
    return reasons


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


_RULES = {
    SHORT_TERM: _short_term_reasons,
    MEDIUM_TERM: _medium_term_reasons,
    LONG_TERM: _long_term_reasons,
}


def build_watchlist(snapshots: list[TickerSnapshot], horizon: str | None = None) -> list[WatchlistItem]:
    horizons = [horizon] if horizon else [SHORT_TERM, MEDIUM_TERM, LONG_TERM]
    items = []
    for snapshot in snapshots:
        for h in horizons:
            reasons = _RULES[h](snapshot)
            if reasons:
                items.append(
                    WatchlistItem(
                        ticker=snapshot.ticker,
                        sector=snapshot.sector,
                        industry=snapshot.industry,
                        cap_tier=snapshot.cap_tier,
                        horizon=h,
                        reasons=reasons,
                        snapshot=snapshot,
                    )
                )
    return sorted(items, key=lambda i: (i.snapshot.rs_rating is None, i.snapshot.rs_rating), reverse=True)
