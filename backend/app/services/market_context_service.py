"""Market-wide context: broad index comparison, VIX regime, a Fear & Greed proxy,
and a dollar-liquidity proxy - the "what's the weather like" layer that sits above
individual-stock analysis.

Where CNN's real Fear & Greed Index uses 7 components (see cnn.com/markets/fear-and-greed),
two require data this app doesn't have a source for and are approximated or dropped,
each documented at the point it's computed:
  - Put/Call ratio: no options-flow data source -> dropped (6 components, not 7).
  - Stock Price Breadth (McClellan Volume Summation, real NYSE advance/decline volume):
    approximated with this app's own ~170-ticker universe's % above its 50-day SMA,
    which is breadth *participation*, not the true cumulative McClellan line.
The other components (momentum, new highs/lows, VIX vs its average, safe-haven and
junk-bond demand) are computed from real yfinance data for the tickers involved.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from app.domain.models.ticker_info import NewsArticle
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import technical_analysis as ta
from app.services.macro_data_service import MacroSnapshot
from app.services.market_data_service import MarketDataService
from app.services.market_universe import (
    BENCHMARK_INDICES,
    BENCHMARK_TICKER,
    MACRO_TICKERS,
    VIX_3M_TICKER,
    VIX_TICKER,
)

MARKET_NEWS_LIMIT = 8

HISTORY_DAYS = 400


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    name: str
    ticker: str
    price: float | None
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_1y: float | None
    trend: str | None


@dataclass(frozen=True, slots=True)
class VixSnapshot:
    level: float | None
    sma50: float | None
    regime: str
    term_structure: str | None


@dataclass(frozen=True, slots=True)
class FearGreed:
    score: float
    label: str
    components: dict[str, float]


@dataclass(frozen=True, slots=True)
class Liquidity:
    proxy_ticker: str
    trend: str
    headwind: bool


REGIME_FAVORABLE = "favorable"
REGIME_CAUTION = "precaucion"
REGIME_AVOID = "evitar"

_VIX_CAUTION_POINTS = {"complacencia": 0, "normal": 0, "miedo elevado": 1, "pánico": 2, "crisis": 3}


@dataclass(frozen=True, slots=True)
class MarketRegime:
    verdict: str  # "favorable" | "precaucion" | "evitar"
    headline: str
    reasons: list[str]


def _normalize(value: float | None, lo: float, hi: float, invert: bool = False) -> float:
    """Linearly maps `value` from [lo, hi] to [0, 100], clamped at the ends."""
    if value is None:
        return 50.0
    span = hi - lo
    pct = 0.0 if span == 0 else (value - lo) / span
    pct = max(0.0, min(1.0, pct))
    score = pct * 100
    return 100 - score if invert else score


def vix_regime(level: float | None) -> str:
    if level is None:
        return "desconocido"
    if level < 12:
        return "complacencia"
    if level < 20:
        return "normal"
    if level < 30:
        return "miedo elevado"
    if level < 40:
        return "pánico"
    return "crisis"


def fear_greed_label(score: float) -> str:
    if score < 25:
        return "Miedo extremo"
    if score < 45:
        return "Miedo"
    if score <= 55:
        return "Neutral"
    if score <= 75:
        return "Codicia"
    return "Codicia extrema"


def assess_market_regime(
    vix: VixSnapshot, fear_greed: FearGreed, liquidity: Liquidity, macro: MacroSnapshot | None = None
) -> MarketRegime:
    """Synthesizes the pieces `MarketContextService` already computes on every
    request (VIX regime/term structure, Fear & Greed, dollar liquidity, plus
    the Treasury yield curve when FRED is configured) into a single "should I
    be trading today" read - the raw numbers were already there, just never
    boiled down into an actual verdict. A VIX reading in "pánico"/"crisis"
    territory (>=30) forces "evitar" outright, independent of anything else:
    that alone is standard risk-off territory. Otherwise, corroborating stress
    signals (elevated VIX, backwardation, sentiment at an extreme in *either*
    direction, a tightening-liquidity headwind, an inverted yield curve)
    accumulate into caution or avoid."""
    caution_points = _VIX_CAUTION_POINTS.get(vix.regime, 0)
    reasons: list[str] = []

    if caution_points > 0:
        level_note = f" ({vix.level:.1f})" if vix.level is not None else ""
        reasons.append(f"VIX en régimen de '{vix.regime}'{level_note}")

    if vix.term_structure is not None and vix.term_structure.startswith("backwardation"):
        caution_points += 1
        reasons.append(
            "Curva de VIX en backwardation: el mercado paga más por protección a corto plazo que a "
            "medio plazo - señal de estrés inmediato"
        )

    if fear_greed.label == "Miedo extremo":
        caution_points += 1
        reasons.append(f"Fear & Greed en miedo extremo ({fear_greed.score:.0f}/100)")
    elif fear_greed.label == "Codicia extrema":
        caution_points += 1
        reasons.append(f"Fear & Greed en codicia extrema ({fear_greed.score:.0f}/100) - riesgo de sobreextensión")

    if liquidity.headwind:
        caution_points += 1
        reasons.append(f"Liquidez en dólares como viento en contra: {liquidity.trend}")

    if macro is not None and macro.yield_curve_inverted:
        caution_points += 1
        spread_note = f" ({macro.yield_curve_spread:.2f})" if macro.yield_curve_spread is not None else ""
        reasons.append(
            f"Curva de tipos invertida (10 años - 2 años){spread_note} - señal clásica de alerta de "
            "recesión, aunque con un plazo histórico muy variable (6-24 meses)"
        )

    if vix.regime in ("pánico", "crisis") or caution_points >= 3:
        verdict = REGIME_AVOID
        headline = (
            "Volatilidad/riesgo elevado - considera esperar antes de abrir posiciones nuevas, o reducir "
            "el tamaño de las que abras."
        )
    elif caution_points >= 1:
        verdict = REGIME_CAUTION
        headline = "Cierta cautela recomendada - hay señales de tensión, pero no generalizadas. Sé selectivo."
    else:
        verdict = REGIME_FAVORABLE
        headline = (
            "Condiciones normales para operar - sin señales de estrés elevado en volatilidad, "
            "sentimiento o liquidez."
        )

    if not reasons:
        reasons.append("Sin señales de estrés en VIX, sentimiento o liquidez en este momento")

    return MarketRegime(verdict=verdict, headline=headline, reasons=reasons)


class MarketContextService:
    def __init__(self, market_data: MarketDataService) -> None:
        self.market_data = market_data

    def _date_range(self) -> tuple[date, date]:
        end = date.today()
        return end - timedelta(days=HISTORY_DAYS), end

    def get_indices(self) -> list[IndexSnapshot]:
        start, end = self._date_range()
        tickers = list(BENCHMARK_INDICES.values())
        ohlcv = self.market_data.get_bulk_ohlcv(tickers, start, end)

        results = []
        for name, ticker in BENCHMARK_INDICES.items():
            df = ohlcv.get(ticker)
            if df is None or df.empty:
                results.append(IndexSnapshot(name, ticker, None, None, None, None, None, None, trend=None))
                continue
            close = df["close"]
            price = float(close.iloc[-1])
            sma20 = ta.sma(close, 20).iloc[-1]
            sma50 = ta.sma(close, 50).iloc[-1]
            sma200 = ta.sma(close, 200).iloc[-1]
            trend = ta.classify_trend(
                price,
                None if pd.isna(sma20) else float(sma20),
                None if pd.isna(sma50) else float(sma50),
                None if pd.isna(sma200) else float(sma200),
            )
            results.append(
                IndexSnapshot(
                    name=name,
                    ticker=ticker,
                    price=price,
                    change_1d=ta.pct_change_over(close, 1),
                    change_1w=ta.pct_change_over(close, 5),
                    change_1m=ta.pct_change_over(close, 21),
                    change_3m=ta.pct_change_over(close, 63),
                    change_1y=ta.pct_change_over(close, 252),
                    trend=trend.value,
                )
            )
        return results

    def get_vix(self) -> VixSnapshot:
        start, end = self._date_range()
        ohlcv = self.market_data.get_bulk_ohlcv([VIX_TICKER, VIX_3M_TICKER], start, end)
        vix_df = ohlcv.get(VIX_TICKER)
        if vix_df is None or vix_df.empty:
            return VixSnapshot(None, None, vix_regime(None), None)

        vix_close = vix_df["close"]
        level = float(vix_close.iloc[-1])
        sma50 = ta.sma(vix_close, 50).iloc[-1]
        sma50_val = None if pd.isna(sma50) else float(sma50)

        term_structure = None
        vix3m_df = ohlcv.get(VIX_3M_TICKER)
        if vix3m_df is not None and not vix3m_df.empty:
            vix3m_level = float(vix3m_df["close"].iloc[-1])
            term_structure = "backwardation (estrés)" if level > vix3m_level else "contango (normal)"

        return VixSnapshot(level=level, sma50=sma50_val, regime=vix_regime(level), term_structure=term_structure)

    def get_fear_greed(self, universe_snapshot: list[TickerSnapshot]) -> FearGreed:
        start, end = self._date_range()
        tickers = list(MACRO_TICKERS.values())
        ohlcv = self.market_data.get_bulk_ohlcv(tickers, start, end)

        def close_of(key: str) -> pd.Series | None:
            df = ohlcv.get(MACRO_TICKERS[key])
            return df["close"] if df is not None and not df.empty else None

        spx = close_of("spx")
        vix = close_of("vix")
        tlt = close_of("treasury_long")
        hyg = close_of("high_yield_bonds")
        lqd = close_of("investment_grade_bonds")

        components: dict[str, float] = {}

        # 1. Momentum: SPX vs its 125-day SMA.
        if spx is not None:
            sma125 = ta.sma(spx, 125).iloc[-1]
            if not pd.isna(sma125) and sma125:
                components["momentum"] = _normalize(float(spx.iloc[-1]) / float(sma125) - 1, -0.08, 0.08)

        # 2. Strength: new-highs vs new-lows within our own universe (proxy for NYSE-wide).
        if universe_snapshot:
            highs = sum(1 for s in universe_snapshot if s.dist_52w_high is not None and s.dist_52w_high >= -0.01)
            lows = sum(1 for s in universe_snapshot if s.dist_52w_low is not None and s.dist_52w_low <= 0.01)
            net = (highs - lows) / len(universe_snapshot)
            components["strength"] = _normalize(net, -0.2, 0.2)

        # 3. Breadth participation proxy (% of our universe above its 50-day SMA) - stands in
        #    for the real McClellan Volume Summation Index, which needs NYSE-wide A/D volume.
        if universe_snapshot:
            above_sma50 = sum(1 for s in universe_snapshot if s.sma50 is not None and s.price > s.sma50)
            components["breadth"] = _normalize(above_sma50 / len(universe_snapshot), 0.3, 0.7)

        # 4. Volatility: VIX vs its own 50-day SMA - VIX above its average = fear (invert).
        if vix is not None:
            vix_sma50 = ta.sma(vix, 50).iloc[-1]
            if not pd.isna(vix_sma50) and vix_sma50:
                components["volatility"] = _normalize(
                    float(vix.iloc[-1]) / float(vix_sma50) - 1, -0.2, 0.2, invert=True
                )

        # 5. Safe-haven demand: stocks' 20d return vs long treasuries' 20d return.
        if spx is not None and tlt is not None:
            spx_ret = ta.pct_change_over(spx, 20)
            tlt_ret = ta.pct_change_over(tlt, 20)
            if spx_ret is not None and tlt_ret is not None:
                components["safe_haven"] = _normalize(spx_ret - tlt_ret, -0.05, 0.05)

        # 6. Junk-bond demand: high-yield vs investment-grade 20d return spread (risk appetite).
        if hyg is not None and lqd is not None:
            hyg_ret = ta.pct_change_over(hyg, 20)
            lqd_ret = ta.pct_change_over(lqd, 20)
            if hyg_ret is not None and lqd_ret is not None:
                components["junk_bond_demand"] = _normalize(hyg_ret - lqd_ret, -0.03, 0.03)

        score = sum(components.values()) / len(components) if components else 50.0
        return FearGreed(score=score, label=fear_greed_label(score), components=components)

    def get_liquidity(self) -> Liquidity:
        start, end = self._date_range()
        ticker = MACRO_TICKERS["dollar_index"]
        ohlcv = self.market_data.get_bulk_ohlcv([ticker], start, end)
        df = ohlcv.get(ticker)
        if df is None or df.empty:
            return Liquidity(proxy_ticker=ticker, trend="desconocido", headwind=False)

        close = df["close"]
        ema20 = ta.ema(close, 20).iloc[-1]
        price = float(close.iloc[-1])
        if pd.isna(ema20):
            return Liquidity(proxy_ticker=ticker, trend="desconocido", headwind=False)

        strengthening = price > float(ema20)
        trend = "dólar fortaleciéndose" if strengthening else "dólar debilitándose"
        # A strengthening dollar tightens global USD liquidity - a headwind for risk assets.
        return Liquidity(proxy_ticker=ticker, trend=trend, headwind=strengthening)

    def get_market_news(self, limit: int = MARKET_NEWS_LIMIT) -> list[NewsArticle]:
        """General market-wide headlines (S&P 500 news, which naturally surfaces
        macro/geopolitical/earnings-season stories that move the whole tape) -
        shown as-is for the user to read and judge, not auto-classified into a
        bullish/bearish signal (no reliable way to score "is this headline
        good or bad for stocks" from the title alone)."""
        return self.market_data.get_ticker_news(BENCHMARK_TICKER, limit)
