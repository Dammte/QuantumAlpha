"""Market-wide context: broad index comparison and VIX regime - the "what's
the weather like" layer that sits above individual-stock analysis.

2026-09: the Fear & Greed proxy (6 components), the dollar-liquidity proxy
and the FRED yield-curve/unemployment/CPI read were retired (see
docs/quant_methodology.md) - no evidence any of them predicted anything at
this portfolio's actual holding horizon, and each added real cost (a second
OHLCV fetch, an external API call) to a panel meant to be a fast daily
check. VIX survives: it directly sizes the Chandelier Exit's volatility
regime input elsewhere in the app and its own extreme readings
("pánico"/"crisis") are still a defensible, if blunt, risk-off signal.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from app.domain.models.ticker_info import NewsArticle
from app.services import technical_analysis as ta
from app.services.market_data_service import MarketDataService
from app.services.market_universe import BENCHMARK_INDICES, BENCHMARK_TICKER, VIX_3M_TICKER, VIX_TICKER

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


REGIME_FAVORABLE = "favorable"
REGIME_CAUTION = "precaucion"
REGIME_AVOID = "evitar"

_VIX_CAUTION_POINTS = {"complacencia": 0, "normal": 0, "miedo elevado": 1, "pánico": 2, "crisis": 3}


@dataclass(frozen=True, slots=True)
class MarketRegime:
    verdict: str  # "favorable" | "precaucion" | "evitar"
    headline: str
    reasons: list[str]


def assess_market_regime(vix: VixSnapshot) -> MarketRegime:
    """Boils VIX regime/term structure down into a single "should I be
    trading today" read. A VIX reading in "pánico"/"crisis" territory forces
    "evitar" outright, independent of anything else - standard risk-off
    territory. Backwardation (near-term VIX futures priced above the 3-month
    contract) adds one more point of caution on top of the regime itself.

    2026-09: this used to also fold in a Fear & Greed proxy, a dollar-
    liquidity proxy and the FRED Treasury yield curve - all three retired
    (see module docstring and docs/quant_methodology.md)."""
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

    if vix.regime in ("pánico", "crisis") or caution_points >= 3:
        verdict = REGIME_AVOID
        headline = (
            "Volatilidad/riesgo elevado - considera esperar antes de abrir posiciones nuevas, o reducir "
            "el tamaño de las que abras."
        )
    elif caution_points >= 1:
        verdict = REGIME_CAUTION
        headline = "Cierta cautela recomendada - hay señales de tensión en volatilidad. Sé selectivo."
    else:
        verdict = REGIME_FAVORABLE
        headline = "Condiciones normales para operar - sin señales de estrés elevado en volatilidad."

    if not reasons:
        reasons.append("Sin señales de estrés en VIX en este momento")

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
            return VixSnapshot(None, None, ta.vix_regime(None), None)

        vix_close = vix_df["close"]
        level = float(vix_close.iloc[-1])
        sma50 = ta.sma(vix_close, 50).iloc[-1]
        sma50_val = None if pd.isna(sma50) else float(sma50)

        term_structure = None
        vix3m_df = ohlcv.get(VIX_3M_TICKER)
        if vix3m_df is not None and not vix3m_df.empty:
            vix3m_level = float(vix3m_df["close"].iloc[-1])
            term_structure = "backwardation (estrés)" if level > vix3m_level else "contango (normal)"

        return VixSnapshot(
            level=level, sma50=sma50_val, regime=ta.vix_regime(level), term_structure=term_structure
        )

    def get_market_news(self, limit: int = MARKET_NEWS_LIMIT) -> list[NewsArticle]:
        """General market-wide headlines (S&P 500 news, which naturally surfaces
        macro/geopolitical/earnings-season stories that move the whole tape) -
        shown as-is for the user to read and judge, not auto-classified into a
        bullish/bearish signal (no reliable way to score "is this headline
        good or bad for stocks" from the title alone)."""
        return self.market_data.get_ticker_news(BENCHMARK_TICKER, limit)
