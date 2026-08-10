"""Full quant risk read on every held ticker: the *exact same* pipeline
"Analizar activo" runs on demand (recommendation, GARCH, Markov, Monte Carlo,
walk-forward backtest, Kelly sizing - see `compute_core_signals()` in
`ticker_analysis_service.py`), applied to real capital already on the table.

This used to be a deliberately lighter subset (no GARCH, no Monte Carlo, no
backtest, no Kelly) to keep one request scoring every holding fast. That
tradeoff is gone: the whole point of holding a position is knowing exactly
when to sell, add, or hold it, and a lighter read that could disagree with
what searching the same ticker individually would show is exactly the kind of
assumption that costs money. Every holding now gets the full suite; the only
concession to cost is that the OHLCV history for every holding + the
benchmark is still fetched in a single batched call (see `get_bulk_ohlcv`),
so scoring N holdings is one network round-trip, not N.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import technical_analysis as ta
from app.services.market_data_service import MarketDataService
from app.services.market_universe import VIX_TICKER, benchmark_for_ticker, currency_of
from app.services.ticker_analysis_service import HISTORY_YEARS, CoreTickerSignals, compute_core_signals

PROXIMITY_THRESHOLD = 0.03  # within 3% of a level counts as "close to it"

# Same idiom as MarketScreenerService/PremiumWatchlistService: the full quant
# suite per holding (GARCH, Markov, Monte Carlo, walk-forward backtest) is
# genuinely slow - ~5s per ticker measured in production - so a cold dashboard
# load with several holdings can take 30-40s. Caching per ticker means that
# cost is only paid once per CACHE_TTL, not on every dashboard reload, which is
# the actual common case. See PortfolioRiskService below.
#
# Cache misses are computed sequentially, not in a thread pool: an earlier
# version parallelized this with ThreadPoolExecutor, but on Render's
# CPU-constrained tier it made things *worse* - each Python thread spins up
# its own BLAS/OpenMP threads inside numpy/scipy (GARCH, Monte Carlo), and a
# handful of Python threads each oversubscribing a shared, throttled vCPU
# turned a 39s sequential response into a request that never completed at
# all. Caching already removes the cost on every reload but the very first
# one, which is the case that actually matters for a personal dashboard.
CACHE_TTL = timedelta(minutes=20)

EXIT_WARNING = "exit_warning"
ADD_CANDIDATE = "add_candidate"
WATCH = "watch"
HOLD = "hold"


@dataclass(frozen=True, slots=True)
class PositionRisk:
    ticker: str
    currency: str
    price: float
    trend: str
    stage: str | None
    ma_cross: str | None
    rs_rating: int | None
    nearest_support: ta.PriceLevel | None
    nearest_resistance: ta.PriceLevel | None
    signal: str
    score: int
    reasons: list[str]
    signals: CoreTickerSignals  # the full quant suite backing this signal


def assess_position_risk(
    ticker: str,
    df: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    rs_rating: int | None = None,
    vix_close: pd.Series | None = None,
) -> PositionRisk | None:
    signals = compute_core_signals(
        df["close"], df["high"], df["low"], df["volume"], benchmark_close, rs_rating, vix_close=vix_close
    )
    if signals is None:
        return None

    near_support = (
        signals.nearest_support is not None and abs(signals.nearest_support.distance_pct) <= PROXIMITY_THRESHOLD
    )
    near_resistance = (
        signals.nearest_resistance is not None
        and abs(signals.nearest_resistance.distance_pct) <= PROXIMITY_THRESHOLD
    )

    if signals.recommendation.verdict == "evitar":
        signal = EXIT_WARNING
    elif signals.recommendation.verdict == "comprar":
        signal = ADD_CANDIDATE
    elif near_support or near_resistance:
        signal = WATCH
    else:
        signal = HOLD

    reasons = [f"{f.label} ({f.points:+d})" for f in signals.recommendation.factors if f.triggered]
    if signal == WATCH:
        nearest = signals.nearest_support if near_support else signals.nearest_resistance
        kind_label = "soporte" if nearest.kind == "support" else "resistencia"
        pct = abs(nearest.distance_pct) * 100
        reasons.append(f"Precio a {pct:.1f}% de un nivel de {kind_label} en {nearest.price:.2f}")
    if not reasons:
        reasons.append("Sin señales técnicas relevantes en este momento")

    return PositionRisk(
        ticker=ticker,
        currency=currency_of(ticker),
        price=signals.price,
        trend=signals.trend.value,
        stage=signals.stage.value if signals.stage else None,
        ma_cross=signals.ma_cross,
        rs_rating=signals.rs_rating,
        nearest_support=signals.nearest_support,
        nearest_resistance=signals.nearest_resistance,
        signal=signal,
        score=signals.recommendation.score,
        reasons=reasons,
        signals=signals,
    )


def get_portfolio_positions_risk(
    tickers: list[str],
    market_data: MarketDataService,
    universe_snapshot: list[TickerSnapshot] | None = None,
) -> list[PositionRisk]:
    """Runs `assess_position_risk` for every ticker actually held, reusing the
    universe snapshot's RS Rating when a holding happens to be in a curated
    universe (most won't be - that's expected for a personal portfolio).
    `universe_snapshot` may combine both regions (US + Europe) - a personal
    portfolio isn't confined to one market, and each holding is benchmarked
    against whichever region it actually belongs to (see `benchmark_for_ticker`)."""
    if not tickers:
        return []

    rs_by_ticker = {s.ticker: s.rs_rating for s in universe_snapshot} if universe_snapshot else {}

    end = date.today()
    start = end - timedelta(days=365 * HISTORY_YEARS)
    benchmark_by_ticker = {ticker: benchmark_for_ticker(ticker) for ticker in tickers}
    # VIX rides along in the same batched call regardless of how many tickers
    # are held - one shared market-regime input, not fetched per position.
    fetch_list = [*tickers, *set(benchmark_by_ticker.values()), VIX_TICKER]
    ohlcv_by_ticker = market_data.get_bulk_ohlcv(fetch_list, start, end)
    vix_df = ohlcv_by_ticker.get(VIX_TICKER)
    vix_close = vix_df["close"] if vix_df is not None else None

    results = []
    for ticker in tickers:
        df = ohlcv_by_ticker.get(ticker)
        if df is None:
            continue
        benchmark_df = ohlcv_by_ticker.get(benchmark_by_ticker[ticker])
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
        risk = assess_position_risk(
            ticker, df, benchmark_close, rs_rating=rs_by_ticker.get(ticker), vix_close=vix_close
        )
        if risk is not None:
            results.append(risk)
    return results


class PortfolioRiskService:
    """Production-facing wrapper around `get_portfolio_positions_risk`: caches
    each ticker's `PositionRisk` for `CACHE_TTL` so the full quant suite is
    only recomputed once per ticker per TTL window, not on every dashboard
    reload. Registered as a singleton in `deps.py` so the cache is actually
    shared across requests, same as MarketScreenerService."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[datetime, PositionRisk]] = {}

    def get_positions_risk(
        self,
        tickers: list[str],
        market_data: MarketDataService,
        universe_snapshot: list[TickerSnapshot] | None = None,
        force_refresh: bool = False,
    ) -> list[PositionRisk]:
        if not tickers:
            return []

        now = datetime.now(UTC)
        fresh_by_ticker: dict[str, PositionRisk] = {}
        to_compute: list[str] = []
        for ticker in tickers:
            cached = None if force_refresh else self._cache.get(ticker)
            if cached is not None and now - cached[0] < CACHE_TTL:
                fresh_by_ticker[ticker] = cached[1]
            else:
                to_compute.append(ticker)

        if to_compute:
            rs_by_ticker = {s.ticker: s.rs_rating for s in universe_snapshot} if universe_snapshot else {}
            end = date.today()
            start = end - timedelta(days=365 * HISTORY_YEARS)
            benchmark_by_ticker = {ticker: benchmark_for_ticker(ticker) for ticker in to_compute}
            fetch_list = [*to_compute, *set(benchmark_by_ticker.values()), VIX_TICKER]
            ohlcv_by_ticker = market_data.get_bulk_ohlcv(fetch_list, start, end)
            vix_df = ohlcv_by_ticker.get(VIX_TICKER)
            vix_close = vix_df["close"] if vix_df is not None else None

            for ticker in to_compute:
                df = ohlcv_by_ticker.get(ticker)
                if df is None:
                    continue
                benchmark_df = ohlcv_by_ticker.get(benchmark_by_ticker[ticker])
                benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
                risk = assess_position_risk(
                    ticker, df, benchmark_close, rs_rating=rs_by_ticker.get(ticker), vix_close=vix_close
                )
                if risk is not None:
                    fresh_by_ticker[ticker] = risk
                    self._cache[ticker] = (now, risk)

        return [fresh_by_ticker[ticker] for ticker in tickers if ticker in fresh_by_ticker]
