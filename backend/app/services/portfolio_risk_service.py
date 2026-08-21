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

import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from app.domain.interfaces.position_signal_snapshot_repository import PositionSignalSnapshotRepositoryPort
from app.domain.interfaces.trade_plan_repository import TradePlanRepositoryPort
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.domain.models.trade_plan import TradePlan
from app.domain.models.transaction import Transaction
from app.services import exit_engine as ee
from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services import trade_manager as tm
from app.services import trade_plan_service as tps
from app.services.market_data_service import MarketDataService
from app.services.market_universe import VIX_TICKER, benchmark_for_ticker, currency_of
from app.services.recommendation_engine import ENGINE_VERSION
from app.services.ticker_analysis_service import HISTORY_YEARS, CoreTickerSignals, compute_core_signals

logger = logging.getLogger(__name__)

PROXIMITY_THRESHOLD = 0.03  # within 3% of a level counts as "close to it"
# How many recent bars count as "recent" for the exit engine's "falling from
# a real peak" reads (RSI/ADX) - two trading weeks, long enough to catch a
# genuine recent overbought/trending peak, short enough that a peak from a
# month ago doesn't still count as "recent".
RECENT_LOOKBACK = 10

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
    # Added for the independent exit engine (exit_engine.py, see its
    # docstring for why this is a *separate* read from `signal`/`score`/
    # `reasons` above, not a replacement) - all `None`/empty when the caller
    # doesn't supply portfolio context (`portfolio_id`/`transactions`/
    # `trade_plan_repo` below), so every existing caller of this function
    # keeps working unchanged.
    exit_urgency: str | None = None  # "exit_now" | "reduce" | "tighten_stop" | "watch" | "hold"
    exit_reasons: list[str] = field(default_factory=list)
    trade_plan: TradePlan | None = None
    r_multiple: float | None = None
    multi_timeframe: mtf.MultiTimeframeRead | None = None
    # trade_manager.py's scaled-exit read (Fase 3) - None under the same
    # conditions as the fields above (no portfolio context supplied, or no
    # open trade plan for this ticker).
    scaled_exit: tm.ScaledExitPlan | None = None
    # Closed daily bars since `trade_plan.entry_date` (Fase 7's position
    # detail card - "sesiones mantenidas"). Same None-when-no-portfolio-
    # context rule as the fields above; computed once already inside the
    # `if portfolio_id is not None...` block below (`tps.bars_held_since`),
    # just not previously threaded out of this function.
    bars_held: int | None = None


def _recent_max(series: pd.Series, lookback: int = RECENT_LOOKBACK) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-lookback:].max())


def assess_position_risk(
    ticker: str,
    df: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    rs_rating: int | None = None,
    vix_close: pd.Series | None = None,
    portfolio_id: int | None = None,
    transactions: list[Transaction] | None = None,
    trade_plan_repo: TradePlanRepositoryPort | None = None,
    position_signal_snapshot_repo: PositionSignalSnapshotRepositoryPort | None = None,
) -> PositionRisk | None:
    signals = compute_core_signals(
        df["close"],
        df["high"],
        df["low"],
        df["volume"],
        df["open"],
        benchmark_close,
        rs_rating,
        vix_close=vix_close,
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
    # A *confirmed* death cross already factors into the recommendation engine's
    # score (and from there, often the "evitar" verdict above). These are the
    # earlier, still-projected cases: nothing else has flagged this position
    # yet, but a pair of moving averages is converging - worth active
    # attention before it's a lagging confirmation, not after. Both the
    # SMA50/SMA200 (medium/long-term) and SMA20/SMA50 (short-term - relevant
    # for a position actively managed on a shorter horizon, which can turn
    # well before the longer-term picture does) versions count here. See
    # technical_analysis.detect_imminent_cross.
    imminent_death_cross = signals.imminent_cross is not None and signals.imminent_cross.direction == "death"
    imminent_death_cross_short = (
        signals.imminent_cross_short_term is not None and signals.imminent_cross_short_term.direction == "death"
    )
    # The one price-action pattern checked here rather than left to the
    # recommendation engine's score: a bearish engulfing candle is a
    # single-session event, not a multi-day trend read like everything else
    # this checklist scores - worth surfacing immediately on the position
    # that just printed one, not waiting for it to show up in slower-moving
    # indicators. See technical_analysis.detect_engulfing_pattern.
    bearish_engulfing = signals.candlestick_pattern == "bearish_engulfing"

    watch_worthy = (
        near_support or near_resistance or imminent_death_cross or imminent_death_cross_short or bearish_engulfing
    )
    if signals.recommendation.verdict == "evitar":
        signal = EXIT_WARNING
    elif signals.recommendation.verdict == "comprar":
        signal = ADD_CANDIDATE
    elif watch_worthy:
        signal = WATCH
    else:
        signal = HOLD

    # Deliberately *not* gated by which `signal` tier this landed in: an
    # add_candidate position with a strong short-term breakdown projected
    # (a "comprar" verdict absolutely can carry one - the recommendation
    # engine looks at the whole picture, this looks at one specific,
    # narrower thing) still deserves that context visible, not silently
    # dropped because the headline signal was upbeat. A real gap this
    # closes: a held position can score "comprar" (ADD_CANDIDATE) while
    # SMA20/SMA50 are actively converging toward a short-term death cross -
    # exactly the situation a short-term-managed position needs a heads-up
    # on, regardless of what the primary badge says.
    reasons = [f"{f.label} ({f.points:+d})" for f in signals.recommendation.factors if f.triggered]
    if near_support or near_resistance:
        nearest = signals.nearest_support if near_support else signals.nearest_resistance
        kind_label = "soporte" if nearest.kind == "support" else "resistencia"
        pct = abs(nearest.distance_pct) * 100
        reasons.append(f"Precio a {pct:.1f}% de un nivel de {kind_label} en {nearest.price:.2f}")

    if imminent_death_cross:
        reasons.append(
            f"Posible cruce de medias bajista (SMA50/SMA200) en ~{signals.imminent_cross.bars_until} "
            "sesiones si continúa la tendencia actual - vigilar de cerca"
        )
    elif signals.imminent_cross is not None and signals.imminent_cross.direction == "golden":
        reasons.append(
            f"Posible cruce de medias alcista (SMA50/SMA200) en ~{signals.imminent_cross.bars_until} "
            "sesiones si continúa la tendencia actual"
        )

    ict_short = signals.imminent_cross_short_term
    if imminent_death_cross_short:
        reasons.append(
            f"Posible cruce de medias bajista de corto plazo (SMA20/SMA50) en ~{ict_short.bars_until} "
            "sesiones - relevante si gestionas esta posición a corto plazo"
        )
    elif ict_short is not None and ict_short.direction == "golden":
        reasons.append(
            f"Posible cruce de medias alcista de corto plazo (SMA20/SMA50) en ~{ict_short.bars_until} sesiones"
        )

    if bearish_engulfing:
        reasons.append("Vela envolvente bajista en la última sesión - posible cambio de tendencia a corto plazo")
    elif signals.candlestick_pattern == "bullish_engulfing":
        reasons.append("Vela envolvente alcista en la última sesión")

    if not reasons:
        reasons.append("Sin señales técnicas relevantes en este momento")

    # --- Independent exit engine (D2/D3 fix - see exit_engine.py's docstring) ---
    # Only runs when the caller supplies portfolio context: a bare technical
    # read (e.g. from a caller that only wants "Analizar activo"-style
    # signals, or an as-yet-unupdated test double) still gets the legacy
    # verdict-based `signal` above unchanged, just without this layered on
    # top - `exit_urgency` stays `None` in that case, never a guess.
    exit_urgency: str | None = None
    exit_reasons: list[str] = []
    trade_plan: TradePlan | None = None
    r_multiple: float | None = None
    multi_timeframe: mtf.MultiTimeframeRead | None = None
    scaled_exit: tm.ScaledExitPlan | None = None
    bars_held: int | None = None

    if portfolio_id is not None and trade_plan_repo is not None and transactions is not None:
        multi_timeframe = mtf.analyze_multi_timeframe(df)
        plan = tps.ensure_trade_plan(trade_plan_repo, portfolio_id, ticker, transactions, df)
        closed = ta.closed_bars(df)
        if plan is not None and not closed.empty:
            trade_plan = plan
            exit_price = float(closed["close"].iloc[-1])
            held_quantity = tps.current_held_quantity(transactions, ticker)
            consecutive_below_sma50 = ta.consecutive_closes_below(closed["close"], ta.sma(closed["close"], 50))
            consecutive_below_fast_sma = ta.consecutive_closes_below(
                closed["close"], ta.sma(closed["close"], mtf.FAST_MA_PERIOD)
            )
            rsi_recent_max = _recent_max(ta.rsi(closed["close"]))
            adx_recent_max = _recent_max(ta.adx(closed["high"], closed["low"], closed["close"]))
            bars_held = tps.bars_held_since(closed, plan.entry_date)
            # Recomputed here on the *closed* bar, not reused from `signals`
            # (computed by compute_core_signals on the live/raw `df`) - the
            # exit engine's own `price` argument below is `exit_price`, the
            # same closed close. Reusing the live-based nearest_support would
            # mix bases: on a bullish overnight gap, a support level found
            # relative to today's (higher) live price can sit *above*
            # yesterday's (lower) closed price used in the comparison,
            # producing a false "rotura de soporte" purely from the gap, not
            # a real break.
            closed_levels = ta.support_resistance_levels(closed["high"], closed["low"], closed["close"])
            nearest_support_closed = min(
                (lv for lv in closed_levels if lv.kind == "support"), key=lambda lv: abs(lv.distance_pct),
                default=None,
            )
            nearest_resistance_closed = min(
                (lv for lv in closed_levels if lv.kind == "resistance"), key=lambda lv: abs(lv.distance_pct),
                default=None,
            )
            # average_cost defaults to this lot's own entry price - a DCA'd
            # position's true blended cost isn't threaded through here yet
            # (that needs Portfolio.positions, which this call chain doesn't
            # fetch today - see portfolios.py's /risk endpoint). Doesn't
            # affect the urgency verdict either way: evaluate_exit never
            # reads average_cost, only current_stop/initial_target/r_multiple.
            position_context = tps.build_position_context(
                plan, exit_price, quantity=held_quantity, average_cost=plan.entry_price, bars_held=bars_held
            )
            assessment = ee.evaluate_exit(
                price=exit_price,
                position=position_context,
                multi_timeframe=multi_timeframe,
                consecutive_closes_below_daily_sma50=consecutive_below_sma50,
                consecutive_closes_below_daily_sma_fast=consecutive_below_fast_sma,
                nearest_support=nearest_support_closed,
                nearest_resistance=nearest_resistance_closed,
                obv_divergence=signals.obv_divergence,
                relative_volume=signals.relative_volume,
                rsi14=signals.rsi14,
                rsi_recent_max=rsi_recent_max,
                adx14=signals.adx14,
                adx_recent_max=adx_recent_max,
                atr_multiple=signals.atr_multiple,
                candlestick_pattern=signals.candlestick_pattern,
            )
            exit_urgency = assessment.urgency.value
            exit_reasons = assessment.reasons
            r_multiple = position_context.r_multiple
            scaled_exit = tm.compute_scaled_exit_plan(
                r_multiple=r_multiple,
                quantity_held=held_quantity,
                initial_quantity=plan.initial_quantity,
                entry_price=plan.entry_price,
            )

            # Trailing-stop update (trade_manager.py, Chandelier Exit) -
            # computed and persisted *after* evaluate_exit ran, deliberately:
            # the stop-breach check above must judge today's close against
            # the stop that was already in force coming into today, never
            # one just widened/raised using today's own bar - otherwise a
            # stop "breach" could be an artifact of raising the stop, not a
            # real adverse move.
            vol_regime = signals.garch.regime if signals.garch is not None else None
            # Bounded to bars on/after entry - the highest high the Chandelier
            # trail should ever consider is one this trade actually lived
            # through. `closed["high"]` unfiltered can reach years before
            # entry; a position opened after a pullback from an even higher
            # pre-entry high would otherwise trail against that pre-entry
            # high, not the trade's own price action.
            high_since_entry = closed["high"][closed["high"].index.date >= plan.entry_date]
            trailing = tm.compute_trailing_stop(
                high_since_entry, ta.atr(closed["high"], closed["low"], closed["close"]),
                current_stop=plan.current_stop, r_multiple=r_multiple, vol_regime=vol_regime, price=exit_price,
            )
            if trailing.stop is not None and plan.id is not None:
                trade_plan_repo.update_trailing(plan.id, trailing.stop, position_context.highest_close_since_entry)
                trade_plan = replace(
                    plan,
                    current_stop=trailing.stop,
                    highest_close_since_entry=position_context.highest_close_since_entry,
                )

            # The exit engine's read outranks a "comprar" verdict for the
            # tiers that mean real trouble (protecting capital already on
            # the table matters more than "would this still be a fresh
            # buy") - this is the actual D2/D3 fix: previously nothing could
            # ever override ADD_CANDIDATE/HOLD except the buy-side score
            # itself. EXIT_NOW/REDUCE still own the strongest badge
            # (EXIT_WARNING). TIGHTEN_STOP/WATCH now also degrade an
            # ADD_CANDIDATE, not only HOLD - a "comprar" verdict (RS Rating,
            # fundamentals) that the exit engine has already flagged for
            # active monitoring must not keep reading "Añadir" as if nothing
            # had changed technically. Never the other way around: a
            # genuine EXIT_WARNING is never *downgraded* back to WATCH by a
            # softer tier.
            softer_tiers = (ee.ExitUrgency.TIGHTEN_STOP, ee.ExitUrgency.WATCH)
            if assessment.urgency in (ee.ExitUrgency.EXIT_NOW, ee.ExitUrgency.REDUCE):
                signal = EXIT_WARNING
            elif assessment.urgency in softer_tiers and signal != EXIT_WARNING:
                signal = WATCH

        # Fase 0 instrumentation: one row per genuinely fresh evaluation (this
        # function only runs on a cache miss - see PortfolioRiskService.get_positions_risk
        # below), capturing the *final* signal/exit_urgency after the
        # escalation above, not an intermediate value - see
        # PositionSignalSnapshotORM's docstring for why this data didn't
        # exist at all before this phase.
        if position_signal_snapshot_repo is not None:
            position_signal_snapshot_repo.save(
                portfolio_id=portfolio_id,
                ticker=ticker,
                signal=signal,
                exit_urgency=exit_urgency,
                score=signals.recommendation.score,
                price=signals.price,
                r_multiple=r_multiple,
                engine_version=ENGINE_VERSION,
            )

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
        exit_urgency=exit_urgency,
        exit_reasons=exit_reasons,
        trade_plan=trade_plan,
        r_multiple=r_multiple,
        multi_timeframe=multi_timeframe,
        scaled_exit=scaled_exit,
        bars_held=bars_held,
    )


def _safe_assess_position_risk(
    ticker: str,
    df: pd.DataFrame,
    benchmark_close: pd.Series | None,
    rs_rating: int | None,
    vix_close: pd.Series | None,
    portfolio_id: int | None = None,
    transactions: list[Transaction] | None = None,
    trade_plan_repo: TradePlanRepositoryPort | None = None,
    position_signal_snapshot_repo: PositionSignalSnapshotRepositoryPort | None = None,
) -> PositionRisk | None:
    """`assess_position_risk`, isolated: one holding's GARCH optimizer failing
    to converge, a backtest edge case, or any other numerical hiccup must
    never take the rest of a real portfolio's risk read down with it - a
    holding that fails to compute simply doesn't get a signal this round
    (rather than the whole request 500ing) and tries again next refresh."""
    try:
        return assess_position_risk(
            ticker,
            df,
            benchmark_close,
            rs_rating=rs_rating,
            vix_close=vix_close,
            portfolio_id=portfolio_id,
            transactions=transactions,
            trade_plan_repo=trade_plan_repo,
            position_signal_snapshot_repo=position_signal_snapshot_repo,
        )
    except Exception:
        logger.exception("Portfolio risk: skipping %s after a compute failure", ticker)
        return None


def get_portfolio_positions_risk(
    tickers: list[str],
    market_data: MarketDataService,
    universe_snapshot: list[TickerSnapshot] | None = None,
    portfolio_id: int | None = None,
    transactions: list[Transaction] | None = None,
    trade_plan_repo: TradePlanRepositoryPort | None = None,
    position_signal_snapshot_repo: PositionSignalSnapshotRepositoryPort | None = None,
) -> list[PositionRisk]:
    """Runs `assess_position_risk` for every ticker actually held, reusing the
    universe snapshot's RS Rating when a holding happens to be in a curated
    universe (most won't be - that's expected for a personal portfolio).
    `universe_snapshot` may combine both regions (US + Europe) - a personal
    portfolio isn't confined to one market, and each holding is benchmarked
    against whichever region it actually belongs to (see `benchmark_for_ticker`).
    `portfolio_id`/`transactions`/`trade_plan_repo` enable the exit engine
    (see `assess_position_risk`) - omit them for a bare technical read with
    no portfolio context (e.g. a caller that only needs the legacy signal)."""
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
        risk = _safe_assess_position_risk(
            ticker,
            df,
            benchmark_close,
            rs_by_ticker.get(ticker),
            vix_close,
            portfolio_id=portfolio_id,
            transactions=transactions,
            trade_plan_repo=trade_plan_repo,
            position_signal_snapshot_repo=position_signal_snapshot_repo,
        )
        if risk is not None:
            results.append(risk)
    return results


class PortfolioRiskService:
    """Production-facing wrapper around `get_portfolio_positions_risk`: caches
    each (portfolio, ticker)'s `PositionRisk` for `CACHE_TTL` so the full
    quant suite is only recomputed once per TTL window, not on every
    dashboard reload. Registered as a singleton in `deps.py` so the cache is
    actually shared across requests, same as MarketScreenerService.

    Cached by (portfolio_id, ticker), not just ticker: once a position's
    trade plan (its own stop/target/entry) participates in the read, the
    *same* ticker held in two different portfolios can legitimately produce
    two different results - caching by ticker alone would leak one
    portfolio's trade plan into another's response."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int | None, str], tuple[datetime, PositionRisk]] = {}

    def get_positions_risk(
        self,
        tickers: list[str],
        market_data: MarketDataService,
        universe_snapshot: list[TickerSnapshot] | None = None,
        force_refresh: bool = False,
        portfolio_id: int | None = None,
        transactions: list[Transaction] | None = None,
        trade_plan_repo: TradePlanRepositoryPort | None = None,
        position_signal_snapshot_repo: PositionSignalSnapshotRepositoryPort | None = None,
    ) -> list[PositionRisk]:
        if not tickers:
            return []

        now = datetime.now(UTC)
        fresh_by_ticker: dict[str, PositionRisk] = {}
        to_compute: list[str] = []
        for ticker in tickers:
            cached = None if force_refresh else self._cache.get((portfolio_id, ticker))
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
                risk = _safe_assess_position_risk(
                    ticker,
                    df,
                    benchmark_close,
                    rs_by_ticker.get(ticker),
                    vix_close,
                    portfolio_id=portfolio_id,
                    transactions=transactions,
                    trade_plan_repo=trade_plan_repo,
                    position_signal_snapshot_repo=position_signal_snapshot_repo,
                )
                if risk is not None:
                    fresh_by_ticker[ticker] = risk
                    self._cache[(portfolio_id, ticker)] = (now, risk)

        return [fresh_by_ticker[ticker] for ticker in tickers if ticker in fresh_by_ticker]
