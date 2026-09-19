"""Full quant risk read on every held ticker: the *exact same* pipeline
"Analizar activo" runs on demand (the levels/triggers gate, triple-barrier
backtest - see `compute_core_signals()` in `ticker_analysis_service.py`),
applied to real capital already on the table.

This used to be a deliberately lighter subset to keep one request scoring
every holding fast, then a much heavier one once GARCH/Markov/Monte
Carlo/Kelly were added (see docs/quant_methodology.md for that history) -
those were removed again 2026-09 for lack of cross-sectional evidence, which
also removes most of the latency that pass added. The whole point of holding
a position is knowing exactly when to sell, add, or hold it, and a lighter
read that could disagree with what searching the same ticker individually
would show is exactly the kind of assumption that costs money - every
holding still gets the exact same analysis, just a cheaper one now. The
OHLCV history for every holding + the benchmark is still fetched in a single
batched call (see `get_bulk_ohlcv`), so scoring N holdings is one network
round-trip, not N.

2026-09 (reconstruction, Fase 4): `signal` (EXIT_WARNING/ADD_CANDIDATE/WATCH/
HOLD below) is now driven by `signals.gate.passes` instead of
`signals.recommendation.verdict` - see `assess_position_risk`'s own docstring
for the one deliberate behavior change this brings (EXIT_WARNING no longer
ever comes from the buy-side signal alone, only from the independent exit
engine).
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
from app.services.levels_engine import GATE_VERSION
from app.services.market_data_service import MarketDataService
from app.services.market_universe import (
    VIX_TICKER,
    benchmark_for_ticker,
    closed_bar_cutoff_for_ticker,
    currency_of,
)
from app.services.ticker_analysis_service import HISTORY_YEARS, CoreTickerSignals, compute_core_signals

logger = logging.getLogger(__name__)

PROXIMITY_THRESHOLD = 0.03  # within 3% of a level counts as "close to it"
# How many recent bars count as "recent" for the exit engine's "falling from
# a real peak" reads (RSI/ADX) - two trading weeks, long enough to catch a
# genuine recent overbought/trending peak, short enough that a peak from a
# month ago doesn't still count as "recent".
RECENT_LOOKBACK = 10

# Same idiom as MarketScreenerService/PremiumWatchlistService: the per-holding
# suite (recommendation, multi-timeframe, walk-forward backtest) still isn't
# free - a cold dashboard load with several holdings adds up. Caching per
# ticker means that cost is only paid once per CACHE_TTL, not on every
# dashboard reload, which is the actual common case. See PortfolioRiskService
# below.
#
# Cache misses are computed sequentially, not in a thread pool: an earlier
# version parallelized this with ThreadPoolExecutor, back when this suite
# also ran GARCH and Monte Carlo per ticker (both since removed - see module
# docstring), but on Render's CPU-constrained tier it made things *worse* -
# each Python thread spins up its own BLAS/OpenMP threads inside numpy/scipy,
# and a handful of Python threads each oversubscribing a shared, throttled
# vCPU turned a 39s sequential response into a request that never completed
# at all. That lesson doesn't expire just because the heaviest models are
# gone: no ThreadPoolExecutor in this service, full stop. Caching already
# removes the cost on every reload but the very first one, which is the case
# that actually matters for a personal dashboard.
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
    score: int  # how many of the gate's 6 conditions passed (2026-09, was the old checklist's point total)
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
    sector_rs_percentile: int | None = None,
) -> PositionRisk | None:
    """2026-09 (reconstruction, Fase 4) - one deliberate behavior change from
    the pre-gate version: `signal` can no longer become `EXIT_WARNING` from
    the buy-side read alone. The old weighted checklist's "evitar" verdict
    (score <= AVOID_THRESHOLD, several bearish factors at once) had enough
    accumulated weight to read as a real warning; the gate is a boolean
    pass/fail, and a *failing* entry gate on an already-open position isn't
    the same claim - ordinary noise (RSI pinned high outside a strong trend,
    a temporary ATR extension) fails it too, on names with nothing actually
    wrong. Conflating "not a fresh buy today" with "you should be worried
    about this holding" is exactly the entry/exit conflation `exit_engine.py`
    was built to avoid (see its own module docstring) - this function no
    longer makes that same mistake one level up. `EXIT_WARNING` now comes
    exclusively from the independent exit engine's own EXIT_NOW/REDUCE
    escalation below, never from `signals.gate.passes` being `False`."""
    signals = compute_core_signals(
        df["close"],
        df["high"],
        df["low"],
        df["volume"],
        df["open"],
        benchmark_close,
        rs_rating,
        vix_close=vix_close,
        ticker=ticker,
        sector_rs_percentile=sector_rs_percentile,
    )
    if signals is None:
        return None

    # How many of the gate's conditions passed (0-6) - the closest honest
    # equivalent to the old weighted checklist's `score` for the two places
    # (the persisted audit trail, `PositionRisk.score`) that still expect a
    # plain int. Nothing branches on its value; it's display/export-only
    # (decision_journal_export.py, the position card's score badge).
    gate_score = sum(1 for c in signals.gate.conditions if c.passed)

    near_support = (
        signals.nearest_support is not None and abs(signals.nearest_support.distance_pct) <= PROXIMITY_THRESHOLD
    )
    near_resistance = (
        signals.nearest_resistance is not None
        and abs(signals.nearest_resistance.distance_pct) <= PROXIMITY_THRESHOLD
    )
    # A *confirmed* death cross already factors into the gate (via trend/
    # stage). These are the earlier, still-projected cases: nothing else has
    # flagged this position yet, but a pair of moving averages is converging -
    # worth active attention before it's a lagging confirmation, not after.
    # Both the SMA50/SMA200 (medium/long-term) and SMA20/SMA50 (short-term -
    # relevant for a position actively managed on a shorter horizon, which can
    # turn well before the longer-term picture does) versions count here. See
    # technical_analysis.detect_imminent_cross.
    imminent_death_cross = signals.imminent_cross is not None and signals.imminent_cross.direction == "death"
    imminent_death_cross_short = (
        signals.imminent_cross_short_term is not None and signals.imminent_cross_short_term.direction == "death"
    )
    # The one price-action pattern checked here rather than left to the
    # gate: a bearish engulfing candle is a single-session event, not a
    # multi-day trend read like everything else the gate weighs - worth
    # surfacing immediately on the position that just printed one, not
    # waiting for it to show up in slower-moving indicators. See
    # technical_analysis.detect_engulfing_pattern.
    bearish_engulfing = signals.candlestick_pattern == "bearish_engulfing"

    watch_worthy = (
        near_support or near_resistance or imminent_death_cross or imminent_death_cross_short or bearish_engulfing
    )
    # See this function's own docstring: a failing gate alone no longer
    # produces EXIT_WARNING here - only the independent exit engine's
    # EXIT_NOW/REDUCE escalation below can.
    if signals.gate.passes:
        signal = ADD_CANDIDATE
    elif watch_worthy:
        signal = WATCH
    else:
        signal = HOLD

    # Deliberately *not* gated by which `signal` tier this landed in: an
    # add_candidate position with a strong short-term breakdown projected
    # (a passing gate absolutely can carry one - the gate looks at the whole
    # entry picture, this looks at one specific, narrower thing) still
    # deserves that context visible, not silently dropped because the
    # headline signal was upbeat. A real gap this closes: a held position can
    # pass the gate (ADD_CANDIDATE) while SMA20/SMA50 are actively converging
    # toward a short-term death cross - exactly the situation a short-term-
    # managed position needs a heads-up on, regardless of what the primary
    # badge says.
    reasons = [c.label for c in signals.gate.conditions if not c.passed]
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
        # Every gate condition passed and none of the independent technical
        # flags above fired - a genuinely clean read, not "nothing to say".
        reasons.append("Cumple las condiciones del gate de entrada, sin señales técnicas adicionales relevantes")

    # --- Independent exit engine (D2/D3 fix - see exit_engine.py's docstring) ---
    # Only runs when the caller supplies portfolio context: a bare technical
    # read (e.g. from a caller that only wants "Analizar activo"-style
    # signals, or an as-yet-unupdated test double) still gets the gate-based
    # `signal` above unchanged, just without this layered on
    # top - `exit_urgency` stays `None` in that case, never a guess.
    exit_urgency: str | None = None
    exit_reasons: list[str] = []
    trade_plan: TradePlan | None = None
    r_multiple: float | None = None
    multi_timeframe: mtf.MultiTimeframeRead | None = None
    scaled_exit: tm.ScaledExitPlan | None = None
    bars_held: int | None = None

    if portfolio_id is not None and trade_plan_repo is not None and transactions is not None:
        # Segunda auditoría, Bloque 2: `compute_core_signals` above now builds
        # this exact same read itself (`ticker_analysis_service.py`) off the
        # same `df` - reusing it here instead of calling
        # `analyze_multi_timeframe` a second time avoids redoing the weekly
        # resample + full indicator suite twice per position on every
        # portfolio refresh.
        multi_timeframe = signals.multi_timeframe
        plan = tps.ensure_trade_plan(trade_plan_repo, portfolio_id, ticker, transactions, df)
        # Region-aware cutoff (Segunda auditoría, Bloque 2) - a European
        # position's already-settled bar shouldn't wait for the US close.
        closed = ta.closed_bars(df, cutoff=closed_bar_cutoff_for_ticker(ticker))
        if plan is not None and not closed.empty:
            trade_plan = plan
            exit_price = float(closed["close"].iloc[-1])
            held_quantity = tps.current_held_quantity(transactions, ticker)
            # Parte 3.2/9 (reconstruction): both legs of the "fast pair" the
            # exit engine's hard triggers key off are now the real EMA21/55
            # multi_timeframe.py itself uses - not a third independent SMA
            # computation (see mtf.FAST_MA_PERIOD/SLOW_MA_PERIOD's own
            # docstring for why that used to be a "3 competing definitions"
            # problem).
            consecutive_below_ema55 = ta.consecutive_closes_below(
                closed["close"], ta.ema(closed["close"], mtf.SLOW_MA_PERIOD)
            )
            consecutive_below_ema21 = ta.consecutive_closes_below(
                closed["close"], ta.ema(closed["close"], mtf.FAST_MA_PERIOD)
            )
            # Parte 9 recalibration: exit_engine's own overextension trigger
            # measures against EMA21 now, not the shared (SMA50-based)
            # signals.atr_multiple every other display in the app still uses
            # - see exit_engine.py's own docstring on why these stay separate.
            atr_multiple_from_ema21 = ta.atr_multiple_from_ema(
                closed["close"], closed["high"], closed["low"], ema_window=mtf.FAST_MA_PERIOD
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
                consecutive_closes_below_daily_ema55=consecutive_below_ema55,
                consecutive_closes_below_daily_ema21=consecutive_below_ema21,
                nearest_support=nearest_support_closed,
                nearest_resistance=nearest_resistance_closed,
                obv_divergence=signals.obv_divergence,
                relative_volume=signals.relative_volume,
                rsi14=signals.rsi14,
                rsi_recent_max=rsi_recent_max,
                adx14=signals.adx14,
                adx_recent_max=adx_recent_max,
                atr_multiple_from_ema21=atr_multiple_from_ema21,
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
                initial_stop=plan.initial_stop,
                bars_held=bars_held,
            )

            # Trailing-stop update (trade_manager.py, Chandelier Exit) -
            # computed and persisted *after* evaluate_exit ran, deliberately:
            # the stop-breach check above must judge today's close against
            # the stop that was already in force coming into today, never
            # one just widened/raised using today's own bar - otherwise a
            # stop "breach" could be an artifact of raising the stop, not a
            # real adverse move.
            # Chandelier's regime multiplier used to come from a per-ticker
            # GARCH fit (volatility_model.py, retired 2026-09 - see
            # ticker_analysis_service.py's module docstring): what it
            # actually contributed here was one of four volatility buckets,
            # not a real forecast (nothing downstream consumed the forecast
            # itself once Monte Carlo was also retired). A percentile of this
            # ticker's own ATR/price over its trailing year inherits the same
            # volatility clustering (ATR is already an EWM average) without
            # fitting a model per ticker per request.
            atr_series_closed = ta.atr(closed["high"], closed["low"], closed["close"])
            vol_regime = ta.volatility_regime_from_atr_percentile(
                ta.atr_percentile(atr_series_closed / closed["close"])
            )
            # Bounded to bars on/after entry - the highest high the Chandelier
            # trail should ever consider is one this trade actually lived
            # through. `closed["high"]` unfiltered can reach years before
            # entry; a position opened after a pullback from an even higher
            # pre-entry high would otherwise trail against that pre-entry
            # high, not the trade's own price action.
            high_since_entry = closed["high"][closed["high"].index.date >= plan.entry_date]
            trailing = tm.compute_trailing_stop(
                high_since_entry, atr_series_closed,
                current_stop=plan.current_stop, r_multiple=r_multiple, vol_regime=vol_regime, price=exit_price,
            )
            if trailing.stop is not None and plan.id is not None:
                trade_plan_repo.update_trailing(
                    plan.id, trailing.stop, position_context.highest_close_since_entry,
                    current_stop_basis=trailing.basis,
                )
                trade_plan = replace(
                    plan,
                    current_stop=trailing.stop,
                    highest_close_since_entry=position_context.highest_close_since_entry,
                    # `trailing.basis` es `None` cuando el Chandelier no gobierna
                    # esta evaluación (Auditoria del Radar, bloque H2) - el mismo
                    # "None significa sin cambio" que `update_trailing` aplica en
                    # la fila persistida, para que este objeto en memoria no
                    # diverja de lo que acaba de escribirse.
                    current_stop_basis=(trailing.basis if trailing.basis is not None else plan.current_stop_basis),
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
                score=gate_score,
                price=signals.price,
                r_multiple=r_multiple,
                engine_version=GATE_VERSION,
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
        score=gate_score,
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
    sector_rs_percentile: int | None = None,
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
            sector_rs_percentile=sector_rs_percentile,
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
    sector_rs_by_ticker = (
        {s.ticker: s.sector_rs_percentile for s in universe_snapshot} if universe_snapshot else {}
    )

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
            sector_rs_percentile=sector_rs_by_ticker.get(ticker),
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
            sector_rs_by_ticker = (
                {s.ticker: s.sector_rs_percentile for s in universe_snapshot} if universe_snapshot else {}
            )
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
                    sector_rs_percentile=sector_rs_by_ticker.get(ticker),
                )
                if risk is not None:
                    fresh_by_ticker[ticker] = risk
                    self._cache[(portfolio_id, ticker)] = (now, risk)

        return [fresh_by_ticker[ticker] for ticker in tickers if ticker in fresh_by_ticker]
