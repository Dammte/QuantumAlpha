"""Market-wide screener: scans a curated universe of liquid stocks and turns
their price history into the kind of at-a-glance signals a discretionary trader
checks every morning - gainers/losers, breakouts, sector/industry rotation,
breadth, relative strength leadership, and a Minervini-style trend checklist.

Recomputing technical indicators over ~170 tickers on every request would be
slow and hammer yfinance, so results are cached in-process for `CACHE_TTL` -
several hours, not minutes: this whole screener runs on daily bars, so nothing
here actually changes meaningfully more than a few times a day, and the
in-process cache is shared across every request in this single-process app (see
`get_market_screener_service` in `deps.py`). `get_universe_snapshot` - the one
call nearly every other method and market endpoint ultimately funnels through -
additionally backs itself with the durable, restart-proof cache in
`durable_cache.py` when a `db` session is supplied: without it, a Render
redeploy (which happens often on a project under active development) would
otherwise force the next visitor to eat the full ~170-ticker recompute cost
synchronously just because the in-process cache was empty again, not because
the data was actually stale.
"""

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.infrastructure.db.repositories.universe_membership_repository import (
    UniverseMembershipRepository,
)
from app.services import durable_cache
from app.services import dynamic_universe_service as dus
from app.services import technical_analysis as ta
from app.services.market_data_service import MarketDataService
from app.services.market_universe import (
    DEFAULT_REGION,
    all_industry_tickers,
    all_sector_tickers,
    benchmark_for_region,
    cap_tier_of,
    currency_of,
)
from app.services.multi_timeframe import FAST_MA_PERIOD

logger = logging.getLogger(__name__)

CACHE_TTL = timedelta(hours=3)
HISTORY_DAYS = 400  # enough calendar days to cover a 252-trading-day lookback + SMA200
MIN_BARS_REQUIRED = 60


def _snapshot_to_dict(snapshot: TickerSnapshot) -> dict[str, Any]:
    data = asdict(snapshot)
    data["trend"] = snapshot.trend.value
    data["stage"] = snapshot.stage.value if snapshot.stage else None
    return data


def _snapshot_from_dict(data: dict[str, Any]) -> TickerSnapshot:
    stage = ta.Stage(data["stage"]) if data["stage"] else None
    # imminent_cross_short_term round-trips through JSON as a plain dict (or
    # None) via asdict()'s recursive nested-dataclass conversion in
    # _snapshot_to_dict - reconstructed explicitly here the same way
    # stage/trend already are, or any code that dereferences .bars_until on a
    # cache-reloaded snapshot would get a plain dict instead of an
    # ImminentCross and crash. .get(), not [...]: a snapshot cached before
    # this field existed simply won't have the key, and that must stay a
    # clean "no imminent cross" default, not a KeyError.
    imminent_raw = data.get("imminent_cross_short_term")
    imminent_cross_short_term = ta.ImminentCross(**imminent_raw) if imminent_raw else None
    return TickerSnapshot(
        **{
            **data,
            "trend": ta.TrendState(data["trend"]),
            "stage": stage,
            "imminent_cross_short_term": imminent_cross_short_term,
        }
    )


@dataclass
class ScreenerFilters:
    sector: str | None = None
    industry: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_change_1d: float | None = None
    max_change_1d: float | None = None
    min_rsi: float | None = None
    max_rsi: float | None = None
    min_relative_volume: float | None = None
    min_rs_rating: float | None = None
    above_sma50: bool | None = None
    above_sma200: bool | None = None
    trend: str | None = None
    stage: str | None = None
    cap_tier: str | None = None
    minervini_pass: bool | None = None
    sort_by: str = "change_1d"
    sort_dir: str = "desc"


@dataclass
class _RawTicker:
    """Every indicator that doesn't need cross-sectional ranking, plus the raw
    inputs (`rs_raw`, `sma200_trending_up`, 52w hi/lo) the second pass needs to
    finish the RS Rating percentile and the Minervini checklist."""

    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    currency: str
    price: float
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_6m: float | None
    change_1y: float | None
    volume: float
    relative_volume: float | None
    rsi14: float | None
    sma20: float | None
    sma50: float | None
    sma150: float | None
    sma200: float | None
    dist_52w_high: float | None
    dist_52w_low: float | None
    atr_multiple: float | None
    adx14: float | None
    plus_di: float | None
    minus_di: float | None
    mansfield_rs: float | None
    trend: ta.TrendState
    stage: ta.Stage | None
    ma_cross: str | None
    rs_raw: float | None
    sma200_trending_up: bool | None
    price_52w_low_abs: float | None
    price_52w_high_abs: float | None
    atr_ratio_50d: float | None
    atr_multiple_sma21: float | None
    range_position_20d: float | None
    mansfield_rs_4w: float | None
    relative_volume_trend: float | None
    ma_cross_short: str | None
    imminent_cross_short_term: ta.ImminentCross | None


def _build_raw(
    ticker: str, sector: str, industry: str | None, df: pd.DataFrame, benchmark_close: pd.Series | None
) -> _RawTicker | None:
    if len(df) < MIN_BARS_REQUIRED:
        return None

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    price = float(close.iloc[-1])

    sma20_series = ta.sma(close, 20)
    sma50_series = ta.sma(close, 50)
    sma150_series = ta.sma(close, 150)
    sma200_series = ta.sma(close, 200)
    sma20 = sma20_series.iloc[-1]
    sma50 = sma50_series.iloc[-1]
    sma150 = sma150_series.iloc[-1]
    sma200 = sma200_series.iloc[-1]
    sma20 = None if pd.isna(sma20) else float(sma20)
    sma50 = None if pd.isna(sma50) else float(sma50)
    sma150_val = None if pd.isna(sma150) else float(sma150)
    sma200_val = None if pd.isna(sma200) else float(sma200)

    latest_rsi = ta.rsi(close).iloc[-1]

    ma_cross = None
    stage = None
    if len(close) >= 200:
        ma_cross = ta.detect_recent_cross(sma50_series, sma200_series, lookback=5)
        stage = ta.classify_stage(price, sma150_series)

    # "Tendencia" screener, corto/mediano plazo (ago 2026): FAST_MA_PERIOD/SMA50
    # is a weeks-scale pair, not the months-scale SMA50/SMA200 above - same
    # already-established short-term pair multi_timeframe.py/exit_engine.py
    # use, reused here rather than inventing a different one. Guarded on 50
    # bars (not 200): both legs are already valid well before SMA200 is.
    ma_cross_short = None
    imminent_cross_short_term = None
    if len(close) >= 50:
        sma_fast_series = ta.sma(close, FAST_MA_PERIOD)
        ma_cross_short = ta.detect_recent_cross(sma_fast_series, sma50_series, lookback=5)
        imminent_cross_short_term = ta.detect_imminent_cross(sma_fast_series, sma50_series)

    adx_series = ta.adx(high, low, close)
    plus_di_series, minus_di_series = ta.dmi(high, low, close)
    latest_adx = adx_series.iloc[-1] if not adx_series.empty else np.nan
    latest_plus_di = plus_di_series.iloc[-1] if not plus_di_series.empty else np.nan
    latest_minus_di = minus_di_series.iloc[-1] if not minus_di_series.empty else np.nan

    mansfield = None
    mansfield_4w = None
    if benchmark_close is not None:
        mansfield_series = ta.mansfield_rs(close, benchmark_close)
        if not mansfield_series.empty and not pd.isna(mansfield_series.iloc[-1]):
            mansfield = float(mansfield_series.iloc[-1])
        # 4-week window (~20 trading sessions) - a much shorter-horizon relative-
        # strength read than the 200-day one above, for the short-term setup score.
        mansfield_4w_series = ta.mansfield_rs(close, benchmark_close, window=20)
        if not mansfield_4w_series.empty and not pd.isna(mansfield_4w_series.iloc[-1]):
            mansfield_4w = float(mansfield_4w_series.iloc[-1])

    atr14_series = ta.atr(high, low, close)
    atr_ratio_50d = None
    if len(atr14_series) >= 50:
        atr_baseline = atr14_series.rolling(50, min_periods=50).mean().iloc[-1]
        latest_atr14 = atr14_series.iloc[-1]
        if pd.notna(atr_baseline) and atr_baseline != 0 and pd.notna(latest_atr14):
            atr_ratio_50d = float(latest_atr14 / atr_baseline)

    range_position_20d_series = ta.rolling_position_in_range(close, 20)
    range_position_20d = None
    if not range_position_20d_series.empty and pd.notna(range_position_20d_series.iloc[-1]):
        range_position_20d = float(range_position_20d_series.iloc[-1])

    # Today's relative_volume vs. its own value 5 sessions ago - the "trend"
    # half of "relative_volume + its trend" (Segunda auditoría, Bloque 3): a
    # single snapshot value can't distinguish "volume just spiked today" from
    # "interest has been building for a week", which matters for how much to
    # trust a breakout.
    relative_volume_trend = None
    current_rel_volume = ta.relative_volume(volume)
    past_rel_volume = ta.relative_volume(volume.iloc[:-5]) if len(volume) > 5 else None
    if current_rel_volume is not None and past_rel_volume is not None:
        relative_volume_trend = current_rel_volume - past_rel_volume

    return _RawTicker(
        ticker=ticker,
        sector=sector,
        industry=industry,
        cap_tier=cap_tier_of(ticker),
        currency=currency_of(ticker),
        price=price,
        change_1d=ta.pct_change_over(close, 1),
        change_1w=ta.pct_change_over(close, 5),
        change_1m=ta.pct_change_over(close, 21),
        change_3m=ta.pct_change_over(close, 63),
        change_6m=ta.pct_change_over(close, 126),
        change_1y=ta.pct_change_over(close, 252),
        volume=float(volume.iloc[-1]),
        relative_volume=ta.relative_volume(volume),
        rsi14=None if pd.isna(latest_rsi) else float(latest_rsi),
        sma20=sma20,
        sma50=sma50,
        sma150=sma150_val,
        sma200=sma200_val,
        dist_52w_high=ta.distance_to_rolling_extreme(close, 252, "high"),
        dist_52w_low=ta.distance_to_rolling_extreme(close, 252, "low"),
        atr_multiple=ta.atr_multiple_from_sma(close, high, low),
        adx14=None if pd.isna(latest_adx) else float(latest_adx),
        plus_di=None if pd.isna(latest_plus_di) else float(latest_plus_di),
        minus_di=None if pd.isna(latest_minus_di) else float(latest_minus_di),
        mansfield_rs=mansfield,
        trend=ta.classify_trend(price, sma20, sma50, sma200_val),
        stage=stage,
        ma_cross=ma_cross,
        rs_raw=ta.rs_raw_score(close),
        sma200_trending_up=ta.sma_slope_positive(sma200_series),
        price_52w_low_abs=ta.rolling_extreme_price(close, 252, "low"),
        price_52w_high_abs=ta.rolling_extreme_price(close, 252, "high"),
        atr_ratio_50d=atr_ratio_50d,
        atr_multiple_sma21=ta.atr_multiple_from_sma(close, high, low, sma_window=21),
        range_position_20d=range_position_20d,
        mansfield_rs_4w=mansfield_4w,
        relative_volume_trend=relative_volume_trend,
        ma_cross_short=ma_cross_short,
        imminent_cross_short_term=imminent_cross_short_term,
    )


def _percentile_rank(values: list[float]) -> list[int]:
    """1-99 percentile rank, IBD RS-Rating style (rank 1 = weakest)."""
    order = np.argsort(np.argsort(values))
    n = len(values)
    return [max(1, int(np.ceil(99 * (rank + 1) / n))) for rank in order]


def _finalize(raw: _RawTicker, rs_rating: int | None) -> TickerSnapshot:
    criteria = ta.minervini_checklist(
        price=raw.price,
        sma50=raw.sma50,
        sma150=raw.sma150,
        sma200=raw.sma200,
        sma200_trending_up=raw.sma200_trending_up,
        price_52w_low=raw.price_52w_low_abs,
        price_52w_high=raw.price_52w_high_abs,
        rs_rating=rs_rating,
    )
    return TickerSnapshot(
        ticker=raw.ticker,
        sector=raw.sector,
        industry=raw.industry,
        cap_tier=raw.cap_tier,
        currency=raw.currency,
        price=raw.price,
        change_1d=raw.change_1d,
        change_1w=raw.change_1w,
        change_1m=raw.change_1m,
        change_3m=raw.change_3m,
        change_6m=raw.change_6m,
        change_1y=raw.change_1y,
        volume=raw.volume,
        relative_volume=raw.relative_volume,
        rsi14=raw.rsi14,
        sma20=raw.sma20,
        sma50=raw.sma50,
        sma150=raw.sma150,
        sma200=raw.sma200,
        dist_52w_high=raw.dist_52w_high,
        dist_52w_low=raw.dist_52w_low,
        atr_multiple=raw.atr_multiple,
        adx14=raw.adx14,
        plus_di=raw.plus_di,
        minus_di=raw.minus_di,
        mansfield_rs=raw.mansfield_rs,
        trend=raw.trend,
        stage=raw.stage,
        ma_cross=raw.ma_cross,
        minervini_score=sum(criteria.values()),
        minervini_pass=all(criteria.values()),
        rs_rating=rs_rating,
        atr_ratio_50d=raw.atr_ratio_50d,
        atr_multiple_sma21=raw.atr_multiple_sma21,
        range_position_20d=raw.range_position_20d,
        mansfield_rs_4w=raw.mansfield_rs_4w,
        relative_volume_trend=raw.relative_volume_trend,
        ma_cross_short=raw.ma_cross_short,
        imminent_cross_short_term=raw.imminent_cross_short_term,
    )


class MarketScreenerService:
    def __init__(self, market_data: MarketDataService) -> None:
        self.market_data = market_data
        self._snapshot_cache: dict[str, tuple[datetime, list[TickerSnapshot]]] = {}
        self._ohlcv_cache: dict[str, tuple[datetime, dict[str, pd.DataFrame]]] = {}

    def _date_range(self) -> tuple[date, date]:
        end = date.today()
        start = end - timedelta(days=HISTORY_DAYS)
        return start, end

    def get_snapshot_computed_at(self, region: str = DEFAULT_REGION) -> datetime | None:
        """When the in-process universe snapshot for `region` was last actually
        computed - None if `get_universe_snapshot` hasn't been called yet this
        process. Lets a caller (e.g. the screener endpoint, which is just a
        filter over this snapshot) report "actualizado hace X" without
        needing its own separate durable-cache entry."""
        cached = self._snapshot_cache.get(region)
        return cached[0] if cached is not None else None

    def get_cached_ohlcv(self, region: str = DEFAULT_REGION) -> dict[str, pd.DataFrame]:
        """Whatever `get_universe_snapshot`'s own OHLCV fetch last cached for
        this region - `{}` until a real (non-durable-cache) recompute has
        happened at least once in this process. `get_proximity_matches`
        already relies on this same cache internally;
        `relationship_map_service.py` (Tercera auditoría, Bloque G) uses this
        public getter to compute cross-sectional correlation/lead-lag
        without a second download."""
        cached = self._ohlcv_cache.get(region)
        return cached[1] if cached is not None else {}

    def _ensure_ohlcv_cache_warm(self, region: str, snapshots: list[TickerSnapshot]) -> None:
        """Tercera auditoría, Bloque G, bug real encontrado en producción: un
        snapshot servido por cualquiera de los dos atajos de caché de abajo
        (TTL en proceso o durable) nunca pasaba por la descarga de OHLCV, así
        que `_ohlcv_cache` se quedaba vacío en cualquier worker que no
        hubiera hecho el recálculo completo todavía - `get_cached_ohlcv()`
        devolvía `{}` y `relationship_map_service.compute_statistical_relations`
        veía un universo vacío en silencio (confirmado contra el backend real:
        AAPL con `statistical: []` nada más desplegar). `get_proximity_matches`
        ya documentaba este mismo hueco como "un hueco silencioso, aceptable" -
        aceptable para un screener que igual muestra `[]` con pocas
        consecuencias, no para una capa entera de una función nueva que se
        anuncia como "estadística, la más fiable". La descarga en sí es una
        sola llamada por lotes (Bloque F-1: cuesta lo mismo para 170 o 1.000
        tickers) - esto nunca repite el cálculo de indicadores, solo rellena
        `_ohlcv_cache` cuando de verdad falta o caducó."""
        cached = self._ohlcv_cache.get(region)
        if cached is not None and datetime.now(UTC) - cached[0] < CACHE_TTL:
            return
        tickers = [s.ticker for s in snapshots]
        if not tickers:
            return
        benchmark_ticker = benchmark_for_region(region)
        start, end = self._date_range()
        ohlcv_by_ticker = self.market_data.get_bulk_ohlcv([*tickers, benchmark_ticker], start, end)
        self._ohlcv_cache[region] = (datetime.now(UTC), ohlcv_by_ticker)

    def get_universe_snapshot(
        self, region: str = DEFAULT_REGION, force_refresh: bool = False, db: Session | None = None
    ) -> list[TickerSnapshot]:
        cached = self._snapshot_cache.get(region)
        if not force_refresh and cached is not None:
            cached_at, snapshots = cached
            if datetime.now(UTC) - cached_at < CACHE_TTL:
                self._ensure_ohlcv_cache_warm(region, snapshots)
                return snapshots

        # In-process cache missed (cold start, just after a deploy, or genuinely
        # expired) - before paying for a live ~170-ticker recompute, check the
        # durable cache: it survives restarts, the in-process one doesn't. Also
        # populates the in-process cache so the *next* call this process makes
        # doesn't even need this DB round trip.
        if db is not None and not force_refresh:

            def _reconstruct_snapshots(payload):
                return [_snapshot_from_dict(d) for d in payload]

            snapshot_key = f"universe_snapshot:{region}"
            snapshots = durable_cache.load_fresh_as(db, snapshot_key, CACHE_TTL, _reconstruct_snapshots)
            if snapshots is not None:
                self._snapshot_cache[region] = (datetime.now(UTC), snapshots)
                self._ensure_ohlcv_cache_warm(region, snapshots)
                return snapshots

        # Tercera auditoría, Bloque F-1: the dynamic, point-in-time universe
        # (dynamic_universe_service.py, Segunda auditoría Bloque 3) had
        # exactly one caller in the whole repo (the offline ablation script)
        # despite its own docstring already claiming this was "cheap enough
        # for premium_watchlist_service.py/watchlist_service.py to call
        # directly" - true now, false before this. Connected in the two
        # stages that resolve the cost objection that blocked it: read the
        # up-to-~1000-ticker snapshot (a plain DB read) if one is on file for
        # this region, then cheaply screen it down to
        # CHEAP_SCREEN_KEEP_TOP_N by price/volume alone (no per-ticker
        # network call) *before* running full indicator computation - the
        # cost that actually worried anyone was computing SMA/RSI/ADX/etc. on
        # ~1000 tickers, never downloading them (one already-batched call
        # either way). Falls back to the curated dict, exactly as before,
        # when no snapshot exists yet for this region.
        ticker_sectors = all_sector_tickers(region)
        if db is not None:
            dynamic = dus.read_dynamic_universe(UniverseMembershipRepository(db), region)
            if dynamic is not None:
                ticker_sectors = dynamic
        ticker_industries = all_industry_tickers(region)
        benchmark_ticker = benchmark_for_region(region)
        start, end = self._date_range()

        fetch_list = [*ticker_sectors.keys(), benchmark_ticker]
        ohlcv_by_ticker = self.market_data.get_bulk_ohlcv(fetch_list, start, end)
        benchmark_close = ohlcv_by_ticker.get(benchmark_ticker)
        benchmark_close_series = benchmark_close["close"] if benchmark_close is not None else None

        if len(ticker_sectors) > dus.CHEAP_SCREEN_KEEP_TOP_N:
            survivors = dus.apply_cheap_price_volume_screen(ohlcv_by_ticker, list(ticker_sectors.keys()))
            logger.info(
                "Universe snapshot: cheap screen kept %d/%d dynamic-universe tickers (region=%s)",
                len(survivors), len(ticker_sectors), region,
            )
            ticker_sectors = {t: ticker_sectors[t] for t in survivors}

        # Tercera auditoría, Bloque F-9: until now, liquidity was only ever
        # checked in the offline monthly refresh
        # (dynamic_universe_service.apply_liquidity_filter) - which never
        # ran at all for the curated-fallback case (no snapshot on file
        # yet for this region), so a curated ticker that's gone illiquid
        # since it was hand-picked had nothing checking it here. Reuses the
        # OHLCV every ticker already has in memory - no new network call
        # except a per-*currency* FX rate, cached once across this whole
        # snapshot (not per ticker) exactly like apply_liquidity_filter's
        # own cache.
        fx_rate_cache: dict[str, float | None] = {}

        def usd_rate(currency: str) -> float | None:
            if currency == "USD":
                return 1.0
            if currency not in fx_rate_cache:
                fx_rate_cache[currency] = self.market_data.get_fx_rate(currency, "USD")
            return fx_rate_cache[currency]

        raw_tickers: list[_RawTicker] = []
        for ticker, sector in ticker_sectors.items():
            df = ohlcv_by_ticker.get(ticker)
            if df is None:
                continue
            currency = currency_of(ticker)
            usd_values = dus.usd_price_and_dollar_volume(df, currency, usd_rate(currency))
            if usd_values is None or usd_values[0] < dus.MIN_PRICE or usd_values[1] < dus.MIN_DOLLAR_VOLUME_20D:
                continue
            # One ticker's malformed data (a NaN run, a stock split artifact,
            # an unexpected data shape from the provider) must never take the
            # other ~170 down with it - this snapshot is the shared foundation
            # nearly every market endpoint depends on.
            try:
                raw = _build_raw(ticker, sector, ticker_industries.get(ticker), df, benchmark_close_series)
            except Exception:
                logger.exception("Universe snapshot: skipping %s after a compute failure", ticker)
                continue
            if raw is not None:
                raw_tickers.append(raw)

        rs_candidates = [r for r in raw_tickers if r.rs_raw is not None]
        rs_percentiles = _percentile_rank([r.rs_raw for r in rs_candidates])
        rs_ranks = dict(zip((r.ticker for r in rs_candidates), rs_percentiles, strict=True))

        snapshots = [_finalize(raw, rs_ranks.get(raw.ticker)) for raw in raw_tickers]

        self._snapshot_cache[region] = (datetime.now(UTC), snapshots)
        self._ohlcv_cache[region] = (datetime.now(UTC), ohlcv_by_ticker)
        if db is not None:
            durable_cache.save(db, f"universe_snapshot:{region}", [_snapshot_to_dict(s) for s in snapshots])
        return snapshots

    def get_proximity_matches(
        self,
        region: str = DEFAULT_REGION,
        threshold: float = 0.03,
        force_refresh: bool = False,
        db: Session | None = None,
    ) -> list[dict]:
        """Universe-wide "which stocks are sitting right on a support or resistance
        level right now" screener - reuses the OHLCV already fetched for the
        universe snapshot rather than re-downloading it.

        Raw OHLCV frames aren't themselves part of the durable cache (only the
        derived `TickerSnapshot`s are, see `durable_cache.py`), so a snapshot
        served from either cache used to leave `_ohlcv_cache` cold - fixed in
        `get_universe_snapshot`/`_ensure_ohlcv_cache_warm` (Tercera auditoría,
        Bloque G: found live in production, `relationship_map_service.py`'s
        Layer 1 was silently returning `[]` for every ticker on a freshly
        deployed worker). This still returns `[]` only in the genuine edge
        case of an empty universe for this region."""
        # ensures _ohlcv_cache is warm
        self.get_universe_snapshot(region=region, force_refresh=force_refresh, db=db)
        cached = self._ohlcv_cache.get(region)
        if cached is None:
            return []
        _, ohlcv_by_ticker = cached
        ticker_sectors = all_sector_tickers(region)

        matches = []
        for ticker, sector in ticker_sectors.items():
            df = ohlcv_by_ticker.get(ticker)
            if df is None or len(df) < MIN_BARS_REQUIRED:
                continue
            levels = ta.support_resistance_levels(df["high"], df["low"], df["close"])
            if not levels:
                continue
            nearest = min(levels, key=lambda lv: abs(lv.distance_pct))
            if abs(nearest.distance_pct) <= threshold:
                matches.append(
                    {
                        "ticker": ticker,
                        "sector": sector,
                        "currency": currency_of(ticker),
                        "price": float(df["close"].iloc[-1]),
                        "level": nearest,
                    }
                )
        return sorted(matches, key=lambda m: abs(m["level"].distance_pct))

    def get_support_resistance(self, ticker: str, start: date, end: date) -> tuple[float, list[ta.PriceLevel]]:
        ohlcv = self.market_data.get_bulk_ohlcv([ticker], start, end)
        df = ohlcv.get(ticker)
        if df is None or df.empty:
            raise ValueError(f"No price history available for {ticker}")
        levels = ta.support_resistance_levels(df["high"], df["low"], df["close"])
        return float(df["close"].iloc[-1]), levels


def apply_filters(snapshots: list[TickerSnapshot], filters: ScreenerFilters) -> list[TickerSnapshot]:
    results = snapshots

    if filters.sector:
        results = [s for s in results if s.sector == filters.sector]
    if filters.industry:
        results = [s for s in results if s.industry == filters.industry]
    if filters.min_price is not None:
        results = [s for s in results if s.price >= filters.min_price]
    if filters.max_price is not None:
        results = [s for s in results if s.price <= filters.max_price]
    if filters.min_change_1d is not None:
        results = [s for s in results if s.change_1d is not None and s.change_1d >= filters.min_change_1d]
    if filters.max_change_1d is not None:
        results = [s for s in results if s.change_1d is not None and s.change_1d <= filters.max_change_1d]
    if filters.min_rsi is not None:
        results = [s for s in results if s.rsi14 is not None and s.rsi14 >= filters.min_rsi]
    if filters.max_rsi is not None:
        results = [s for s in results if s.rsi14 is not None and s.rsi14 <= filters.max_rsi]
    if filters.min_relative_volume is not None:
        min_rel_vol = filters.min_relative_volume
        results = [s for s in results if s.relative_volume is not None and s.relative_volume >= min_rel_vol]
    if filters.min_rs_rating is not None:
        min_rs = filters.min_rs_rating
        results = [s for s in results if s.rs_rating is not None and s.rs_rating >= min_rs]
    if filters.above_sma50 is not None:
        results = [s for s in results if s.sma50 is not None and (s.price > s.sma50) == filters.above_sma50]
    if filters.above_sma200 is not None:
        results = [s for s in results if s.sma200 is not None and (s.price > s.sma200) == filters.above_sma200]
    if filters.trend:
        results = [s for s in results if s.trend.value == filters.trend]
    if filters.stage:
        results = [s for s in results if s.stage is not None and s.stage.value == filters.stage]
    if filters.cap_tier:
        results = [s for s in results if s.cap_tier == filters.cap_tier]
    if filters.minervini_pass is not None:
        results = [s for s in results if s.minervini_pass == filters.minervini_pass]

    sort_key = filters.sort_by
    reverse = filters.sort_dir == "desc"

    def _sort_key(s: TickerSnapshot) -> tuple[bool, float]:
        # Ascending key, direction encoded via negation instead of
        # `sorted(reverse=...)` - reverse=True (the "desc" default) flips a
        # tuple's leading bool too, which used to put every row with a None
        # sort field at the *top* of the default descending screener instead
        # of the bottom (Tercera auditoría, Bloque A-4, same bug as
        # watchlist_service._sort_key). Missing data sorts last regardless of
        # sort direction - never regardless of "desc" being the more
        # dangerous default.
        value = getattr(s, sort_key)
        if value is None:
            return (True, 0.0)
        return (False, -value if reverse else value)

    results = sorted(results, key=_sort_key)
    return results


def get_movers(snapshots: list[TickerSnapshot], top_n: int = 10) -> dict[str, list[TickerSnapshot]]:
    def top(items: list[TickerSnapshot], key, reverse: bool) -> list[TickerSnapshot]:
        valid = [s for s in items if key(s) is not None]
        return sorted(valid, key=key, reverse=reverse)[:top_n]

    return {
        "gainers": top(snapshots, lambda s: s.change_1d, True),
        "losers": top(snapshots, lambda s: s.change_1d, False),
        "near_52w_high": top(
            [s for s in snapshots if s.dist_52w_high is not None and s.dist_52w_high >= -0.03],
            lambda s: s.dist_52w_high,
            True,
        ),
        "near_52w_low": top(
            [s for s in snapshots if s.dist_52w_low is not None and s.dist_52w_low <= 0.05],
            lambda s: s.dist_52w_low,
            False,
        ),
        "high_volume": top(
            [s for s in snapshots if s.relative_volume is not None and s.relative_volume >= 1.5],
            lambda s: s.relative_volume,
            True,
        ),
        "oversold": top(
            [s for s in snapshots if s.rsi14 is not None and s.rsi14 <= 35],
            lambda s: s.rsi14,
            False,
        ),
        "overbought": top(
            [s for s in snapshots if s.rsi14 is not None and s.rsi14 >= 65],
            lambda s: s.rsi14,
            True,
        ),
        "golden_cross": top(
            [s for s in snapshots if s.ma_cross == "golden"],
            lambda s: s.change_1m,
            True,
        ),
        "death_cross": top(
            [s for s in snapshots if s.ma_cross == "death"],
            lambda s: s.change_1m,
            False,
        ),
        "rs_leaders": top(snapshots, lambda s: s.rs_rating, True),
        "strong_trend": top(
            [s for s in snapshots if s.adx14 is not None and s.adx14 >= 25],
            lambda s: s.adx14,
            True,
        ),
    }


@dataclass(frozen=True, slots=True)
class TrendBreadth:
    total: int
    pct_above_sma50: float
    pct_above_sma200: float
    count_uptrend: int
    count_downtrend: int
    count_sideways: int
    golden_crosses: int
    death_crosses: int
    count_overbought: int
    count_oversold: int
    count_stage2: int
    count_minervini_pass: int
    count_imminent_golden: int = 0
    count_imminent_death: int = 0


def get_trend_breadth(snapshots: list[TickerSnapshot]) -> TrendBreadth:
    total = len(snapshots)
    if total == 0:
        return TrendBreadth(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    above_sma50 = sum(1 for s in snapshots if s.sma50 is not None and s.price > s.sma50)
    above_sma200 = sum(1 for s in snapshots if s.sma200 is not None and s.price > s.sma200)

    return TrendBreadth(
        total=total,
        pct_above_sma50=above_sma50 / total,
        pct_above_sma200=above_sma200 / total,
        count_uptrend=sum(1 for s in snapshots if s.trend == ta.TrendState.UPTREND),
        count_downtrend=sum(1 for s in snapshots if s.trend == ta.TrendState.DOWNTREND),
        count_sideways=sum(1 for s in snapshots if s.trend == ta.TrendState.SIDEWAYS),
        # Corto/mediano plazo (ago 2026): este panel usa ma_cross_short
        # (SMA{FAST_MA_PERIOD}/SMA50, semanas) en vez de ma_cross
        # (SMA50/SMA200, meses) - ver el campo en ticker_snapshot.py. `get_movers`
        # (un panel distinto) sigue usando el par largo a propósito, sin cambios.
        golden_crosses=sum(1 for s in snapshots if s.ma_cross_short == "golden"),
        death_crosses=sum(1 for s in snapshots if s.ma_cross_short == "death"),
        count_overbought=sum(1 for s in snapshots if s.rsi14 is not None and s.rsi14 >= 70),
        count_oversold=sum(1 for s in snapshots if s.rsi14 is not None and s.rsi14 <= 30),
        count_stage2=sum(1 for s in snapshots if s.stage == ta.Stage.STAGE_2),
        count_minervini_pass=sum(1 for s in snapshots if s.minervini_pass),
        count_imminent_golden=sum(
            1 for s in snapshots if s.imminent_cross_short_term is not None
            and s.imminent_cross_short_term.direction == "golden"
        ),
        count_imminent_death=sum(
            1 for s in snapshots if s.imminent_cross_short_term is not None
            and s.imminent_cross_short_term.direction == "death"
        ),
    )


def get_trend_detail(snapshots: list[TickerSnapshot], top_n: int = 40) -> dict[str, list[TickerSnapshot]]:
    """The ticker-level detail behind `get_trend_breadth`'s counts - so instead of
    just "2 golden crosses" you get to see it's AAPL and MSFT.

    `deprioritize` (Segunda auditoría, ago 2026 - "activos recomendados más
    precisos"): an optional predicate that pushes a matching name to the
    bottom of its own group instead of excluding it outright - used on the
    bullish/"recommended" groups to stop an already-parabolic name
    (`atr_multiple > 4`, the same overextension threshold
    `recommendation_engine.py`'s `atr_parabolic` factor already scores) from
    crowding out a healthier trend candidate at the top of the list, without
    hiding it (it's still shown, just not first)."""

    def top(items: list[TickerSnapshot], key, reverse: bool, deprioritize=None) -> list[TickerSnapshot]:
        valid = [s for s in items if key(s) is not None]
        if deprioritize is None:
            return sorted(valid, key=key, reverse=reverse)[:top_n]
        preferred = sorted((s for s in valid if not deprioritize(s)), key=key, reverse=reverse)
        demoted = sorted((s for s in valid if deprioritize(s)), key=key, reverse=reverse)
        return (preferred + demoted)[:top_n]

    def is_overextended(s: TickerSnapshot) -> bool:
        return s.atr_multiple is not None and s.atr_multiple > 4

    uptrend = [s for s in snapshots if s.trend == ta.TrendState.UPTREND]
    downtrend = [s for s in snapshots if s.trend == ta.TrendState.DOWNTREND]
    overbought = [s for s in snapshots if s.rsi14 is not None and s.rsi14 >= 70]
    oversold = [s for s in snapshots if s.rsi14 is not None and s.rsi14 <= 30]
    imminent_cross = [s for s in snapshots if s.imminent_cross_short_term is not None]

    return {
        "uptrend": top(uptrend, lambda s: s.rs_rating, True, deprioritize=is_overextended),
        "downtrend": top(downtrend, lambda s: s.rs_rating, False),
        "golden_cross": top(
            [s for s in snapshots if s.ma_cross_short == "golden"], lambda s: s.change_1m, True,
            deprioritize=is_overextended,
        ),
        "death_cross": top([s for s in snapshots if s.ma_cross_short == "death"], lambda s: s.change_1m, False),
        "overbought": top(overbought, lambda s: s.rsi14, True),
        "oversold": top(oversold, lambda s: s.rsi14, False),
        "stage2": top(
            [s for s in snapshots if s.stage == ta.Stage.STAGE_2], lambda s: s.rs_rating, True,
            deprioritize=is_overextended,
        ),
        "stage4": top([s for s in snapshots if s.stage == ta.Stage.STAGE_4], lambda s: s.rs_rating, False),
        "minervini_pass": top(
            [s for s in snapshots if s.minervini_pass], lambda s: s.rs_rating, True, deprioritize=is_overextended
        ),
        "strong_trend": top(
            [s for s in snapshots if s.adx14 is not None and s.adx14 >= 25], lambda s: s.adx14, True,
            deprioritize=is_overextended,
        ),
        # Corto/mediano plazo (ago 2026): proyección de detect_imminent_cross
        # sobre el par corto (SMA{FAST_MA_PERIOD}/SMA50) - antes de que el
        # cruce ocurra, no después. Ordenado por sesiones estimadas: el más
        # próximo primero.
        "imminent_cross": top(imminent_cross, lambda s: s.imminent_cross_short_term.bars_until, False),
    }
