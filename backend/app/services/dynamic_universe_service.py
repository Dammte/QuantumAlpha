"""Dynamic, monthly-refreshed investable universe (D14 - Segunda auditoría,
Bloque 3): `market_universe.py`'s curated dict has no survivorship-bias
protection - a ticker delisted or dropped from an index simply isn't in
today's hardcoded list, so nothing built on it (the premium watchlist, the
ablation study) can ever see that failure. This fetches real index
constituents (S&P 500, S&P 400, STOXX Europe 600) from their public
Wikipedia pages, applies a hard liquidity filter, and persists one dated
snapshot via `UniverseMembershipRepositoryPort` - see that port's docstring
for why history is appended, never overwritten, so real point-in-time data
accumulates from here forward.

Honest limitation, not silently glossed over: this cannot retroactively
reconstruct who was in each index years ago - only a paid historical-
constituents feed could. Every snapshot this saves is "the universe as of
the day this ran", so a factor study using an early snapshot to evaluate
much older price history still carries some survivorship bias for those
older dates - it just no longer *compounds* going forward, and the bias
shrinks to zero for any sample date on or after this shipped.

This is a monthly BATCH job (`scripts/refresh_universe_membership.py`),
never triggered from a live request path: fetching + liquidity-filtering
~1000+ combined constituents is much heavier than anything else in this
codebase's hot paths, and "no llamadas de red por ticker en los caminos
calientes" (CLAUDE.md) applies here too even though the cost is compute as
much as network - a user should never be the one paying for this by chance.
The read side (`read_dynamic_universe`) is a plain DB read, cheap enough for
`market_screener_service.get_universe_snapshot` to call directly - and
(Tercera auditoría, Bloque F-1) it now actually does, which
`premium_watchlist_service.py`/`watchlist_service.py` inherit for free since
both build on that same shared snapshot. Connecting it in one step wasn't
actually cheap: full indicator computation on the ~1000 constituents this
table can hold is the real cost that blocked the decision (Segunda
auditoría, Bloque 3's own note on this), so `get_universe_snapshot` cheaply
screens the snapshot down to `CHEAP_SCREEN_KEEP_TOP_N` by price/volume alone
(`apply_cheap_price_volume_screen`, no per-ticker network call) *before*
computing indicators on any of it.
"""

import io
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

from app.domain.interfaces.universe_membership_repository import UniverseMembershipRepositoryPort
from app.domain.models.universe_membership import UniverseMember
from app.services import market_universe as mu
from app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_DAYS = 30

# Hard liquidity floor (the brief's own numbers) - a name failing any one of
# these is simply not tradeable at the scale/frequency this system assumes,
# regardless of how good its technicals look.
MIN_DOLLAR_VOLUME_20D = 20_000_000.0
MIN_PRICE = 5.0
MIN_MARKET_CAP = 1_000_000_000.0
LIQUIDITY_HISTORY_DAYS = 45  # comfortably covers 20 trading days incl. weekends/holidays

# Wikipedia asks for a descriptive User-Agent identifying the client - not a
# workaround, its own documented API etiquette (meta.wikimedia.org/wiki/User-Agent_policy).
WIKIPEDIA_USER_AGENT = "QuantumAlphaPortfolioTool/1.0 (personal-use research tool; no scraping at scale)"

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
STOXX600_WIKI_URL = "https://en.wikipedia.org/wiki/STOXX_Europe_600"

# Yahoo Finance suffix per STOXX 600 "Country" column value - same suffixes
# `market_universe.EUROPEAN_EXCHANGE_SUFFIXES` already recognizes on the read
# side, just keyed by country name here since that's what the source table
# carries instead of an exchange code. A country with no entry here (a small
# handful of names on exchanges this system doesn't otherwise cover) is
# simply skipped - logged, not guessed.
YAHOO_SUFFIX_BY_COUNTRY: dict[str, str] = {
    "France": ".PA",
    "Germany": ".DE",
    "Netherlands": ".AS",
    "Spain": ".MC",
    "Italy": ".MI",
    "Switzerland": ".SW",
    "United Kingdom": ".L",
    "Belgium": ".BR",
    "Austria": ".VI",
    "Portugal": ".LS",
    "Sweden": ".ST",
    "Denmark": ".CO",
    "Finland": ".HE",
    "Norway": ".OL",
}


@dataclass(frozen=True, slots=True)
class RawConstituent:
    ticker: str
    sector: str | None


def _fetch_html(url: str) -> str:
    """Isolated so tests can monkeypatch this one function with a fixed HTML
    fixture instead of hitting the real network - everything downstream
    (parsing, normalizing) is then a pure function of that HTML string."""
    response = requests.get(url, headers={"User-Agent": WIKIPEDIA_USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def parse_us_constituents(html: str) -> list[RawConstituent]:
    """S&P 500 and S&P 400 Wikipedia pages share the same first-table shape:
    a `Symbol` column (already a bare Yahoo-compatible ticker for a US
    listing) and a `GICS Sector` column."""
    table = pd.read_html(io.StringIO(html))[0]
    out = []
    for _, row in table.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue
        # Wikipedia uses a dot for a share-class suffix (e.g. "BRK.B"); Yahoo
        # Finance uses a dash for the same thing ("BRK-B").
        ticker = symbol.replace(".", "-").upper()
        sector = row.get("GICS Sector")
        out.append(RawConstituent(ticker=ticker, sector=str(sector) if pd.notna(sector) else None))
    return out


def parse_stoxx600_constituents(html: str) -> list[RawConstituent]:
    """The STOXX Europe 600 page's constituents table carries a bare ticker
    (no exchange suffix) plus a `Country` column - resolved to a Yahoo
    Finance suffix via `YAHOO_SUFFIX_BY_COUNTRY`. A country this system
    doesn't have a mapping for is skipped (logged), never guessed."""
    tables = pd.read_html(io.StringIO(html))
    # The constituents table is identified by shape, not a fixed index - the
    # page carries over a dozen small unrelated tables (index history,
    # sector weights, a navbox) whose position could shift between edits.
    candidates = [t for t in tables if {"Ticker", "Country"} <= set(t.columns)]
    if not candidates:
        return []
    table = max(candidates, key=len)

    out = []
    skipped_countries: set[str] = set()
    for _, row in table.iterrows():
        raw_ticker = str(row.get("Ticker", "")).strip()
        country = str(row.get("Country", "")).strip()
        if not raw_ticker or not country:
            continue
        suffix = YAHOO_SUFFIX_BY_COUNTRY.get(country)
        if suffix is None:
            skipped_countries.add(country)
            continue
        sector = row.get("ICB Sector")
        sector_str = str(sector) if pd.notna(sector) else None
        out.append(RawConstituent(ticker=f"{raw_ticker}{suffix}", sector=sector_str))
    if skipped_countries:
        logger.info("Dynamic universe: skipped STOXX 600 countries with no suffix mapping: %s", skipped_countries)
    return out


def _dedupe(constituents: list[RawConstituent]) -> list[RawConstituent]:
    """Last write wins (same shape either way) - a ticker appearing twice in
    a source table (a real S&P 500/400 overlap, or a genuine duplicate row
    on the STOXX 600 page - confirmed live: "SHEL.L" appears twice there)
    would otherwise violate `universe_memberships`'s own
    (region, ticker, as_of_date) uniqueness constraint at save time."""
    return list({c.ticker: c for c in constituents}.values())


def fetch_live_constituents(region: str) -> list[RawConstituent] | None:
    """Best-effort live fetch for one region - `None` (not an empty list, so
    the caller can tell "the fetch itself failed" apart from "the fetch
    worked but found nothing") on any network/parsing failure. Combines
    S&P 500 + S&P 400 for "us"; STOXX Europe 600 for "europe"."""
    try:
        if region == "us":
            sp500 = parse_us_constituents(_fetch_html(SP500_WIKI_URL))
            sp400 = parse_us_constituents(_fetch_html(SP400_WIKI_URL))
            return _dedupe([*sp500, *sp400])
        if region == "europe":
            return _dedupe(parse_stoxx600_constituents(_fetch_html(STOXX600_WIKI_URL)))
        return None
    except Exception:
        logger.exception("Dynamic universe: live constituent fetch failed for region=%s", region)
        return None


MAX_STALE_DAYS = 10  # comfortably covers a weekend + a holiday - beyond this, "still trading" is doubtful

# Tercera auditoría, Bloque F-1: how many survivors the request-time cheap
# screen below keeps out of the ~1000 dynamic-universe constituents, before
# get_universe_snapshot runs full indicator computation on any of them - the
# actual cost this was blocking on was computing SMA/RSI/ADX/etc. on ~1000
# tickers per cache-miss request, not downloading their OHLCV (one already-
# batched call either way).
CHEAP_SCREEN_KEEP_TOP_N = 400


def apply_cheap_price_volume_screen(
    ohlcv_by_ticker: dict[str, pd.DataFrame], tickers: list[str], keep_top_n: int = CHEAP_SCREEN_KEEP_TOP_N
) -> list[str]:
    """Request-time (hot-path-safe) pre-screen: keeps the `keep_top_n`
    tickers by 20-day $ volume among those clearing `MIN_PRICE`, using only
    the bulk OHLCV every caller already has in memory - no per-ticker
    network call (unlike `apply_liquidity_filter`'s own `get_ticker_info`
    market-cap check, which is fine for the monthly *batch* script but would
    reintroduce exactly the "network call per ticker in a hot path" CLAUDE.md
    forbids if run here). This is the first of the two stages that make
    connecting the ~1000-ticker dynamic universe to the live snapshot
    affordable: full indicator computation (`_build_raw`) then only runs on
    these survivors, not the full constituent list.

    Known, disclosed limitation: this ranks by *raw local-currency* dollar
    volume, not FX-normalized - unlike `apply_liquidity_filter` (Bloque A-6),
    which already FX-corrected the absolute liquidity bar every one of these
    tickers had to clear during the monthly refresh that put it in
    `universe_memberships` to begin with. So this never lets an illiquid
    name back in - it only affects *which* already-liquid 400 survive the
    truncation when Europe's mixed currencies are ranked against each other
    by a not-currency-normalized number. Full FX normalization here would
    need a `get_fx_rate` call per distinct currency in the request path;
    revisit if the truncation itself (not the liquidity bar) turns out to
    matter in practice."""
    candidates: list[tuple[str, float]] = []
    for ticker in tickers:
        df = ohlcv_by_ticker.get(ticker)
        if df is None or df.empty or len(df) < 20:
            continue
        recent = df.iloc[-20:]
        last_price = float(recent["close"].iloc[-1])
        if last_price < MIN_PRICE:
            continue
        dollar_volume_20d = float((recent["close"] * recent["volume"]).mean())
        if dollar_volume_20d < MIN_DOLLAR_VOLUME_20D:
            continue
        candidates.append((ticker, dollar_volume_20d))
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return [ticker for ticker, _ in candidates[:keep_top_n]]


def usd_price_and_dollar_volume(
    df: pd.DataFrame, currency: str, fx_rate: float | None
) -> tuple[float, float] | None:
    """Last close and 20-day mean $ volume, converted to USD via `fx_rate`
    (already resolved by the caller, typically cached per-currency across a
    whole batch - see `usd_rate` closures in `apply_liquidity_filter` and
    `market_screener_service.get_universe_snapshot`). `None` when there
    isn't `>= 20` bars of history, or `fx_rate` is `None` (couldn't be
    resolved - fail safe, never silently assume USD). Shared by
    `apply_liquidity_filter` (offline monthly batch, Bloque A-6) and the
    live-path liquidity gate in `market_screener_service._build_raw`
    (Tercera auditoría, Bloque F-9), so the two currency-conversion paths
    never drift apart."""
    if df is None or df.empty or len(df) < 20 or fx_rate is None:
        return None
    recent = df.iloc[-20:]
    last_price_usd = float(recent["close"].iloc[-1]) * fx_rate
    dollar_volume_20d_usd = float((recent["close"] * recent["volume"]).mean()) * fx_rate
    return last_price_usd, dollar_volume_20d_usd


def apply_liquidity_filter(
    constituents: list[RawConstituent], market_data: MarketDataService
) -> list[RawConstituent]:
    """The brief's own hard floor: 20-day $ volume >= $20M, price >= $5,
    market cap >= $1B. A name failing to even fetch (delisted since the
    Wikipedia snapshot, a ticker Yahoo Finance doesn't recognize, a data gap)
    fails the filter the same way a genuinely illiquid one does - it isn't
    tradeable at this system's scale either way.

    Tercera auditoría, Bloque A-6, two real bugs fixed here:
    - Price/volume were compared against these USD thresholds in whatever
      currency the ticker itself quotes in - GBp (London Stock Exchange
      pence) made the filter ~127x laxer (a stock priced in pence reads as a
      100x larger raw number, on top of GBP itself being roughly 1:1 with
      USD), SEK/NOK/DKK ~10x laxer, EUR/CHF a smaller but real ~10% gap. A
      Swedish name doing 20M SEK/day (~$1.9M) used to clear the "$20M" bar
      outright. `market_data.get_fx_rate` already normalizes GBp/GBX to a
      proper GBP-based rate (see yfinance_provider.py) - this just has to
      actually call it. Market cap is deliberately left unconverted here:
      unlike price/volume, whether yfinance's own `marketCap` field is
      already USD-normalized for a foreign listing isn't something this
      audit verified against the live API, and guessing wrong in either
      direction would silently corrupt the one number left alone - flagged,
      not fixed blind.
    - A ticker whose last available bar is stale (delisted or halted since
      the 45-day window's start) used to pass with rancid prices as long as
      it had `>= 20` bars *at some point* in that window - recency of the
      *latest* bar was never checked."""
    if not constituents:
        return []
    tickers = [c.ticker for c in constituents]
    end = date.today()
    start = end - timedelta(days=LIQUIDITY_HISTORY_DAYS)
    ohlcv_by_ticker = market_data.get_bulk_ohlcv(tickers, start, end)
    stale_cutoff = end - timedelta(days=MAX_STALE_DAYS)

    fx_rate_cache: dict[str, float | None] = {}

    def usd_rate(currency: str) -> float | None:
        if currency not in fx_rate_cache:
            fx_rate_cache[currency] = market_data.get_fx_rate(currency, "USD")
        return fx_rate_cache[currency]

    survivors: list[RawConstituent] = []
    for constituent in constituents:
        df = ohlcv_by_ticker.get(constituent.ticker)
        if df is None or df.empty or len(df) < 20:
            continue
        if df.index[-1].date() < stale_cutoff:
            continue
        info = market_data.get_ticker_info(constituent.ticker)
        if info is None or info.market_cap is None or info.market_cap < MIN_MARKET_CAP:
            continue

        fx_rate = usd_rate(info.currency or "USD")
        usd_values = usd_price_and_dollar_volume(df, info.currency or "USD", fx_rate)
        if usd_values is None:
            continue  # can't confirm this clears a USD bar - fail safe, never assume USD silently
        last_price_usd, dollar_volume_20d_usd = usd_values
        if last_price_usd < MIN_PRICE or dollar_volume_20d_usd < MIN_DOLLAR_VOLUME_20D:
            continue
        survivors.append(constituent)
    return survivors


def refresh_universe_membership(
    region: str,
    market_data: MarketDataService,
    repo: UniverseMembershipRepositoryPort,
    as_of_date: date | None = None,
) -> tuple[int, str]:
    """Orchestrates one region's monthly refresh: live fetch -> liquidity
    filter -> persisted snapshot. Falls back to the curated universe
    (`market_universe.all_sector_tickers`) - logged, not silent - when the
    live fetch fails outright; a live fetch that *succeeds* but the
    liquidity filter leaves thin is trusted as-is (that's the filter doing
    its job, not a failure). Returns (member_count, source) for the caller
    (the script) to report."""
    as_of_date = as_of_date or date.today()
    raw = fetch_live_constituents(region)
    if raw is not None:
        members = apply_liquidity_filter(raw, market_data)
        members = [UniverseMember(m.ticker, region, m.sector, as_of_date, "live") for m in members]
        if members:
            repo.save_snapshot(region, as_of_date, "live", members)
            return len(members), "live"
        logger.warning(
            "Dynamic universe: live fetch for region=%s returned zero survivors after the liquidity filter - "
            "falling back to the curated universe rather than persisting an empty snapshot", region
        )

    logger.warning("Dynamic universe: falling back to the curated universe for region=%s", region)
    curated = mu.all_sector_tickers(region)
    members = [
        UniverseMember(ticker, region, sector, as_of_date, "curated_fallback")
        for ticker, sector in curated.items()
    ]
    repo.save_snapshot(region, as_of_date, "curated_fallback", members)
    return len(members), "curated_fallback"


def is_refresh_due(repo: UniverseMembershipRepositoryPort, region: str, today: date | None = None) -> bool:
    today = today or date.today()
    latest = repo.latest_as_of_date(region)
    return latest is None or (today - latest).days >= REFRESH_INTERVAL_DAYS


def read_dynamic_universe(
    repo: UniverseMembershipRepositoryPort, region: str, as_of_date: date | None = None
) -> dict[str, str | None] | None:
    """Cheap DB read for the actual candidate-generation paths
    (`premium_watchlist_service.py`/`watchlist_service.py`) - `ticker ->
    sector`, same shape `market_universe.all_sector_tickers` returns, so
    either can be dropped in as the universe source. `None` (not `{}`) when
    nothing is on file yet for this region, so the caller can tell "not
    refreshed yet - use the curated universe" apart from "refreshed, but
    genuinely empty"."""
    members = repo.members_as_of(region, as_of_date)
    if not members:
        return None
    return {m.ticker: m.sector for m in members}
