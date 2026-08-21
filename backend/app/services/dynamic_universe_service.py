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
`premium_watchlist_service.py`/`watchlist_service.py` to call directly.
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


def apply_liquidity_filter(
    constituents: list[RawConstituent], market_data: MarketDataService
) -> list[RawConstituent]:
    """The brief's own hard floor: 20-day $ volume >= $20M, price >= $5,
    market cap >= $1B. A name failing to even fetch (delisted since the
    Wikipedia snapshot, a ticker Yahoo Finance doesn't recognize, a data gap)
    fails the filter the same way a genuinely illiquid one does - it isn't
    tradeable at this system's scale either way."""
    if not constituents:
        return []
    tickers = [c.ticker for c in constituents]
    end = date.today()
    start = end - timedelta(days=LIQUIDITY_HISTORY_DAYS)
    ohlcv_by_ticker = market_data.get_bulk_ohlcv(tickers, start, end)

    survivors: list[RawConstituent] = []
    for constituent in constituents:
        df = ohlcv_by_ticker.get(constituent.ticker)
        if df is None or df.empty or len(df) < 20:
            continue
        recent = df.iloc[-20:]
        last_price = float(recent["close"].iloc[-1])
        dollar_volume_20d = float((recent["close"] * recent["volume"]).mean())
        if last_price < MIN_PRICE or dollar_volume_20d < MIN_DOLLAR_VOLUME_20D:
            continue
        info = market_data.get_ticker_info(constituent.ticker)
        if info is None or info.market_cap is None or info.market_cap < MIN_MARKET_CAP:
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
