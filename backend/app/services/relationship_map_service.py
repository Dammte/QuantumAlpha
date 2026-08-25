"""Tercera auditoría, Bloque G (función nueva, explícitamente pedida): "que se
muestren acciones relacionadas... un mapa completo del proceso para buscar
nuevas opciones."

No existe ninguna fuente gratuita y fiable de cadena de suministro
estructurada (FactSet Supply Chain y similares son de pago). Este módulo
construye el mapa en tres capas, de la más fiable/objetiva a la más
especulativa - etiquetadas visualmente distintas por el frontend, nunca
mezcladas como si fueran la misma clase de evidencia:

1. **Estadística** (la más útil para operar, y gratis): correlación de
   retornos a 60/250 sesiones, beta relativa, correlación con desfase
   (lead-lag, k∈[-5,+5] sesiones - qué activo se mueve primero), co-
   movimiento en días extremos (±2σ), y divergencia actual frente a la
   correlación histórica. Todo calculado sobre el OHLCV que
   `MarketScreenerService.get_universe_snapshot` ya descargó para el
   universo entero - cero llamadas de red nuevas por par.
2. **Pares de sector/industria** (gratis, ya en el repo): mismos
   `market_universe.Industry.tickers`, con su fuerza relativa (RS Rating) y
   estado técnico actual. (El cuadrante RRG del Bloque E no está
   implementado en esta pasada - Bloque E quedó fuera del alcance de esta
   sesión; si se implementa después, es la adición natural aquí.)
3. **Relaciones declaradas en documentos SEC** (la que responde
   literalmente a "le compra a tal"): búsqueda de texto completo en EDGAR
   (gratuita, sin clave) de qué otras empresas mencionan a la empresa
   analizada en su propio 10-K/10-Q - la señal de relación comercial más
   literal disponible sin pagar por un feed. Solo funciona para tickers
   estadounidenses (EDGAR no cubre emisores no domiciliados en EE.UU. de la
   misma forma) - para Europa, `disclosed_available=False` explícito, nunca
   una lista vacía silenciosa que parezca "sin relaciones". Cacheado 30+
   días vía `durable_cache` (las relaciones de un 10-K cambian una vez al
   año, no en cada request) y nunca llamado desde un camino caliente - un
   único ticker por análisis, igual que `get_ticker_info`. Si EDGAR falla o
   va lento, degrada a las capas 1 y 2 sin romper la pantalla.

    Alcance explícito, no implementado en esta pasada: extraer la sección de
    concentración de clientes del propio 10-K de la empresa (el otro sentido
    de la relación que el encargo pide) necesitaría resolver el CIK del
    ticker y parsear el documento completo, no solo la búsqueda de texto
    completo - más frágil de lo que esta pasada puede verificar con
    confianza sin pruebas contra la API real. Queda para una iteración
    posterior, señalado aquí en vez de construido a medias.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sqlalchemy.orm import Session

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services import durable_cache
from app.services import watchlist_service as wl
from app.services.market_data_service import MarketDataService
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import DEFAULT_REGION, region_config, sector_of

logger = logging.getLogger(__name__)

# --- Layer 1: statistical relations -----------------------------------------

CORRELATION_SHORT_WINDOW = 60  # trading days
CORRELATION_LONG_WINDOW = 250  # trading days - ~1 year, the "historical baseline" divergence compares against
MIN_OBSERVATIONS = 30  # overlapping return observations before trusting any statistic below
LEAD_LAG_RANGE = range(-5, 6)  # sessions - matches the brief's own k in {-5..+5}
EXTREME_DAY_SIGMA = 2.0
DIVERGENCE_MIN_HISTORICAL_CORRELATION = 0.5  # only a pair that WAS genuinely correlated can "diverge"
DIVERGENCE_DROP_THRESHOLD = 0.3  # how far the 60d correlation must fall below the 250d baseline to flag it
TOP_N_STATISTICAL_RELATIONS = 15


@dataclass(frozen=True, slots=True)
class StatisticalRelation:
    ticker: str
    sector: str | None
    correlation_60d: float | None
    correlation_250d: float | None
    relative_beta: float | None  # how much this ticker moves for a 1% move in the analyzed ticker
    lead_lag_days: int | None  # >0: analyzed ticker leads this one by that many sessions; <0: the reverse
    lead_lag_correlation: float | None  # the correlation achieved at that best lag
    comovement_extreme_days_pct: float | None  # fraction of the analyzed ticker's >=2sigma days this moved with it
    is_diverging: bool  # historically correlated (>=250d baseline) but recently decoupled
    setup: str | None  # this ticker's own current watchlist setup, if it has one today
    percentile_score: float | None

    @property
    def setup_label(self) -> str | None:
        """See `watchlist_service.WatchlistItem.setup_label` - same root-cause
        fix (recomendación FE-1 de la cuarta auditoría independiente)."""
        return wl.SETUP_LABELS.get(self.setup) if self.setup else None


def _lead_lag(aligned: pd.DataFrame) -> tuple[int | None, float | None]:
    """Best (lag_days, correlation) - lag_days > 0 means column "a" (the
    analyzed ticker) leads "b" by that many sessions: a's return at t
    predicts b's return at t+lag_days. `aligned` must already be a 2-column
    ("a", "b") frame with no NaN gaps."""
    best_lag: int | None = None
    best_corr = 0.0
    for lag in LEAD_LAG_RANGE:
        shifted_b = aligned["b"].shift(-lag)
        corr = aligned["a"].corr(shifted_b)
        if pd.notna(corr) and abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, float(corr)
    if best_lag is None:
        return None, None
    return best_lag, best_corr


def _relative_beta(aligned: pd.DataFrame) -> float | None:
    var_a = aligned["a"].var()
    if not var_a or pd.isna(var_a):
        return None
    cov = aligned["a"].cov(aligned["b"])
    if pd.isna(cov):
        return None
    return float(cov / var_a)


def _extreme_day_comovement(aligned: pd.DataFrame, sigma: float = EXTREME_DAY_SIGMA) -> float | None:
    std_a = aligned["a"].std()
    if not std_a or pd.isna(std_a):
        return None
    extreme = aligned[aligned["a"].abs() >= sigma * std_a]
    if extreme.empty:
        return None
    same_direction = np.sign(extreme["a"]) == np.sign(extreme["b"])
    return float(same_direction.mean())


def compute_statistical_relations(
    ticker: str,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    universe_snapshot: list[TickerSnapshot],
    top_n: int = TOP_N_STATISTICAL_RELATIONS,
) -> list[StatisticalRelation]:
    """The `top_n` most related tickers to `ticker` by |correlation_60d|
    (falling back to |lead_lag_correlation| for a pair too thin on 60-session
    history but with enough for the lead-lag scan), computed entirely from
    `ohlcv_by_ticker` - the same universe-wide OHLCV
    `MarketScreenerService.get_universe_snapshot`/`get_cached_ohlcv` already
    holds in memory. No network call, no per-pair cost beyond a handful of
    vectorized pandas operations."""
    target_df = ohlcv_by_ticker.get(ticker)
    if target_df is None or target_df.empty:
        return []
    target_returns = target_df["close"].pct_change().dropna()
    if len(target_returns) < MIN_OBSERVATIONS:
        return []

    sector_by_ticker = {s.ticker: s.sector for s in universe_snapshot}
    setup_by_ticker: dict[str, tuple[str | None, float | None]] = {}
    for item in wl.build_watchlist(universe_snapshot):
        if item.ticker not in setup_by_ticker:
            setup_by_ticker[item.ticker] = (item.setup, item.percentile_score)

    relations: list[StatisticalRelation] = []
    for other_ticker, other_df in ohlcv_by_ticker.items():
        if other_ticker == ticker or other_df is None or other_df.empty:
            continue
        other_returns = other_df["close"].pct_change().dropna()
        aligned = pd.DataFrame({"a": target_returns, "b": other_returns}).dropna()
        if len(aligned) < MIN_OBSERVATIONS:
            continue

        # A zero-variance window (a ticker genuinely flat for the whole
        # window - halted, a data gap, a very thin fake/test series) makes
        # pandas' own .corr() return NaN, not a real correlation - treated
        # as "undefined", same as too little history, never left as a
        # silent NaN masquerading as a float in the output.
        recent = aligned.iloc[-CORRELATION_SHORT_WINDOW:]
        corr_60d = None
        if len(recent) >= MIN_OBSERVATIONS:
            raw_corr_60d = recent["a"].corr(recent["b"])
            corr_60d = float(raw_corr_60d) if pd.notna(raw_corr_60d) else None
        corr_250d = None
        if len(aligned) >= CORRELATION_LONG_WINDOW:
            long_window = aligned.iloc[-CORRELATION_LONG_WINDOW:]
            raw_corr_250d = long_window["a"].corr(long_window["b"])
            corr_250d = float(raw_corr_250d) if pd.notna(raw_corr_250d) else None

        lag_days, lag_corr = _lead_lag(aligned)
        beta = _relative_beta(aligned)
        comovement = _extreme_day_comovement(aligned)

        is_diverging = (
            corr_60d is not None
            and corr_250d is not None
            and corr_250d >= DIVERGENCE_MIN_HISTORICAL_CORRELATION
            and (corr_250d - corr_60d) >= DIVERGENCE_DROP_THRESHOLD
        )

        setup, percentile_score = setup_by_ticker.get(other_ticker, (None, None))
        ranking_value = abs(corr_60d) if corr_60d is not None else abs(lag_corr) if lag_corr is not None else 0.0
        relations.append(
            (
                ranking_value,
                StatisticalRelation(
                    ticker=other_ticker,
                    sector=sector_by_ticker.get(other_ticker) or sector_of(other_ticker),
                    correlation_60d=corr_60d,
                    correlation_250d=corr_250d,
                    relative_beta=beta,
                    lead_lag_days=lag_days,
                    lead_lag_correlation=lag_corr,
                    comovement_extreme_days_pct=comovement,
                    is_diverging=is_diverging,
                    setup=setup,
                    percentile_score=percentile_score,
                ),
            )
        )

    relations.sort(key=lambda pair: pair[0], reverse=True)
    return [relation for _, relation in relations[:top_n]]


# --- Layer 2: sector/industry peers ------------------------------------------


@dataclass(frozen=True, slots=True)
class SectorPeer:
    ticker: str
    sector: str
    industry: str
    rs_rating: int | None
    trend: str
    setup: str | None
    percentile_score: float | None

    @property
    def setup_label(self) -> str | None:
        return wl.SETUP_LABELS.get(self.setup) if self.setup else None


def compute_sector_peers(ticker: str, region: str, universe_snapshot: list[TickerSnapshot]) -> list[SectorPeer]:
    """Every other ticker in the same curated `Industry` bucket as `ticker`
    (trivial - `market_universe.py` already maps ticker -> sector -> industry
    - Segunda auditoría's own note on this), with its current RS Rating/trend
    and today's watchlist setup if it has one. No RRG quadrant here yet -
    see the module docstring."""
    industries = region_config(region).industries
    own_industry = next((ind for ind in industries if ticker in ind.tickers), None)
    if own_industry is None:
        return []

    snapshot_by_ticker = {s.ticker: s for s in universe_snapshot}
    setup_by_ticker: dict[str, tuple[str | None, float | None]] = {}
    for item in wl.build_watchlist(universe_snapshot):
        if item.ticker not in setup_by_ticker:
            setup_by_ticker[item.ticker] = (item.setup, item.percentile_score)

    peers = []
    for peer_ticker in own_industry.tickers:
        if peer_ticker == ticker:
            continue
        snapshot = snapshot_by_ticker.get(peer_ticker)
        setup, percentile_score = setup_by_ticker.get(peer_ticker, (None, None))
        peers.append(
            SectorPeer(
                ticker=peer_ticker,
                sector=own_industry.sector,
                industry=own_industry.name,
                rs_rating=snapshot.rs_rating if snapshot else None,
                trend=snapshot.trend.value if snapshot else "desconocido",
                setup=setup,
                percentile_score=percentile_score,
            )
        )
    peers.sort(key=lambda p: (p.rs_rating is None, -(p.rs_rating or 0)))
    return peers


# --- Layer 3: SEC EDGAR full-text search (US tickers only) ------------------

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
# SEC's own documented etiquette (sec.gov/os/webmaster-faq#developers): a
# descriptive User-Agent identifying the client, same rationale as
# dynamic_universe_service.py's WIKIPEDIA_USER_AGENT.
EDGAR_USER_AGENT = "QuantumAlphaPortfolioTool/1.0 (personal-use research tool; contact via repository)"
EDGAR_CACHE_TTL_DAYS = 30  # the brief's own floor: 10-K-derived relationships change once a year, not per request
EDGAR_LOOKBACK_DAYS = 365 * 2  # only recent filings - a mention in a 10-K from 5 years ago may no longer hold
EDGAR_MAX_RESULTS = 10
EDGAR_TIMEOUT_SECONDS = 10


@dataclass(frozen=True, slots=True)
class DisclosedRelation:
    """One SEC full-text search hit: another company's 10-K/10-Q that
    mentions the analyzed company by name - the most literal available
    signal of a real commercial relationship, and (per the module docstring)
    the most speculative layer: a text match, not a verified fact. Always
    carries its own source form and filing date so a caller can judge
    whether it's still current (a 10-K from 2022 may describe a relationship
    that's since ended)."""

    filer_name: str
    filer_ticker: str | None
    form: str
    filing_date: date


def _fetch_edgar_mentions(company_name: str) -> list[DisclosedRelation] | None:
    """Raw EDGAR full-text search for `company_name` inside other filers' own
    10-K/10-Q text - `None` (not `[]`) on any failure (network, malformed
    response, rate limit), so the caller can tell "genuinely found nothing"
    apart from "couldn't check"."""
    end = date.today()
    start = end - timedelta(days=EDGAR_LOOKBACK_DAYS)
    params = {
        "q": f'"{company_name}"',
        "forms": "10-K,10-Q",
        "dateRange": "custom",
        "startdt": start.isoformat(),
        "enddt": end.isoformat(),
    }
    try:
        response = requests.get(
            EDGAR_SEARCH_URL, params=params, headers={"User-Agent": EDGAR_USER_AGENT},
            timeout=EDGAR_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        logger.exception("EDGAR full-text search failed for %r", company_name)
        return None

    hits = payload.get("hits", {}).get("hits", []) if isinstance(payload, dict) else []
    relations = []
    for hit in hits[:EDGAR_MAX_RESULTS]:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        display_names = source.get("display_names") or []
        filer_name = display_names[0] if display_names else None
        if not filer_name:
            continue
        filing_date_raw = source.get("file_date")
        try:
            filing_date = pd.Timestamp(filing_date_raw).date() if filing_date_raw else None
        except Exception:
            filing_date = None
        if filing_date is None:
            continue
        relations.append(
            DisclosedRelation(
                filer_name=str(filer_name),
                filer_ticker=None,  # the search index doesn't reliably carry the filer's own ticker
                form=str(source.get("form") or source.get("root_form") or "?"),
                filing_date=filing_date,
            )
        )
    return relations


def get_disclosed_relations(
    ticker: str, company_name: str | None, db: Session | None
) -> tuple[list[DisclosedRelation] | None, bool]:
    """Returns (relations, available). `available=False` means "don't even
    show this layer" (non-US ticker, or no company name to search for) -
    distinct from `relations=[]` with `available=True` ("checked EDGAR,
    genuinely found nothing"). Cached >= EDGAR_CACHE_TTL_DAYS via
    durable_cache, keyed by ticker - a 10-K's disclosed relationships don't
    change between requests the way a live quote does."""
    if company_name is None:
        return None, False

    cache_key = f"edgar_mentions:{ticker}"
    max_age = timedelta(days=EDGAR_CACHE_TTL_DAYS)
    if db is not None:

        def _reconstruct(payload: list[dict]) -> list[DisclosedRelation]:
            return [
                DisclosedRelation(
                    filer_name=r["filer_name"], filer_ticker=r["filer_ticker"], form=r["form"],
                    filing_date=pd.Timestamp(r["filing_date"]).date(),
                )
                for r in payload
            ]

        cached = durable_cache.load_fresh_as(db, cache_key, max_age, _reconstruct)
        if cached is not None:
            return cached, True

    relations = _fetch_edgar_mentions(company_name)
    if relations is None:
        return None, False  # EDGAR itself failed - degrade silently, never show a broken/empty layer as "checked"

    if db is not None:
        durable_cache.save(
            db, cache_key,
            [
                {
                    "filer_name": r.filer_name, "filer_ticker": r.filer_ticker, "form": r.form,
                    "filing_date": r.filing_date.isoformat(),
                }
                for r in relations
            ],
        )
    return relations, True


# --- Orchestration ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelationshipMap:
    ticker: str
    region: str
    statistical: list[StatisticalRelation]
    sector_peers: list[SectorPeer]
    disclosed: list[DisclosedRelation] | None
    disclosed_available: bool
    computed_at: datetime


def build_relationship_map(
    ticker: str,
    region: str,
    screener: MarketScreenerService,
    market_data: MarketDataService,
    db: Session | None = None,
) -> RelationshipMap:
    """The full three-layer map for one ticker - "buscar nuevas opciones" is
    the whole point, so every related name carries its own current setup/
    percentile_score when it has one, not just a static label. Layer 3
    (EDGAR) only ever runs for `region == "us"` - EDGAR doesn't cover non-US
    issuers the same way, and the brief is explicit: for Europe, say so
    (`disclosed_available=False`), never show an empty list that reads as
    "no relationships found"."""
    universe_snapshot = screener.get_universe_snapshot(region=region, db=db)
    ohlcv_by_ticker = screener.get_cached_ohlcv(region=region)

    statistical = compute_statistical_relations(ticker, ohlcv_by_ticker, universe_snapshot)
    sector_peers = compute_sector_peers(ticker, region, universe_snapshot)

    disclosed: list[DisclosedRelation] | None = None
    disclosed_available = False
    if region == DEFAULT_REGION:  # "us" - EDGAR only meaningfully covers US-domiciled filers
        info = market_data.get_ticker_info(ticker)
        company_name = info.name if info else None
        disclosed, disclosed_available = get_disclosed_relations(ticker, company_name, db)

    return RelationshipMap(
        ticker=ticker,
        region=region,
        statistical=statistical,
        sector_peers=sector_peers,
        disclosed=disclosed,
        disclosed_available=disclosed_available,
        computed_at=datetime.now(UTC),
    )
