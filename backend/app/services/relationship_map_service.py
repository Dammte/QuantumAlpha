"""Tercera auditoría, Bloque G (función nueva, explícitamente pedida): "que se
muestren acciones relacionadas... un mapa completo del proceso para buscar
nuevas opciones."

No existe ninguna fuente gratuita y fiable de cadena de suministro
estructurada (FactSet Supply Chain y similares son de pago). Este módulo
construye el mapa en dos capas, de la más fiable/objetiva a la más
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
   estado técnico actual.

2026-09: la tercera capa (menciones en documentos SEC EDGAR - qué otras
empresas citan a la analizada en su propio 10-K/10-Q) se retiró: lenta, solo
cubría tickers estadounidenses, y una mención en un 10-K no mueve un trade
de días (ver docs/quant_methodology.md). Las dos capas que quedan no tenían
esa dependencia de horizonte ni de red - sobreviven sin cambios.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.domain.models.ticker_snapshot import TickerSnapshot
from app.services.market_screener_service import MarketScreenerService
from app.services.market_universe import region_config, sector_of

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


def compute_sector_peers(ticker: str, region: str, universe_snapshot: list[TickerSnapshot]) -> list[SectorPeer]:
    """Every other ticker in the same curated `Industry` bucket as `ticker`
    (trivial - `market_universe.py` already maps ticker -> sector -> industry
    - Segunda auditoría's own note on this), with its current RS Rating/trend.
    No RRG quadrant here yet - see the module docstring.

    2026-09 (Fase 5 retirement): used to also annotate each peer with its
    current `watchlist_service.py` setup/percentile score - dropped along
    with that module's retirement (docs/quant_methodology.md §25) rather
    than duplicating its setup-detection logic here just to keep two badge
    fields alive. `GET /market/radar` is the surface for "which tickers have
    an active setup right now" going forward."""
    industries = region_config(region).industries
    own_industry = next((ind for ind in industries if ticker in ind.tickers), None)
    if own_industry is None:
        return []

    snapshot_by_ticker = {s.ticker: s for s in universe_snapshot}

    peers = []
    for peer_ticker in own_industry.tickers:
        if peer_ticker == ticker:
            continue
        snapshot = snapshot_by_ticker.get(peer_ticker)
        peers.append(
            SectorPeer(
                ticker=peer_ticker,
                sector=own_industry.sector,
                industry=own_industry.name,
                rs_rating=snapshot.rs_rating if snapshot else None,
                trend=snapshot.trend.value if snapshot else "desconocido",
            )
        )
    peers.sort(key=lambda p: (p.rs_rating is None, -(p.rs_rating or 0)))
    return peers


# --- Orchestration ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelationshipMap:
    ticker: str
    region: str
    statistical: list[StatisticalRelation]
    sector_peers: list[SectorPeer]
    computed_at: datetime


def build_relationship_map(
    ticker: str,
    region: str,
    screener: MarketScreenerService,
    db: Session | None = None,
) -> RelationshipMap:
    """The two-layer map for one ticker - "buscar nuevas opciones" is the
    whole point."""
    universe_snapshot = screener.get_universe_snapshot(region=region, db=db)
    ohlcv_by_ticker = screener.get_cached_ohlcv(region=region)

    statistical = compute_statistical_relations(ticker, ohlcv_by_ticker, universe_snapshot)
    sector_peers = compute_sector_peers(ticker, region, universe_snapshot)

    return RelationshipMap(
        ticker=ticker,
        region=region,
        statistical=statistical,
        sector_peers=sector_peers,
        computed_at=datetime.now(UTC),
    )
