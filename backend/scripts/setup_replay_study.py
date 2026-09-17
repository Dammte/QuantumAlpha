"""Parte 10 del encargo (biblioteca de setups del Radar), cierre: script
offline que ejecuta `setup_replay.py` contra el universo real y persiste el
resultado en la tabla `setup_performance` - el mismo papel que
`factor_ablation_study.py` cumple para `backtest_engine.py`. Reutiliza sus
mismas funciones de resolución/descarga del universo
(`resolve_universe_tickers`/`download_universe_ohlcv`) en vez de repetir esa
pieza - ambas son funciones puras de propósito general, sin nada específico
de la ablación de factores en su firma ni en su comportamiento.

Uso (desde `backend/`):
    python scripts/setup_replay_study.py [--regions us europe]

Tarda varios minutos (descarga yfinance de ~217 tickers x 10 años, igual que
`factor_ablation_study.py`, más la propia reproducción punto-en-el-tiempo de
los siete detectores en cada punto de la rejilla de cada ticker) - un script
de investigación/calibración offline, nunca parte de un camino de petición
en vivo ni del propio `daily_close.py` (que solo LEE la tabla que este
script escribe, vía `setup_replay.apply_measured_confidence`)."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain.models.setup_performance import SetupPerformance  # noqa: E402
from app.infrastructure.db.repositories.setup_performance_repository import (
    SetupPerformanceRepository,  # noqa: E402
)
from app.infrastructure.db.session import SessionLocal  # noqa: E402
from app.services import setup_replay as sr  # noqa: E402
from scripts.factor_ablation_study import download_universe_ohlcv  # noqa: E402

MIN_BARS_REQUIRED = sr.REPLAY_WARMUP_BARS + 100  # margen sobre el mínimo real, igual que factor_ablation_study.py


def run_setup_replay_study(regions: list[str]) -> list[SetupPerformance]:
    """Descarga el universo, reproduce los siete detectores punto-en-el-
    tiempo sobre cada ticker, y agrega el resultado - sin tocar la base de
    datos (eso es tarea de `main`, para que esta función se pueda probar/
    invocar sin persistir nada)."""
    ohlcv_by_ticker, benchmark_ticker_by_ticker, _vix_close, ticker_region = download_universe_ohlcv(regions)

    all_observations: list[sr.SetupReplayObservation] = []
    for ticker, region in sorted(ticker_region.items()):
        df = ohlcv_by_ticker.get(ticker)
        if df is None or len(df) < MIN_BARS_REQUIRED:
            continue
        benchmark_ticker = benchmark_ticker_by_ticker.get(ticker)
        benchmark_df = ohlcv_by_ticker.get(benchmark_ticker) if benchmark_ticker else None
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None

        observations = sr.replay_setups_for_ticker(df, ticker, region, benchmark_close=benchmark_close)
        if observations:
            print(f"[{ticker}] {len(observations)} observaciones READY")
        all_observations.extend(observations)

    print(f"\nTotal: {len(all_observations)} observaciones sobre {len(ticker_region)} tickers")
    return _stats_to_rows(sr.aggregate_setup_performance(all_observations))


def _stats_to_rows(stats: list[sr.SetupPerformanceStats]) -> list[SetupPerformance]:
    now = datetime.now(UTC)
    return [
        SetupPerformance(
            id=None,
            setup_name=s.setup_name,
            family=s.family,
            grade=s.grade,
            market_regime=s.market_regime,
            n_observations=s.n_observations,
            trigger_rate=s.trigger_rate,
            win_rate=s.win_rate,
            expectancy_r=s.expectancy_r,
            median_bars_held=s.median_bars_held,
            mae_p80_pct=s.mae_p80_pct,
            failure_rate_3d=s.failure_rate_3d,
            confidence=s.confidence.value,
            computed_at=now,
        )
        for s in stats
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=["us", "europe"])
    args = parser.parse_args()

    rows = run_setup_replay_study(args.regions)

    db = SessionLocal()
    try:
        SetupPerformanceRepository(db).replace_all(rows)
    finally:
        db.close()
    print(f"\nPersistidas {len(rows)} filas en setup_performance.")

    # Parte 10.3: solo la fila sin segmentar de cada nombre - las filas por
    # grado/régimen son para el análisis, no para este resumen de consola.
    overall = [r for r in rows if r.grade is None and r.market_regime is None]
    measured = [r for r in overall if r.confidence == "measured"]
    print(f"\n{len(measured)}/{len(overall)} setups alcanzaron MEASURED (>= {sr.MIN_SAMPLE_FOR_STATS} disparos):")
    for r in sorted(measured, key=lambda row: row.expectancy_r or 0.0, reverse=True):
        win = f"{r.win_rate:.0%}" if r.win_rate is not None else "?"
        expectancy = f"{r.expectancy_r:.2f}R" if r.expectancy_r is not None else "?"
        print(f"  {r.setup_name}: n={r.n_observations} win_rate={win} expectancy={expectancy}")


if __name__ == "__main__":
    main()
