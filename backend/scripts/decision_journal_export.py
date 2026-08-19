"""Dumps every recorded verdict/position signal, together with its realized
forward return at 5/10/21/63 sessions, to a CSV - so the whole decision
history can be audited outside the app (a spreadsheet, a notebook, a
second pair of eyes) instead of only through the aggregated
`GET /api/v1/system/signal-performance` endpoint.

Reads straight from the real database (`RecommendationSnapshotORM` +
`PositionSignalSnapshotORM`) via the same repositories the API uses, then
fetches price history for every distinct ticker in a *single* batched call
(never one network round-trip per ticker - see `PortfolioRiskService`'s
docstring for the incident this is careful not to repeat) to compute the
forward returns with `signal_performance_service.forward_return`, the exact
same function the API's aggregated report uses - this export and that
endpoint can never quietly disagree on what "the return after this
snapshot" means.

Usage:
    python scripts/decision_journal_export.py [--out decision_journal.csv]

Requires DATABASE_URL to point at the real database (reads `.env` via the
app's own settings) - this is an offline audit script, not something that
runs as part of the API.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.db.repositories.position_signal_snapshot_repository import (  # noqa: E402
    PositionSignalSnapshotRepository,
)
from app.infrastructure.db.repositories.recommendation_snapshot_repository import (  # noqa: E402
    RecommendationSnapshotRepository,
)
from app.infrastructure.db.session import SessionLocal  # noqa: E402
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider  # noqa: E402
from app.services import signal_performance_service as sps  # noqa: E402
from app.services.market_data_service import MarketDataService  # noqa: E402

HISTORY_YEARS = 10


def _price_by_ticker(market_data: MarketDataService, tickers: list[str]) -> dict[str, pd.Series]:
    if not tickers:
        return {}
    end = date.today()
    start = date(end.year - HISTORY_YEARS, end.month, end.day)
    ohlcv = market_data.get_bulk_ohlcv(tickers, start, end)
    return {ticker: frame["close"] for ticker, frame in ohlcv.items()}


def build_decision_journal() -> pd.DataFrame:
    db = SessionLocal()
    try:
        recommendation_snapshots = RecommendationSnapshotRepository(db).list_all()
        position_snapshots = PositionSignalSnapshotRepository(db).list_all()
    finally:
        db.close()

    market_data = MarketDataService(YFinanceProvider())
    tickers = sorted({s.ticker for s in recommendation_snapshots} | {s.ticker for s in position_snapshots})
    price_by_ticker = _price_by_ticker(market_data, tickers)

    rows = []
    for snap in recommendation_snapshots:
        row = {
            "type": "verdict",
            "portfolio_id": None,
            "ticker": snap.ticker,
            "created_at": snap.created_at,
            "label": snap.verdict,
            "exit_urgency": None,
            "score": snap.score,
            "price": snap.price,
            "r_multiple": None,
        }
        close = price_by_ticker.get(snap.ticker)
        for horizon in sps.FORWARD_HORIZONS:
            row[f"fwd_return_{horizon}d"] = (
                sps.forward_return(close, snap.created_at.date(), horizon) if close is not None else None
            )
        rows.append(row)

    for snap in position_snapshots:
        row = {
            "type": "position_signal",
            "portfolio_id": snap.portfolio_id,
            "ticker": snap.ticker,
            "created_at": snap.created_at,
            "label": snap.signal,
            "exit_urgency": snap.exit_urgency,
            "score": snap.score,
            "price": snap.price,
            "r_multiple": snap.r_multiple,
        }
        close = price_by_ticker.get(snap.ticker)
        for horizon in sps.FORWARD_HORIZONS:
            row[f"fwd_return_{horizon}d"] = (
                sps.forward_return(close, snap.created_at.date(), horizon) if close is not None else None
            )
        rows.append(row)

    if not rows:
        columns = [
            "type", "portfolio_id", "ticker", "created_at", "label", "exit_urgency", "score", "price",
            "r_multiple", *(f"fwd_return_{h}d" for h in sps.FORWARD_HORIZONS),
        ]
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("created_at")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="decision_journal.csv", help="Output CSV path")
    args = parser.parse_args()

    journal = build_decision_journal()
    journal.to_csv(args.out, index=False)
    print(f"{len(journal)} rows written to {args.out}")
