"""Monthly refresh of the point-in-time universe membership table (D14 -
Segunda auditoría, Bloque 3). See `dynamic_universe_service.py`'s module
docstring for the full reasoning (survivorship bias, why history is
appended rather than overwritten, why this is a batch script and never a
live-request code path).

Run by hand, or wired to a monthly cron (e.g. a Render Cron Job) - either
way, this is the *only* place a live index-constituent fetch happens; every
other caller (`premium_watchlist_service.py`, `watchlist_service.py`) only
ever reads the table this writes.

Usage:
    python scripts/refresh_universe_membership.py            # both regions, only if due
    python scripts/refresh_universe_membership.py --force    # refresh regardless of the 30-day interval
    python scripts/refresh_universe_membership.py --region us
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.db.repositories.universe_membership_repository import (  # noqa: E402
    UniverseMembershipRepository,
)
from app.infrastructure.db.session import SessionLocal  # noqa: E402
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider  # noqa: E402
from app.services import dynamic_universe_service as dus  # noqa: E402
from app.services.market_data_service import MarketDataService  # noqa: E402

REGIONS = ("us", "europe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=REGIONS, help="Refresh only this region (default: both)")
    parser.add_argument("--force", action="store_true", help="Refresh even if the last snapshot isn't stale yet")
    args = parser.parse_args()

    regions = [args.region] if args.region else list(REGIONS)
    market_data = MarketDataService(YFinanceProvider())
    db = SessionLocal()
    try:
        repo = UniverseMembershipRepository(db)
        for region in regions:
            if not args.force and not dus.is_refresh_due(repo, region):
                latest = repo.latest_as_of_date(region)
                print(f"[{region}] up to date (last snapshot: {latest}) - skipping, use --force to override")
                continue
            print(f"[{region}] refreshing universe membership...")
            count, source = dus.refresh_universe_membership(region, market_data, repo)
            print(f"[{region}] saved {count} tickers (source={source})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
