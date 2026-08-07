"""Thin client for FRED (Federal Reserve Economic Data) - a free, official
source for macro series this app has no other way to get (Treasury yield
curve, unemployment, inflation). Free instant-signup key at
fredaccount.stlouisfed.org/apikeys; without one configured, every call here
returns None so the rest of the app treats "no macro data" the same way it
already treats "no fundamentals" or "no news" for a given ticker - a normal,
expected state, not an error.
"""

import logging

import requests

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SECONDS = 10
RECENT_OBSERVATIONS_TO_SCAN = 10


class FredClient:
    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def latest_observation(self, series_id: str, units: str = "lin") -> tuple[str, float] | None:
        """Returns (date, value) for the most recent non-missing observation of
        a FRED series. `units="pc1"` asks FRED to compute the year-over-year %
        change server-side (used for CPI) instead of returning the raw index."""
        if not self.api_key:
            return None

        try:
            response = requests.get(
                FRED_BASE_URL,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": RECENT_OBSERVATIONS_TO_SCAN,
                    "units": units,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.exception("Failed to fetch FRED series %s", series_id)
            return None

        for observation in payload.get("observations", []):
            raw_value = observation.get("value")
            if raw_value is None or raw_value == ".":
                continue  # FRED's own marker for "no data this period"
            try:
                return observation["date"], float(raw_value)
            except (TypeError, ValueError, KeyError):
                continue
        return None
