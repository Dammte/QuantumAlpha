"""Real macro-economic context via FRED - the Treasury yield curve,
unemployment and inflation data this app otherwise has no source for.

The yield curve spread (10-year minus 2-year Treasury yield) inverting
(going negative) is one of the most well-documented recession leading
indicators - every US recession since the 1970s was preceded by an
inversion, though with a lead time that varies widely (roughly 6-24
months), so it's a warning to weigh, not a timing signal.

The whole snapshot is None when no `FRED_API_KEY` is configured (see
`app/core/config.py`) or every series fails to fetch - callers already treat
a missing macro read the same way they treat missing fundamentals or news.
"""

from dataclasses import dataclass

from app.infrastructure.macro_data.fred_client import FredClient

YIELD_CURVE_SERIES = "T10Y2Y"
UNEMPLOYMENT_SERIES = "UNRATE"
CPI_SERIES = "CPIAUCSL"


@dataclass(frozen=True, slots=True)
class MacroSnapshot:
    yield_curve_spread: float | None
    yield_curve_date: str | None
    yield_curve_inverted: bool
    unemployment_rate: float | None
    unemployment_date: str | None
    cpi_yoy_change: float | None
    cpi_date: str | None


class MacroDataService:
    def __init__(self, client: FredClient) -> None:
        self.client = client

    def get_macro_snapshot(self) -> MacroSnapshot | None:
        if not self.client.is_configured:
            return None

        yield_curve = self.client.latest_observation(YIELD_CURVE_SERIES)
        unemployment = self.client.latest_observation(UNEMPLOYMENT_SERIES)
        cpi_yoy = self.client.latest_observation(CPI_SERIES, units="pc1")

        if yield_curve is None and unemployment is None and cpi_yoy is None:
            return None

        yield_curve_date, yield_curve_value = yield_curve if yield_curve else (None, None)
        unemployment_date, unemployment_value = unemployment if unemployment else (None, None)
        cpi_date, cpi_value = cpi_yoy if cpi_yoy else (None, None)

        return MacroSnapshot(
            yield_curve_spread=yield_curve_value,
            yield_curve_date=yield_curve_date,
            yield_curve_inverted=yield_curve_value is not None and yield_curve_value < 0,
            unemployment_rate=unemployment_value,
            unemployment_date=unemployment_date,
            cpi_yoy_change=cpi_value,
            cpi_date=cpi_date,
        )
