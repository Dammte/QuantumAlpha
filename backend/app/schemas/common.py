"""Response models with no dependency on any other schema module - kept here
(rather than in `market.py`) purely so `quant_analysis.py` and `market.py` can
both use `PriceLevelResponse` without importing from each other."""

from pydantic import BaseModel


class PriceLevelResponse(BaseModel):
    price: float
    kind: str
    strength: int
    distance_pct: float
