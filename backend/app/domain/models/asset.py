from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    BOND = "bond"
    CASH = "cash"


@dataclass(frozen=True, slots=True)
class Asset:
    ticker: str
    name: str
    asset_class: AssetClass
    currency: str = "USD"
