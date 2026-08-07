from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Transaction:
    id: int | None
    portfolio_id: int
    ticker: str
    transaction_type: TransactionType
    quantity: float
    price: float
    fees: float
    executed_at: datetime

    @property
    def total_cost(self) -> float:
        signed_quantity = self.quantity if self.transaction_type == TransactionType.BUY else -self.quantity
        return signed_quantity * self.price + self.fees
