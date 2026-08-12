from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    # Pure cash movements - no ticker, no market data involved. DEPOSIT is money
    # added to the portfolio from outside (a bank transfer into the brokerage
    # account); WITHDRAWAL is money taken out. Together with BUY/SELL these are
    # what let `PortfolioRepository.build_portfolio` derive an actual liquidity
    # balance instead of a sale's proceeds simply vanishing from the portfolio's
    # total once the position they came from is gone - see its docstring.
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"


# A DEPOSIT/WITHDRAWAL always carries price=1.0 (see PortfolioRepository.add_transaction)
# so `quantity` reads directly as a cash amount - these two are never fetched from
# market data, only entered by hand, always in the portfolio's own base currency.
CASH_MOVEMENT_TYPES = frozenset({TransactionType.DEPOSIT, TransactionType.WITHDRAWAL})
# Which type increases vs decreases the money side of the ledger - BUY spends cash to
# gain shares, DEPOSIT adds cash directly; SELL/WITHDRAWAL are the mirror image. Used
# both for `total_cost`'s external-cash-flow sign (see PortfolioService._cash_flow_series)
# and for the liquidity balance in `PortfolioRepository.build_portfolio`.
INFLOW_TYPES = frozenset({TransactionType.BUY, TransactionType.DEPOSIT})


@dataclass(frozen=True, slots=True)
class Transaction:
    id: int | None
    portfolio_id: int
    ticker: str | None  # None for DEPOSIT/WITHDRAWAL - a cash movement isn't tied to any asset
    transaction_type: TransactionType
    quantity: float
    price: float
    fees: float
    executed_at: datetime

    @property
    def total_cost(self) -> float:
        signed_quantity = self.quantity if self.transaction_type in INFLOW_TYPES else -self.quantity
        return signed_quantity * self.price + self.fees
