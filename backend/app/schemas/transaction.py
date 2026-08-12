from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.domain.models.transaction import CASH_MOVEMENT_TYPES, TransactionType


class TransactionCreate(BaseModel):
    # None for DEPOSIT/WITHDRAWAL - see Transaction's docstring. Required for BUY/SELL,
    # enforced below rather than at the field level so both shapes share one schema.
    ticker: str | None = Field(default=None, max_length=20)
    transaction_type: TransactionType
    quantity: float = Field(gt=0)  # shares for BUY/SELL; a cash amount for DEPOSIT/WITHDRAWAL
    # None for DEPOSIT/WITHDRAWAL: always forced to 1.0 server-side regardless of what's
    # sent here (see PortfolioRepository.add_transaction) so a cash movement can never be
    # smuggled in at a fabricated "price". Required for BUY/SELL.
    price: float | None = Field(default=None, gt=0)
    fees: float = Field(default=0, ge=0)
    executed_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_by_type(self) -> "TransactionCreate":
        if self.transaction_type in CASH_MOVEMENT_TYPES:
            if self.ticker is not None:
                raise ValueError("Los depósitos y retiros de liquidez no llevan ticker")
        else:
            if not self.ticker:
                raise ValueError("El ticker es obligatorio para compras y ventas")
            if self.price is None:
                raise ValueError("El precio es obligatorio para compras y ventas")
        return self

    @model_validator(mode="after")
    def _normalize_ticker(self) -> "TransactionCreate":
        # Normalized here too (not just client-side) so the same ticker never
        # ends up split across "sap.de" and "SAP.DE" as two different assets.
        if self.ticker is not None:
            self.ticker = self.ticker.strip().upper()
        return self


class TransactionRead(BaseModel):
    id: int
    portfolio_id: int
    ticker: str | None
    transaction_type: TransactionType
    quantity: float
    price: float
    fees: float
    executed_at: datetime
