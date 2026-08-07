from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.domain.models.transaction import TransactionType


class TransactionCreate(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)
    transaction_type: TransactionType
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    fees: float = Field(default=0, ge=0)
    executed_at: datetime | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        # Normalized here too (not just client-side) so the same ticker never
        # ends up split across "sap.de" and "SAP.DE" as two different assets.
        return value.strip().upper()


class TransactionRead(BaseModel):
    id: int
    portfolio_id: int
    ticker: str
    transaction_type: TransactionType
    quantity: float
    price: float
    fees: float
    executed_at: datetime
