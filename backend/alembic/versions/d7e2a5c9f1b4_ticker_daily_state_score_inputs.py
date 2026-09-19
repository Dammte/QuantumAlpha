"""ticker_daily_states: relative_volume, next_earnings_date, atr_pct
(Auditoria del Radar, bloque E2: entradas del score compuesto)

Revision ID: d7e2a5c9f1b4
Revises: c1d9e4b2f6a3
Create Date: 2026-09-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd7e2a5c9f1b4'
down_revision: Union[str, None] = 'c1d9e4b2f6a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ticker_daily_states', sa.Column('relative_volume', sa.Numeric(10, 4), nullable=True))
    op.add_column('ticker_daily_states', sa.Column('next_earnings_date', sa.Date(), nullable=True))
    op.add_column('ticker_daily_states', sa.Column('atr_pct', sa.Numeric(10, 6), nullable=True))


def downgrade() -> None:
    op.drop_column('ticker_daily_states', 'atr_pct')
    op.drop_column('ticker_daily_states', 'next_earnings_date')
    op.drop_column('ticker_daily_states', 'relative_volume')
