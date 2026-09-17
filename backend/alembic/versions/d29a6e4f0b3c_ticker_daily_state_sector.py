"""ticker_daily_states.sector / sector_rs_percentile (Parte 8, biblioteca de setups del Radar)

Revision ID: d29a6e4f0b3c
Revises: b4c8f3e2a7d1
Create Date: 2026-09-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd29a6e4f0b3c'
down_revision: Union[str, None] = 'b4c8f3e2a7d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ticker_daily_states', sa.Column('sector', sa.String(length=60), nullable=True))
    op.add_column('ticker_daily_states', sa.Column('sector_rs_percentile', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('ticker_daily_states', 'sector_rs_percentile')
    op.drop_column('ticker_daily_states', 'sector')
