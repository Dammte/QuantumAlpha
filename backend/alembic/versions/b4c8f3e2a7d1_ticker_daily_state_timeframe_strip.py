"""ticker_daily_states.timeframe_strip (Parte 7, biblioteca de setups del Radar)

Revision ID: b4c8f3e2a7d1
Revises: e1297786f1da
Create Date: 2026-09-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b4c8f3e2a7d1'
down_revision: Union[str, None] = 'e1297786f1da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ticker_daily_states', sa.Column('timeframe_strip', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('ticker_daily_states', 'timeframe_strip')
