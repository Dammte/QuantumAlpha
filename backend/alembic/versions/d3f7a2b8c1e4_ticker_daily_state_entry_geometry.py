"""ticker_daily_states.entry_geometry (Parte 7 reconstruction, later pass)

Revision ID: d3f7a2b8c1e4
Revises: 8706dbd91554
Create Date: 2026-09-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd3f7a2b8c1e4'
down_revision: Union[str, None] = '8706dbd91554'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ticker_daily_states', sa.Column('entry_geometry', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('ticker_daily_states', 'entry_geometry')
