"""ticker_daily_states: broken_levels
(Auditoria del Radar, bloque 10: subseccion "rompiendo por abajo")

Revision ID: a4c6e9d2f7b1
Revises: e8b3f6a1d4c7
Create Date: 2026-09-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a4c6e9d2f7b1'
down_revision: Union[str, None] = 'e8b3f6a1d4c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ticker_daily_states', sa.Column('broken_levels', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('ticker_daily_states', 'broken_levels')
