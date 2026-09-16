"""ticker_daily_states.grade (Parte 5.3 reconstruction, later pass)

Revision ID: 4f7f279777aa
Revises: d3f7a2b8c1e4
Create Date: 2026-09-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '4f7f279777aa'
down_revision: Union[str, None] = 'd3f7a2b8c1e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ticker_daily_states', sa.Column('grade', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('ticker_daily_states', 'grade')
