"""ticker_daily_states.setups (biblioteca de setups del Radar, en curso)

Revision ID: e1297786f1da
Revises: 4f7f279777aa
Create Date: 2026-09-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e1297786f1da'
down_revision: Union[str, None] = '4f7f279777aa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ticker_daily_states', sa.Column('setups', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('ticker_daily_states', 'setups')
