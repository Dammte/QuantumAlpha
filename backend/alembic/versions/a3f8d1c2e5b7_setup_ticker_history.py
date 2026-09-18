"""setup_ticker_history table (Parte 11.2, biblioteca de setups del Radar)

Revision ID: a3f8d1c2e5b7
Revises: f7a1c9e3b6d2
Create Date: 2026-09-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a3f8d1c2e5b7'
down_revision: Union[str, None] = 'f7a1c9e3b6d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'setup_ticker_history',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('region', sa.String(length=20), nullable=False),
        sa.Column('setup_name', sa.String(length=60), nullable=False),
        sa.Column('family', sa.String(length=30), nullable=False),
        sa.Column('n_observations', sa.Integer(), nullable=False),
        sa.Column('n_triggered', sa.Integer(), nullable=False),
        sa.Column('n_target_hit', sa.Integer(), nullable=False),
        sa.Column('first_ready_date', sa.Date(), nullable=False),
        sa.Column('last_ready_date', sa.Date(), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_setup_ticker_history_ticker_region', 'setup_ticker_history', ['ticker', 'region'])


def downgrade() -> None:
    op.drop_index('ix_setup_ticker_history_ticker_region', table_name='setup_ticker_history')
    op.drop_table('setup_ticker_history')
