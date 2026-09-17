"""setup_performance table (Parte 10.2, biblioteca de setups del Radar)

Revision ID: f7a1c9e3b6d2
Revises: d29a6e4f0b3c
Create Date: 2026-09-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f7a1c9e3b6d2'
down_revision: Union[str, None] = 'd29a6e4f0b3c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'setup_performance',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('setup_name', sa.String(length=60), nullable=False),
        sa.Column('family', sa.String(length=30), nullable=False),
        sa.Column('grade', sa.String(length=1), nullable=True),
        sa.Column('market_regime', sa.String(length=30), nullable=True),
        sa.Column('n_observations', sa.Integer(), nullable=False),
        sa.Column('trigger_rate', sa.Numeric(6, 4), nullable=True),
        sa.Column('win_rate', sa.Numeric(6, 4), nullable=True),
        sa.Column('expectancy_r', sa.Numeric(10, 4), nullable=True),
        sa.Column('median_bars_held', sa.Numeric(10, 2), nullable=True),
        sa.Column('mae_p80_pct', sa.Numeric(6, 4), nullable=True),
        sa.Column('failure_rate_3d', sa.Numeric(6, 4), nullable=True),
        sa.Column('confidence', sa.String(length=20), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_setup_performance_setup_name', 'setup_performance', ['setup_name'])


def downgrade() -> None:
    op.drop_index('ix_setup_performance_setup_name', table_name='setup_performance')
    op.drop_table('setup_performance')
