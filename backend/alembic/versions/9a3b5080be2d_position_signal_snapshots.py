"""position signal snapshots (Fase 0 instrumentation - position-level audit trail)

Revision ID: 9a3b5080be2d
Revises: 49ac08cef677
Create Date: 2026-08-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9a3b5080be2d'
down_revision: Union[str, None] = '49ac08cef677'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'position_signal_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('signal', sa.String(length=20), nullable=False),
        sa.Column('exit_urgency', sa.String(length=20), nullable=True),
        sa.Column('score', sa.Integer(), nullable=False),
        sa.Column('price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('r_multiple', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('engine_version', sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_position_signal_snapshots_ticker'), 'position_signal_snapshots', ['ticker'], unique=False
    )
    op.create_index(
        op.f('ix_position_signal_snapshots_created_at'), 'position_signal_snapshots', ['created_at'], unique=False
    )
    op.create_index(
        'ix_position_signal_snapshots_portfolio_ticker',
        'position_signal_snapshots',
        ['portfolio_id', 'ticker'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_position_signal_snapshots_portfolio_ticker', table_name='position_signal_snapshots')
    op.drop_index(op.f('ix_position_signal_snapshots_created_at'), table_name='position_signal_snapshots')
    op.drop_index(op.f('ix_position_signal_snapshots_ticker'), table_name='position_signal_snapshots')
    op.drop_table('position_signal_snapshots')
