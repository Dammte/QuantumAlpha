"""trade plans (persisted stop/target/thesis per open position - exit engine)

Revision ID: 81f9cb75933e
Revises: f80147653a32
Create Date: 2026-08-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '81f9cb75933e'
down_revision: Union[str, None] = 'f80147653a32'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'trade_plans',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('entry_date', sa.Date(), nullable=False),
        sa.Column('initial_stop', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('initial_target', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('current_stop', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('highest_close_since_entry', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('thesis', sa.String(length=500), nullable=False),
        sa.Column('engine_version', sa.String(length=40), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    # No uniqueness constraint on (portfolio_id, ticker) on purpose - a ticker
    # can be bought, fully sold, and bought again, and each lot gets its own
    # row (closed_at marks a lot as done). This index only makes "the open
    # plan for this ticker" (closed_at IS NULL) cheap to find.
    op.create_index(
        'ix_trade_plans_portfolio_ticker', 'trade_plans', ['portfolio_id', 'ticker'], unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_trade_plans_portfolio_ticker', table_name='trade_plans')
    op.drop_table('trade_plans')
