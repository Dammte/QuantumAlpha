"""trade_plans.initial_quantity (baseline for scaled-exit tracking)

Revision ID: 49ac08cef677
Revises: 81f9cb75933e
Create Date: 2026-08-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '49ac08cef677'
down_revision: Union[str, None] = '81f9cb75933e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # How many shares were held the moment this plan was created (real entry
    # or point-in-time reconstruction) - the baseline trade_manager.py
    # compares the *currently* held quantity against to know whether the
    # +1R/+2R scaled exits are still pending, already partially done, or
    # already complete, without needing a separate "already suggested" flag
    # that could drift from what was actually sold. Nullable + backfilled
    # from entry_price's own row via a data migration would be nicer, but
    # this table has no production data yet (introduced in the same
    # unreleased branch), so a straight NOT NULL default is safe here.
    op.add_column(
        'trade_plans', sa.Column('initial_quantity', sa.Numeric(precision=20, scale=8), nullable=False, server_default='0')
    )
    op.alter_column('trade_plans', 'initial_quantity', server_default=None)


def downgrade() -> None:
    op.drop_column('trade_plans', 'initial_quantity')
