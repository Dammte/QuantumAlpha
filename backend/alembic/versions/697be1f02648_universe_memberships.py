"""universe memberships (D14 - point-in-time universe, Segunda auditoría Bloque 3)

Revision ID: 697be1f02648
Revises: 9a3b5080be2d
Create Date: 2026-08-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '697be1f02648'
down_revision: Union[str, None] = '9a3b5080be2d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'universe_memberships',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('region', sa.String(length=20), nullable=False),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('sector', sa.String(length=80), nullable=True),
        sa.Column('as_of_date', sa.Date(), nullable=False),
        sa.Column('source', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('region', 'ticker', 'as_of_date', name='uq_universe_membership_region_ticker_date'),
    )
    op.create_index(
        'ix_universe_memberships_region_date', 'universe_memberships', ['region', 'as_of_date'], unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_universe_memberships_region_date', table_name='universe_memberships')
    op.drop_table('universe_memberships')
