"""recommendation snapshots (signal traceability audit log)

Revision ID: c45b38675825
Revises: 8727dfb1f509
Create Date: 2026-08-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c45b38675825'
down_revision: Union[str, None] = '8727dfb1f509'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'recommendation_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('verdict', sa.String(length=20), nullable=False),
        sa.Column('score', sa.Integer(), nullable=False),
        sa.Column('price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False),
        sa.Column('horizon', sa.String(length=10), nullable=False),
        sa.Column('engine_version', sa.String(length=40), nullable=False),
        sa.Column('factors', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_recommendation_snapshots_ticker'), 'recommendation_snapshots', ['ticker'], unique=False)
    op.create_index(
        op.f('ix_recommendation_snapshots_created_at'), 'recommendation_snapshots', ['created_at'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_recommendation_snapshots_created_at'), table_name='recommendation_snapshots')
    op.drop_index(op.f('ix_recommendation_snapshots_ticker'), table_name='recommendation_snapshots')
    op.drop_table('recommendation_snapshots')
