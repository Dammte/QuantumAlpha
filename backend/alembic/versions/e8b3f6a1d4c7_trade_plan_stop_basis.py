"""trade_plans: initial_stop_basis, initial_stop_level_kind, current_stop_basis
(Auditoria del Radar, bloque H2: el ancla del stop en texto, no solo el numero)

Revision ID: e8b3f6a1d4c7
Revises: d7e2a5c9f1b4
Create Date: 2026-09-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e8b3f6a1d4c7'
down_revision: Union[str, None] = 'd7e2a5c9f1b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trade_plans', sa.Column('initial_stop_basis', sa.String(200), nullable=True))
    op.add_column('trade_plans', sa.Column('initial_stop_level_kind', sa.String(40), nullable=True))
    op.add_column('trade_plans', sa.Column('current_stop_basis', sa.String(200), nullable=True))


def downgrade() -> None:
    op.drop_column('trade_plans', 'current_stop_basis')
    op.drop_column('trade_plans', 'initial_stop_level_kind')
    op.drop_column('trade_plans', 'initial_stop_basis')
