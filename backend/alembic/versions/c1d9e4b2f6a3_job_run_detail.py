"""job_runs.detail column (Auditoria del Radar, bloque C.1: resumen ejecutable)

Revision ID: c1d9e4b2f6a3
Revises: a3f8d1c2e5b7
Create Date: 2026-09-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c1d9e4b2f6a3'
down_revision: Union[str, None] = 'a3f8d1c2e5b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('job_runs', sa.Column('detail', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('job_runs', 'detail')
