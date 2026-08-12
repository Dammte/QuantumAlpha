"""cash movements (deposit/withdrawal) and durable computation cache

Revision ID: f80147653a32
Revises: c45b38675825
Create Date: 2026-08-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f80147653a32'
down_revision: Union[str, None] = 'c45b38675825'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # New cash-movement transaction types - see app/domain/models/transaction.py.
    # Must run outside the values' own INSERT/SELECT usage (a bare ALTER TYPE is
    # fine as its own migration/transaction, just can't be combined with a DML
    # statement that uses the new value in that same transaction).
    #
    # Uppercase to match the existing 'BUY'/'SELL' labels: SQLAlchemy's `Enum`
    # column type stores a Python enum's *member name* (TransactionType.BUY ->
    # "BUY"), not its `.value` ("buy") - see the original migration
    # (8727dfb1f509) creating this type as sa.Enum('BUY', 'SELL', ...).
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'DEPOSIT'")
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'WITHDRAWAL'")

    # A DEPOSIT/WITHDRAWAL isn't tied to any asset, so it can no longer be a hard
    # NOT NULL FK - see Transaction.ticker's docstring.
    op.alter_column('transactions', 'ticker', existing_type=sa.String(length=20), nullable=True)

    op.create_table(
        'computation_cache',
        sa.Column('cache_key', sa.String(length=200), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('cache_key'),
    )
    op.create_index(
        op.f('ix_computation_cache_computed_at'), 'computation_cache', ['computed_at'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_computation_cache_computed_at'), table_name='computation_cache')
    op.drop_table('computation_cache')
    op.alter_column('transactions', 'ticker', existing_type=sa.String(length=20), nullable=False)
    # Postgres has no ALTER TYPE ... DROP VALUE - removing 'deposit'/'withdrawal'
    # from the enum would require rebuilding the type entirely; not worth it for
    # a downgrade path on a personal project, so this intentionally leaves them.
