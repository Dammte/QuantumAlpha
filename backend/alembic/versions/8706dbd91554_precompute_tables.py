"""precompute tables (Fase 2 reconstruction - job_runs, ticker_daily_states,
ticker_intraday_states, position_daily_states, trigger_events, daily_briefs)

Revision ID: 8706dbd91554
Revises: 697be1f02648
Create Date: 2026-09-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '8706dbd91554'
down_revision: Union[str, None] = '697be1f02648'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'job_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_name', sa.String(length=60), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('rows_processed', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.String(length=2000), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_job_runs_job_name'), 'job_runs', ['job_name'], unique=False)
    op.create_index(op.f('ix_job_runs_started_at'), 'job_runs', ['started_at'], unique=False)

    op.create_table(
        'ticker_daily_states',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('region', sa.String(length=20), nullable=False),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
        sa.Column('price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False),
        sa.Column('trend', sa.String(length=20), nullable=False),
        sa.Column('stage', sa.String(length=20), nullable=True),
        sa.Column('rs_rating', sa.Integer(), nullable=True),
        sa.Column('adx14', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('atr_multiple', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('rsi14', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('gate_passes', sa.Boolean(), nullable=False),
        sa.Column('gate_conditions', sa.JSON(), nullable=False),
        sa.Column('gate_version', sa.String(length=40), nullable=False),
        sa.Column('entry_trigger_type', sa.String(length=20), nullable=True),
        sa.Column('entry_trigger_price', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('entry_already_triggered', sa.Boolean(), nullable=False),
        sa.Column('stop_loss', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('take_profit', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('take_profit_method', sa.String(length=60), nullable=True),
        sa.Column('risk_reward', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_ticker_daily_states_ticker'), 'ticker_daily_states', ['ticker'], unique=False)
    op.create_index(
        'ix_ticker_daily_states_region_date', 'ticker_daily_states', ['region', 'trade_date'], unique=False
    )
    op.create_unique_constraint(
        'uq_ticker_daily_state_region_ticker_date', 'ticker_daily_states', ['region', 'ticker', 'trade_date']
    )

    op.create_table(
        'ticker_intraday_states',
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('region', sa.String(length=20), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('entry_already_triggered', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('ticker'),
    )

    op.create_table(
        'position_daily_states',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
        sa.Column('urgency', sa.String(length=20), nullable=False),
        sa.Column('reasons', sa.JSON(), nullable=False),
        sa.Column('price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('r_multiple', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('current_stop', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('engine_version', sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_position_daily_states_ticker'), 'position_daily_states', ['ticker'], unique=False)
    op.create_index(
        'ix_position_daily_states_portfolio_date',
        'position_daily_states',
        ['portfolio_id', 'trade_date'],
        unique=False,
    )
    op.create_unique_constraint(
        'uq_position_daily_state_portfolio_ticker_date',
        'position_daily_states',
        ['portfolio_id', 'ticker', 'trade_date'],
    )

    op.create_table(
        'trigger_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entity_type', sa.String(length=20), nullable=False),
        sa.Column('entity_key', sa.String(length=60), nullable=False),
        sa.Column('event_type', sa.String(length=40), nullable=False),
        sa.Column('previous_value', sa.String(length=60), nullable=True),
        sa.Column('new_value', sa.String(length=60), nullable=True),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_trigger_events_entity', 'trigger_events', ['entity_type', 'entity_key'], unique=False
    )
    op.create_index(op.f('ix_trigger_events_occurred_at'), 'trigger_events', ['occurred_at'], unique=False)

    op.create_table(
        'daily_briefs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('brief_date', sa.Date(), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
        sa.Column('positions_needing_action', sa.Integer(), nullable=False),
        sa.Column('new_entry_triggers', sa.Integer(), nullable=False),
        sa.Column('new_gate_passes', sa.Integer(), nullable=False),
        sa.Column('headline', sa.String(length=500), nullable=False),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_daily_briefs_brief_date'), 'daily_briefs', ['brief_date'], unique=False)
    op.create_unique_constraint('uq_daily_brief_portfolio_date', 'daily_briefs', ['portfolio_id', 'brief_date'])


def downgrade() -> None:
    op.drop_table('daily_briefs')
    op.drop_table('trigger_events')
    op.drop_table('position_daily_states')
    op.drop_table('ticker_intraday_states')
    op.drop_table('ticker_daily_states')
    op.drop_table('job_runs')
