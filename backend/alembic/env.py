from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.infrastructure.db import models  # noqa: F401 - ensures models are registered on Base.metadata
from app.infrastructure.db.session import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # `migrate.py` (run automatically at application startup) passes in the
    # app's own already-connected engine's connection via this attribute, so
    # migrations run through the exact same connection configuration used for
    # everything else (notably `prepare_threshold=None`, required for
    # Supabase's Supavisor transaction-mode pooler - see session.py) instead
    # of a second, separately-configured engine built fresh from the URL
    # alone. The CLI (`alembic upgrade head`) has no such connection to pass
    # in, so it still falls back to building its own here exactly as before.
    connectable = config.attributes.get("connection")
    if connectable is not None:
        context.configure(connection=connectable, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
        return

    connectable = engine_from_config(config.get_section(config.config_ini_section, {}), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
