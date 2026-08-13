"""Unit tests for migrate.py - the `engine` itself is always replaced with a
fake so these never touch a real database connection, regardless of what
DATABASE_URL happens to resolve to in whatever environment runs this suite."""

from app.infrastructure.db import migrate


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeEngine:
    def connect(self):
        return _FakeConnection()


def test_run_migrations_on_startup_calls_alembic_upgrade_head(monkeypatch):
    calls = []
    monkeypatch.setattr(migrate, "engine", _FakeEngine())
    monkeypatch.setattr(migrate.command, "upgrade", lambda cfg, target: calls.append((cfg, target)))
    # Bypass the pytest guard just for this one call, to exercise the real body.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    migrate.run_migrations_on_startup()

    assert len(calls) == 1
    _, target = calls[0]
    assert target == "head"


def test_run_migrations_on_startup_swallows_failures(monkeypatch):
    """A migration failure (bad connection, permissions, anything) must never
    propagate - the whole point is that the app still starts and serves
    traffic against whatever schema is already live."""

    def boom(cfg, target):
        raise RuntimeError("simulated connection failure")

    monkeypatch.setattr(migrate, "engine", _FakeEngine())
    monkeypatch.setattr(migrate.command, "upgrade", boom)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    migrate.run_migrations_on_startup()  # must not raise


def test_run_migrations_on_startup_skipped_under_pytest(monkeypatch):
    calls = []
    monkeypatch.setattr(migrate, "engine", _FakeEngine())
    monkeypatch.setattr(migrate.command, "upgrade", lambda cfg, target: calls.append(1))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some::test")

    migrate.run_migrations_on_startup()

    assert calls == []
