import logging
from datetime import UTC, datetime, timedelta

from app.services import durable_cache as dc


class _RaisingRepo:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def get(self, cache_key):
        raise self.exc

    def set(self, cache_key, payload, computed_at):
        raise self.exc


class _StaticRepo:
    """Always returns the same (payload, computed_at) pair, regardless of key."""

    def __init__(self, payload, computed_at) -> None:
        self.payload = payload
        self.computed_at = computed_at

    def get(self, cache_key):
        return self.payload, self.computed_at


class _FakeSession:
    def __init__(self) -> None:
        self.rolled_back = False

    def rollback(self) -> None:
        self.rolled_back = True


def _patch_repo(monkeypatch, exc: Exception) -> None:
    monkeypatch.setattr(dc, "ComputationCacheRepository", lambda db: _RaisingRepo(exc))


def test_load_fresh_as_returns_reconstructed_value_on_success(monkeypatch):
    monkeypatch.setattr(dc, "ComputationCacheRepository", lambda db: _StaticRepo({"n": 5}, datetime.now(UTC)))
    result = dc.load_fresh_as(_FakeSession(), "key1", timedelta(hours=1), lambda payload: payload["n"] * 2)
    assert result == 10


def test_load_fresh_as_treats_a_reconstruction_failure_as_a_miss(monkeypatch, caplog):
    """The exact bug this exists to prevent: a cached payload that's the
    *wrong shape* for what the caller expects now (e.g. after a response
    schema gained a field) must come back as None, not raise."""
    repo = _StaticRepo({"unexpected": "shape"}, datetime.now(UTC))
    monkeypatch.setattr(dc, "ComputationCacheRepository", lambda db: repo)

    def reconstruct(payload):
        return payload["n"]  # KeyError - this payload doesn't have that key

    with caplog.at_level(logging.WARNING, logger="app.services.durable_cache"):
        result = dc.load_fresh_as(_FakeSession(), "key1", timedelta(hours=1), reconstruct)

    assert result is None
    assert any("no longer matches" in r.message for r in caplog.records)


def test_load_fresh_as_none_when_nothing_cached(monkeypatch):
    class _EmptyRepo:
        def get(self, cache_key):
            return None

    monkeypatch.setattr(dc, "ComputationCacheRepository", lambda db: _EmptyRepo())
    called = False

    def reconstruct(payload):
        nonlocal called
        called = True
        return payload

    assert dc.load_fresh_as(_FakeSession(), "key1", timedelta(hours=1), reconstruct) is None
    assert not called  # reconstruct is never even attempted when there's nothing cached


def test_missing_table_read_failure_rolls_back_and_returns_none(monkeypatch):
    monkeypatch.setattr(dc, "_warned_missing_table", False)
    _patch_repo(monkeypatch, Exception('relation "computation_cache" does not exist'))
    session = _FakeSession()

    assert dc.load_fresh(session, "key1", timedelta(hours=1)) is None
    assert session.rolled_back


def test_missing_table_read_failure_warns_only_once(monkeypatch, caplog):
    """Otherwise every single cache lookup before the migration is run logs a
    full-traceback error - exactly the log spam this is meant to avoid."""
    monkeypatch.setattr(dc, "_warned_missing_table", False)
    _patch_repo(monkeypatch, Exception('relation "computation_cache" does not exist'))
    session = _FakeSession()

    with caplog.at_level(logging.WARNING, logger="app.services.durable_cache"):
        dc.load_fresh(session, "key1", timedelta(hours=1))
        dc.load_fresh(session, "key2", timedelta(hours=1))
        dc.load_any(session, "key3")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 0


def test_sqlite_missing_table_wording_is_also_recognized(monkeypatch, caplog):
    monkeypatch.setattr(dc, "_warned_missing_table", False)
    _patch_repo(monkeypatch, Exception("no such table: computation_cache"))
    session = _FakeSession()

    with caplog.at_level(logging.WARNING, logger="app.services.durable_cache"):
        dc.load_fresh(session, "key1", timedelta(hours=1))

    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert not any(r.levelno == logging.ERROR for r in caplog.records)


def test_unexpected_read_failure_still_logs_as_an_error(monkeypatch, caplog):
    monkeypatch.setattr(dc, "_warned_missing_table", False)
    _patch_repo(monkeypatch, RuntimeError("connection reset by peer"))
    session = _FakeSession()

    with caplog.at_level(logging.WARNING, logger="app.services.durable_cache"):
        result = dc.load_fresh(session, "key1", timedelta(hours=1))

    assert result is None
    assert session.rolled_back
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_save_rolls_back_and_suppresses_duplicate_missing_table_warning(monkeypatch, caplog):
    monkeypatch.setattr(dc, "_warned_missing_table", True)  # simulate an earlier read already having warned
    _patch_repo(monkeypatch, Exception('relation "computation_cache" does not exist'))
    session = _FakeSession()

    with caplog.at_level(logging.WARNING, logger="app.services.durable_cache"):
        computed_at = dc.save(session, "key1", {"a": 1})

    assert session.rolled_back
    assert computed_at is not None
    assert len(caplog.records) == 0  # already warned once by the read path - no repeat


def test_save_unexpected_failure_still_logs_as_an_error(monkeypatch, caplog):
    monkeypatch.setattr(dc, "_warned_missing_table", False)
    _patch_repo(monkeypatch, RuntimeError("connection reset by peer"))
    session = _FakeSession()

    with caplog.at_level(logging.WARNING, logger="app.services.durable_cache"):
        dc.save(session, "key1", {"a": 1})

    assert session.rolled_back
    assert any(r.levelno == logging.ERROR for r in caplog.records)
