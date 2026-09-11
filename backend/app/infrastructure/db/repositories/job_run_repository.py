from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.interfaces.job_run_repository import JobRunRepositoryPort
from app.domain.models.job_run import JobRun
from app.infrastructure.db.models import JobRunORM


def _to_domain(orm: JobRunORM) -> JobRun:
    return JobRun(
        id=orm.id,
        job_name=orm.job_name,
        started_at=orm.started_at,
        finished_at=orm.finished_at,
        status=orm.status,
        rows_processed=orm.rows_processed,
        error_message=orm.error_message,
    )


class JobRunRepository(JobRunRepositoryPort):
    def __init__(self, db: Session) -> None:
        self.db = db

    def start(self, job_name: str) -> JobRun:
        orm = JobRunORM(
            job_name=job_name,
            started_at=datetime.now(UTC),
            finished_at=None,
            status="running",
            rows_processed=0,
            error_message=None,
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def finish(self, job_run_id: int, status: str, rows_processed: int, error_message: str | None) -> JobRun:
        orm = self.db.get(JobRunORM, job_run_id)
        if orm is None:
            raise ValueError(f"No existe JobRun con id={job_run_id}")
        orm.finished_at = datetime.now(UTC)
        orm.status = status
        orm.rows_processed = rows_processed
        orm.error_message = error_message
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain(orm)

    def latest(self, job_name: str) -> JobRun | None:
        stmt = (
            select(JobRunORM)
            .where(JobRunORM.job_name == job_name)
            .order_by(JobRunORM.started_at.desc())
            .limit(1)
        )
        orm = self.db.scalars(stmt).first()
        return _to_domain(orm) if orm is not None else None
