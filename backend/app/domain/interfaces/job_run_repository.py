from abc import ABC, abstractmethod

from app.domain.models.job_run import JobRun


class JobRunRepositoryPort(ABC):
    """Port for recording Fase 2 cron-job executions - see `JobRun`'s
    docstring. Same domain/services-stay-framework-free separation as every
    other port in this package."""

    @abstractmethod
    def start(self, job_name: str) -> JobRun:
        """Creates a "running" row - called at the very top of a job, before
        any real work, so a hard crash mid-run still leaves a row behind
        (status stuck at "running") rather than no record at all."""
        ...

    @abstractmethod
    def finish(
        self,
        job_run_id: int,
        status: str,
        rows_processed: int,
        error_message: str | None,
        detail: dict | None = None,
    ) -> JobRun:
        ...

    @abstractmethod
    def latest(self, job_name: str) -> JobRun | None:
        """The most recent run of one named job - what a "system health" read
        (GET /system/...) checks to answer "did last night's job actually
        run"."""
        ...
