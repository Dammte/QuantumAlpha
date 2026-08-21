from abc import ABC, abstractmethod
from datetime import date

from app.domain.models.universe_membership import UniverseMember


class UniverseMembershipRepositoryPort(ABC):
    """Port for the point-in-time universe membership table (D14 - see
    `UniverseMember`'s docstring). One full snapshot per (region, as_of_date)
    - never overwritten by a later refresh, so real point-in-time history
    accumulates over successive monthly runs instead of only ever reflecting
    "today"."""

    @abstractmethod
    def save_snapshot(self, region: str, as_of_date: date, source: str, members: list[UniverseMember]) -> None:
        """Persists one full snapshot. Idempotent per (region, as_of_date) -
        re-running the same month's refresh (a retry after a partial
        failure) replaces that exact snapshot rather than duplicating it;
        an *older* month's snapshot is never touched."""
        ...

    @abstractmethod
    def latest_as_of_date(self, region: str) -> date | None:
        """The most recent snapshot date on file for this region - `None` if
        this region has never been refreshed yet. Used to decide whether a
        monthly refresh is due."""
        ...

    @abstractmethod
    def members_as_of(self, region: str, as_of_date: date | None = None) -> list[UniverseMember]:
        """Every member of the snapshot at or immediately before `as_of_date`
        (the latest one on file when `as_of_date` is `None`) - empty if
        nothing is on file yet for this region, which the caller reads as
        "fall back to the curated universe", never as "the universe is
        empty"."""
        ...

    @abstractmethod
    def all_as_of_dates(self, region: str) -> list[date]:
        """Every distinct snapshot date on file for this region, oldest
        first - what a point-in-time consumer (the ablation study) iterates
        over to reconstruct the universe as it stood at each of its own
        sample dates."""
        ...
