from abc import ABC, abstractmethod

from app.domain.models.daily_brief import DailyBrief


class DailyBriefRepositoryPort(ABC):
    """Port for the "Hoy" dashboard's precomputed daily summary - see
    `DailyBrief`'s docstring."""

    @abstractmethod
    def upsert(self, brief: DailyBrief) -> DailyBrief:
        ...

    @abstractmethod
    def latest_for_portfolio(self, portfolio_id: int) -> DailyBrief | None:
        ...
