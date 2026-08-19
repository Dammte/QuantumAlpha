from abc import ABC, abstractmethod
from datetime import date

from app.domain.models.trade_plan import TradePlan


class TradePlanRepositoryPort(ABC):
    """Port for persisting trade plans (see `TradePlan`/`TradePlanORM`).

    Keeping this abstract means `trade_plan_service.py`/`portfolio_risk_service.py`
    (the `services/` layer) never import SQLAlchemy directly, matching the
    same domain/services-stay-framework-free separation `MarketDataProvider`
    already enforces for market data - see this module's sibling.
    """

    @abstractmethod
    def get_open(self, portfolio_id: int, ticker: str) -> TradePlan | None:
        """The currently-open lot's plan for this (portfolio, ticker), if
        any - `None` when nothing is held or no plan has been created yet."""
        ...

    @abstractmethod
    def create(
        self,
        portfolio_id: int,
        ticker: str,
        entry_price: float,
        entry_date: date,
        initial_stop: float | None,
        initial_target: float | None,
        initial_quantity: float,
        thesis: str,
        engine_version: str,
    ) -> TradePlan:
        ...

    @abstractmethod
    def update_trailing(self, plan_id: int, current_stop: float, highest_close_since_entry: float) -> None:
        ...

    @abstractmethod
    def close(self, portfolio_id: int, ticker: str) -> None:
        """Marks the currently-open plan (if any) as closed - see
        `TradePlanORM`'s docstring for why a ticker can have more than one
        plan/lot over its history."""
        ...
