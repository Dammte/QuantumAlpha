from dataclasses import dataclass, field

from app.domain.models.position import Position


@dataclass(slots=True)
class Portfolio:
    id: int | None
    name: str
    base_currency: str
    positions: list[Position] = field(default_factory=list)
    total_realized_pnl: float = 0.0  # banked from every sell ever made, including fully-closed positions,
    # already converted to base_currency at build time (see PortfolioRepository.build_portfolio)
    cash_balance: float = 0.0  # deposits + sale proceeds - withdrawals - purchase cost, in base_currency -
    # see PortfolioRepository.build_portfolio's docstring for why this exists

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value_base for p in self.positions if p.market_value_base is not None)

    @property
    def total_portfolio_value(self) -> float:
        """The actual answer to "what is my portfolio worth right now": open
        positions *plus* uninvested cash (sale proceeds not yet redeployed,
        deposits not yet invested). `total_market_value` alone silently drops
        that cash, which is exactly the bug this field exists to fix - see
        PortfolioRepository.build_portfolio."""
        return self.total_market_value + self.cash_balance

    @property
    def total_cost_basis(self) -> float:
        return sum(p.cost_basis_base for p in self.positions if p.cost_basis_base is not None)

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl_base for p in self.positions if p.unrealized_pnl_base is not None)

    @property
    def total_pnl(self) -> float:
        """The portfolio's true all-time accumulated gain: realized (banked from
        past sells, including positions closed out entirely) plus unrealized (on
        whatever's still held) - not just what today's open positions show, which
        would silently drop any profit already taken off the table."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def total_day_change(self) -> float:
        """Sum of each position's today-vs-yesterday move, in base currency. A
        position whose previous close (or FX rate) couldn't be fetched simply
        doesn't contribute (rather than blanking out the whole figure) - a
        graceful, if slightly conservative, degradation."""
        return sum(p.day_change_base for p in self.positions if p.day_change_base is not None)

    @property
    def total_day_change_pct(self) -> float | None:
        starting_value = self.total_market_value - self.total_day_change
        if starting_value <= 0:
            return None
        return self.total_day_change / starting_value

    def weight_of(self, ticker: str) -> float:
        if self.total_market_value == 0:
            return 0.0
        position = next((p for p in self.positions if p.ticker == ticker), None)
        if position is None or position.market_value_base is None:
            return 0.0
        return position.market_value_base / self.total_market_value
