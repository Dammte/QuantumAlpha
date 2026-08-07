from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    ticker: str
    quantity: float
    average_cost: float
    current_price: float | None  # None when a live quote couldn't be fetched right now
    previous_close: float | None = None
    currency: str = "USD"
    realized_pnl: float = 0.0  # banked from partial sells of this ticker while still holding some
    fx_rate_to_base: float | None = 1.0  # multiplies a `currency` amount into the portfolio's base currency

    # --- native-currency figures: what you'd see quoting this ticker directly ---

    @property
    def market_value(self) -> float | None:
        return self.quantity * self.current_price if self.current_price is not None else None

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.average_cost

    @property
    def unrealized_pnl(self) -> float | None:
        market_value = self.market_value
        return market_value - self.cost_basis if market_value is not None else None

    @property
    def unrealized_pnl_pct(self) -> float | None:
        if self.cost_basis == 0:
            return None
        unrealized = self.unrealized_pnl
        return unrealized / self.cost_basis if unrealized is not None else None

    @property
    def day_change(self) -> float | None:
        """Today's move on this position in its own currency (quantity x price
        delta) - None when either side of the comparison is unavailable."""
        if self.current_price is None or self.previous_close is None:
            return None
        return self.quantity * (self.current_price - self.previous_close)

    @property
    def day_change_pct(self) -> float | None:
        if self.current_price is None or self.previous_close is None or self.previous_close == 0:
            return None
        return (self.current_price - self.previous_close) / self.previous_close

    # --- base-currency figures: what this position contributes to portfolio-wide
    # totals, so a USD portfolio holding a EUR stock doesn't just add euros and
    # dollars together as if they were the same unit ---

    @property
    def market_value_base(self) -> float | None:
        market_value = self.market_value
        if market_value is None or self.fx_rate_to_base is None:
            return None
        return market_value * self.fx_rate_to_base

    @property
    def cost_basis_base(self) -> float | None:
        return self.cost_basis * self.fx_rate_to_base if self.fx_rate_to_base is not None else None

    @property
    def unrealized_pnl_base(self) -> float | None:
        unrealized = self.unrealized_pnl
        if unrealized is None or self.fx_rate_to_base is None:
            return None
        return unrealized * self.fx_rate_to_base

    @property
    def day_change_base(self) -> float | None:
        change = self.day_change
        if change is None or self.fx_rate_to_base is None:
            return None
        return change * self.fx_rate_to_base
