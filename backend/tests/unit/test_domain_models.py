from app.domain.models.portfolio import Portfolio
from app.domain.models.position import Position


def test_position_unrealized_pnl() -> None:
    position = Position(ticker="AAPL", quantity=10, average_cost=100, current_price=120)
    assert position.market_value == 1200
    assert position.cost_basis == 1000
    assert position.unrealized_pnl == 200
    assert position.unrealized_pnl_pct == 0.2


def test_position_unrealized_pnl_with_zero_cost_basis_is_undefined() -> None:
    # 0/0 is undefined, not "0% return" - None distinguishes "no meaningful
    # percentage" from a position that's actually flat.
    position = Position(ticker="AAPL", quantity=0, average_cost=0, current_price=120)
    assert position.unrealized_pnl_pct is None


def test_position_with_no_live_quote_reports_none_not_zero() -> None:
    """A ticker whose price couldn't be fetched (delisted, mistyped, provider
    hiccup) must never look like a real $0 position - every price-derived field
    reports None so the frontend can show "unavailable" instead of a misleading
    wipeout, and the position is excluded from portfolio-wide sums."""
    position = Position(ticker="GHOST", quantity=10, average_cost=50, current_price=None)
    assert position.market_value is None
    assert position.unrealized_pnl is None
    assert position.unrealized_pnl_pct is None
    assert position.day_change is None
    assert position.day_change_pct is None
    assert position.cost_basis == 500  # cost basis is always known - it's our own data


def test_position_day_change() -> None:
    position = Position(ticker="AAPL", quantity=10, average_cost=100, current_price=105, previous_close=100)
    assert position.day_change == 50
    assert position.day_change_pct == 0.05


def test_position_day_change_none_without_previous_close() -> None:
    position = Position(ticker="AAPL", quantity=10, average_cost=100, current_price=105, previous_close=None)
    assert position.day_change is None
    assert position.day_change_pct is None


def test_portfolio_aggregates_positions() -> None:
    portfolio = Portfolio(
        id=1,
        name="Main",
        base_currency="USD",
        positions=[
            Position(ticker="AAPL", quantity=10, average_cost=100, current_price=120),
            Position(ticker="MSFT", quantity=5, average_cost=200, current_price=180),
        ],
    )
    assert portfolio.total_market_value == 1200 + 900
    assert portfolio.total_cost_basis == 1000 + 1000
    assert portfolio.total_unrealized_pnl == portfolio.total_market_value - portfolio.total_cost_basis


def test_portfolio_aggregates_skip_positions_with_no_live_quote() -> None:
    portfolio = Portfolio(
        id=1,
        name="Main",
        base_currency="USD",
        positions=[
            Position(ticker="AAPL", quantity=10, average_cost=100, current_price=120),
            Position(ticker="GHOST", quantity=10, average_cost=50, current_price=None),
        ],
    )
    # GHOST contributes nothing to market value/unrealized P&L rather than
    # crashing the whole aggregate or being silently treated as worth $0.
    assert portfolio.total_market_value == 1200
    assert portfolio.total_unrealized_pnl == 200


def test_portfolio_total_pnl_combines_realized_and_unrealized() -> None:
    portfolio = Portfolio(
        id=1,
        name="Main",
        base_currency="USD",
        positions=[Position(ticker="AAPL", quantity=10, average_cost=100, current_price=120)],
        total_realized_pnl=300.0,
    )
    assert portfolio.total_unrealized_pnl == 200
    assert portfolio.total_pnl == 500.0  # 300 realized (past sells) + 200 unrealized (still held)


def test_portfolio_day_change_sums_positions() -> None:
    portfolio = Portfolio(
        id=1,
        name="Main",
        base_currency="USD",
        positions=[
            Position(ticker="AAPL", quantity=10, average_cost=100, current_price=105, previous_close=100),
            Position(ticker="MSFT", quantity=5, average_cost=200, current_price=190, previous_close=200),
        ],
    )
    assert portfolio.total_day_change == 50 + (-50)
    # starting value = today's value - today's change = yesterday's value
    assert portfolio.total_day_change_pct == 0.0


def test_portfolio_weight_of_ticker() -> None:
    portfolio = Portfolio(
        id=1,
        name="Main",
        base_currency="USD",
        positions=[
            Position(ticker="AAPL", quantity=10, average_cost=100, current_price=100),
            Position(ticker="MSFT", quantity=10, average_cost=100, current_price=100),
        ],
    )
    assert portfolio.weight_of("AAPL") == 0.5
    assert portfolio.weight_of("UNKNOWN") == 0.0
