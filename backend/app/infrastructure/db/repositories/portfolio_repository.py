from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models.portfolio import Portfolio
from app.domain.models.position import Position
from app.domain.models.price_quote import PriceQuote
from app.domain.models.transaction import Transaction, TransactionType
from app.infrastructure.db.models import PortfolioORM, TransactionORM


def _to_domain_transaction(orm: TransactionORM) -> Transaction:
    return Transaction(
        id=orm.id,
        portfolio_id=orm.portfolio_id,
        ticker=orm.ticker,
        transaction_type=orm.transaction_type,
        quantity=float(orm.quantity),
        price=float(orm.price),
        fees=float(orm.fees),
        executed_at=orm.executed_at,
    )


class PortfolioRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, name: str, base_currency: str = "USD") -> PortfolioORM:
        portfolio = PortfolioORM(name=name, base_currency=base_currency)
        self.db.add(portfolio)
        self.db.commit()
        self.db.refresh(portfolio)
        return portfolio

    def get(self, portfolio_id: int) -> PortfolioORM | None:
        return self.db.get(PortfolioORM, portfolio_id)

    def list_all(self) -> list[PortfolioORM]:
        return list(self.db.scalars(select(PortfolioORM)).all())

    def delete(self, portfolio_id: int) -> None:
        portfolio = self.get(portfolio_id)
        if portfolio is not None:
            self.db.delete(portfolio)
            self.db.commit()

    def _currently_held_quantity(self, portfolio_id: int, ticker: str) -> float:
        quantity = 0.0
        for tx in self.get_transactions(portfolio_id):
            if tx.ticker != ticker:
                continue
            quantity += tx.quantity if tx.transaction_type == TransactionType.BUY else -tx.quantity
        return quantity

    def add_transaction(
        self,
        portfolio_id: int,
        ticker: str,
        transaction_type: TransactionType,
        quantity: float,
        price: float,
        fees: float = 0.0,
        executed_at: datetime | None = None,
    ) -> Transaction:
        # Guards against the realized-P&L math going nonsensical (selling from a
        # position that doesn't exist would price the "cost" of those shares at
        # 0, booking the entire sale as pure profit). Compares against shares
        # held right now rather than truly reconstructing history in
        # `executed_at` order, so a backdated sell inserted out of order can
        # still slip through - a reasonable simplification for a personal tool.
        if transaction_type == TransactionType.SELL:
            held = self._currently_held_quantity(portfolio_id, ticker)
            if quantity > held + 1e-9:
                raise ValueError(
                    f"No se pueden vender {quantity:g} unidades de {ticker}: solo hay {held:g} en cartera."
                )

        orm = TransactionORM(
            portfolio_id=portfolio_id,
            ticker=ticker,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            fees=fees,
            executed_at=executed_at or datetime.now(UTC),
        )
        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)
        return _to_domain_transaction(orm)

    def get_transactions(self, portfolio_id: int) -> list[Transaction]:
        stmt = (
            select(TransactionORM)
            .where(TransactionORM.portfolio_id == portfolio_id)
            .order_by(TransactionORM.executed_at)
        )
        return [_to_domain_transaction(orm) for orm in self.db.scalars(stmt).all()]

    def delete_transaction(self, portfolio_id: int, transaction_id: int) -> bool:
        orm = self.db.get(TransactionORM, transaction_id)
        if orm is None or orm.portfolio_id != portfolio_id:
            return False
        self.db.delete(orm)
        self.db.commit()
        return True

    def build_portfolio(
        self,
        portfolio_id: int,
        price_quotes: dict[str, PriceQuote],
        fx_rates: dict[str, float] | None = None,
    ) -> Portfolio | None:
        """Derives current positions (weighted-average cost) and realized P&L from
        the full transaction history. Every sell's proceeds are compared against
        the *average cost at that moment* (not today's) to bank its realized gain
        - `total_realized_pnl` sums this across every sell ever made, including
        ones that fully closed out a position (which then holds 0 shares and
        wouldn't otherwise show up anywhere).

        `fx_rates` (currency -> multiplier into the portfolio's base currency,
        see `PortfolioService.get_portfolio_summary`) lets a position priced in
        a different currency than the portfolio's base still contribute
        correctly to portfolio-wide totals instead of being added as if 1 EUR
        were 1 USD. A currency missing from `fx_rates` (rate unavailable right
        now) makes that position's contribution to the totals unavailable too,
        same graceful-degradation treatment as a missing price quote."""
        orm = self.get(portfolio_id)
        if orm is None:
            return None
        fx_rates = fx_rates or {}

        transactions = self.get_transactions(portfolio_id)
        holdings: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # ticker -> [quantity, cost_basis]
        realized_pnl_by_ticker: dict[str, float] = defaultdict(float)

        for tx in transactions:
            state = holdings[tx.ticker]
            if tx.transaction_type == TransactionType.BUY:
                state[0] += tx.quantity
                state[1] += tx.quantity * tx.price + tx.fees
            else:
                avg_cost = state[1] / state[0] if state[0] else 0.0
                cost_of_shares_sold = tx.quantity * avg_cost
                proceeds = tx.quantity * tx.price - tx.fees
                realized_pnl_by_ticker[tx.ticker] += proceeds - cost_of_shares_sold
                state[0] -= tx.quantity
                state[1] -= cost_of_shares_sold

        def fx_rate_for(ticker: str) -> float | None:
            quote = price_quotes.get(ticker)
            return fx_rates.get(quote.currency) if quote is not None else None

        positions = [
            Position(
                ticker=ticker,
                quantity=quantity,
                average_cost=(cost_basis / quantity) if quantity else 0.0,
                current_price=price_quotes[ticker].price if ticker in price_quotes else None,
                previous_close=price_quotes[ticker].previous_close if ticker in price_quotes else None,
                currency=price_quotes[ticker].currency if ticker in price_quotes else orm.base_currency,
                realized_pnl=realized_pnl_by_ticker.get(ticker, 0.0),
                fx_rate_to_base=fx_rate_for(ticker),
            )
            for ticker, (quantity, cost_basis) in holdings.items()
            if quantity > 1e-9
        ]

        total_realized_pnl = sum(
            pnl * rate
            for ticker, pnl in realized_pnl_by_ticker.items()
            if (rate := fx_rate_for(ticker)) is not None
        )

        return Portfolio(
            id=orm.id,
            name=orm.name,
            base_currency=orm.base_currency,
            positions=positions,
            total_realized_pnl=total_realized_pnl,
        )
