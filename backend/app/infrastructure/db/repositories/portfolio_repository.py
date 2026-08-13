from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models.portfolio import Portfolio
from app.domain.models.position import Position
from app.domain.models.price_quote import PriceQuote
from app.domain.models.transaction import CASH_MOVEMENT_TYPES, Transaction, TransactionType
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
        transaction_type: TransactionType,
        quantity: float,
        ticker: str | None = None,
        price: float | None = None,
        fees: float = 0.0,
        executed_at: datetime | None = None,
    ) -> Transaction:
        # DEPOSIT/WITHDRAWAL are pure cash movements: no ticker, and `price` is
        # always pinned to 1.0 so `quantity` reads directly as a cash amount in
        # the portfolio's own base currency (see Transaction's docstring) -
        # whatever the caller passed for ticker/price is ignored, not merely
        # optional, so a client can never smuggle a priced "cash trade" in.
        if transaction_type in CASH_MOVEMENT_TYPES:
            ticker, price = None, 1.0
        elif ticker is None or price is None:
            raise ValueError("ticker y price son obligatorios para compras y ventas")
        else:
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
        """Derives current positions (weighted-average cost), realized P&L and cash
        balance from the full transaction history. Every sell's proceeds are
        compared against the *average cost at that moment* (not today's) to bank
        its realized gain - `total_realized_pnl` sums this across every sell ever
        made, including ones that fully closed out a position (which then holds 0
        shares and wouldn't otherwise show up anywhere).

        `cash_balance` is what actually answers "where did the money from that
        sale go": a sell's proceeds, a deposit, or a withdrawal all move this
        figure (a buy consumes it) instead of just disappearing once the shares
        they came from are no longer held - the whole portfolio total
        (`Portfolio.total_portfolio_value`) is positions *plus* this, not
        positions alone. Tracked in a single running total in the portfolio's own
        base currency (DEPOSIT/WITHDRAWAL are always entered in base currency
        already; a BUY/SELL in another currency is converted with *today's* FX
        rate, same simplification `total_realized_pnl` already makes - see
        `fx_rate_for` below) rather than as separate per-currency pots: this app
        already presents one converted, base-currency portfolio total everywhere
        else, and a broker that lets you buy foreign-listed stock from one
        account is doing exactly this conversion for you under the hood anyway.

        **Floored at 0 after every single transaction, not just at the end** -
        this is load-bearing, not cosmetic. Nobody is required to log a DEPOSIT
        before their very first BUY (most existing portfolios never will, since
        this concept didn't exist until now), and this app has no concept of
        margin/debt - a real buy always has to be funded by cash that actually
        exists. So whenever running the numbers straight would send the balance
        negative, the only consistent reading is that an un-logged deposit must
        have covered the gap right then, and the balance resets to 0 rather than
        carrying a debt forward. Concretely: buy $1,000 of a stock with no prior
        deposit logged, then sell it for $1,500 - flooring at each step gives a
        cash balance of $1,500 (0 after the buy, +1,500 after the sale), while
        flooring only the final total would wrongly give $500 (-1,000 then
        +1,500), silently treating the original $1,000 purchase as a debt the
        sale had to pay off instead of principal that was simply never logged
        going in. Floored the same way after a WITHDRAWAL for the identical
        reason: a withdrawal this app doesn't have covered cash for.
        See `test_selling_moves_proceeds_into_cash_balance` and
        `test_cash_balance_never_goes_negative_without_a_logged_deposit` in
        `test_portfolios_api.py` for these exact scenarios.

        `fx_rates` (currency -> multiplier into the portfolio's base currency,
        see `PortfolioService.get_portfolio_summary`) lets a position priced in
        a different currency than the portfolio's base still contribute
        correctly to portfolio-wide totals instead of being added as if 1 EUR
        were 1 USD. A currency missing from `fx_rates` (rate unavailable right
        now) makes that position's contribution to the totals unavailable too,
        same graceful-degradation treatment as a missing price quote - and the
        same applies to `cash_balance`: a buy/sell whose currency's rate isn't
        available right now simply doesn't move it, understating rather than
        guessing at the true balance."""
        orm = self.get(portfolio_id)
        if orm is None:
            return None
        fx_rates = fx_rates or {}

        transactions = self.get_transactions(portfolio_id)
        holdings: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # ticker -> [quantity, cost_basis]
        realized_pnl_by_ticker: dict[str, float] = defaultdict(float)

        def fx_rate_for(ticker: str) -> float | None:
            quote = price_quotes.get(ticker)
            return fx_rates.get(quote.currency) if quote is not None else None

        cash_balance = 0.0
        for tx in transactions:
            if tx.transaction_type == TransactionType.DEPOSIT:
                cash_balance = max(0.0, cash_balance + tx.quantity)
                continue
            if tx.transaction_type == TransactionType.WITHDRAWAL:
                cash_balance = max(0.0, cash_balance - tx.quantity)
                continue

            state = holdings[tx.ticker]
            if tx.transaction_type == TransactionType.BUY:
                state[0] += tx.quantity
                state[1] += tx.quantity * tx.price + tx.fees
                if (rate := fx_rate_for(tx.ticker)) is not None:
                    cash_balance = max(0.0, cash_balance - (tx.quantity * tx.price + tx.fees) * rate)
            else:
                avg_cost = state[1] / state[0] if state[0] else 0.0
                cost_of_shares_sold = tx.quantity * avg_cost
                proceeds = tx.quantity * tx.price - tx.fees
                realized_pnl_by_ticker[tx.ticker] += proceeds - cost_of_shares_sold
                state[0] -= tx.quantity
                state[1] -= cost_of_shares_sold
                if (rate := fx_rate_for(tx.ticker)) is not None:
                    cash_balance = max(0.0, cash_balance + proceeds * rate)

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
            cash_balance=cash_balance,
        )
