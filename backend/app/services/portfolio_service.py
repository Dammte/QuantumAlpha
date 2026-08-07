from datetime import date

import pandas as pd

from app.domain.models.portfolio import Portfolio
from app.domain.models.transaction import Transaction, TransactionType
from app.infrastructure.db.repositories.portfolio_repository import PortfolioRepository
from app.services.market_data_service import MarketDataService
from app.services.metrics_service import (
    PortfolioMetrics,
    compute_portfolio_metrics_from_returns,
    daily_returns,
    time_weighted_returns,
)


class PortfolioNotFoundError(Exception):
    pass


class PortfolioService:
    def __init__(self, repository: PortfolioRepository, market_data: MarketDataService) -> None:
        self.repository = repository
        self.market_data = market_data

    def get_portfolio_summary(self, portfolio_id: int) -> Portfolio:
        orm = self.repository.get(portfolio_id)
        if orm is None:
            raise PortfolioNotFoundError(f"Portfolio {portfolio_id} not found")
        base_currency = orm.base_currency

        transactions = self.repository.get_transactions(portfolio_id)
        tickers = sorted({tx.ticker for tx in transactions})
        price_quotes = self.market_data.get_latest_quotes(tickers) if tickers else {}

        # One FX lookup per distinct *currency* actually held (not per ticker),
        # and none at all for an all-base-currency portfolio (the common case) -
        # base_currency itself is pre-seeded at 1.0 so it never needs a fetch.
        fx_rates = {base_currency: 1.0}
        for currency in {quote.currency for quote in price_quotes.values()}:
            if currency in fx_rates:
                continue
            rate = self.market_data.get_fx_rate(currency, base_currency)
            if rate is not None:
                fx_rates[currency] = rate

        portfolio = self.repository.build_portfolio(portfolio_id, price_quotes, fx_rates)
        if portfolio is None:
            raise PortfolioNotFoundError(f"Portfolio {portfolio_id} not found")
        return portfolio

    def _held_quantity_series(
        self, transactions: list[Transaction], ticker: str, dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Cumulative quantity held of `ticker` as of each date in `dates`.

        A transaction counts from its own calendar day onward (matched on date, not
        time-of-day) so this stays aligned with `_cash_flow_series`, which buckets the
        same transactions by calendar day too.
        """
        ticker_txs = sorted((tx for tx in transactions if tx.ticker == ticker), key=lambda t: t.executed_at)
        quantity = 0.0
        tx_idx = 0
        held = []
        for as_of in dates:
            while (
                tx_idx < len(ticker_txs)
                and pd.Timestamp(ticker_txs[tx_idx].executed_at).normalize() <= as_of
            ):
                tx = ticker_txs[tx_idx]
                quantity += tx.quantity if tx.transaction_type == TransactionType.BUY else -tx.quantity
                tx_idx += 1
            held.append(quantity)
        return pd.Series(held, index=dates)

    def _cash_flow_series(self, transactions: list[Transaction], dates: pd.DatetimeIndex) -> pd.Series:
        """Net external money added to the portfolio on each date (buys positive, sells negative).

        This has no separate cash account to draw from, so every buy is treated as fresh
        capital coming in and every sell as capital leaving, rather than as investment
        performance. See `time_weighted_returns` for why that distinction matters.
        """
        flows = pd.Series(0.0, index=dates)
        for tx in transactions:
            tx_date = pd.Timestamp(tx.executed_at).normalize()
            if tx_date in flows.index:
                flows.loc[tx_date] += tx.total_cost
        return flows

    def get_portfolio_value_history(self, portfolio_id: int, start: date, end: date) -> pd.Series:
        transactions = self.repository.get_transactions(portfolio_id)
        tickers = sorted({tx.ticker for tx in transactions})
        if not tickers:
            return pd.Series(dtype=float)

        price_series = {t: self.market_data.get_close_price_series(t, start, end) for t in tickers}
        all_dates = sorted(set().union(*(s.index for s in price_series.values() if not s.empty)))
        if not all_dates:
            return pd.Series(dtype=float)
        date_index = pd.DatetimeIndex(all_dates)

        portfolio_value = pd.Series(0.0, index=date_index)
        for ticker in tickers:
            prices = price_series[ticker].reindex(date_index).ffill().fillna(0.0)
            quantity_held = self._held_quantity_series(transactions, ticker, date_index)
            portfolio_value = portfolio_value.add(prices * quantity_held, fill_value=0.0)

        # Drop the leading stretch where the portfolio is worth nothing (before the first
        # transaction settles): a 0 -> positive jump there would otherwise show up as an
        # infinite one-day return and poison every metric that follows.
        nonzero = portfolio_value.to_numpy().nonzero()[0]
        if len(nonzero) == 0:
            return pd.Series(dtype=float)
        portfolio_value = portfolio_value.iloc[nonzero[0] :]

        return portfolio_value

    def get_portfolio_metrics(
        self,
        portfolio_id: int,
        start: date,
        end: date,
        risk_free_rate: float = 0.0,
        benchmark_ticker: str | None = None,
    ) -> PortfolioMetrics:
        value_series = self.get_portfolio_value_history(portfolio_id, start, end)
        if value_series.empty:
            raise PortfolioNotFoundError(f"No price history available for portfolio {portfolio_id}")

        transactions = self.repository.get_transactions(portfolio_id)
        cash_flows = self._cash_flow_series(transactions, value_series.index)
        returns = time_weighted_returns(value_series, cash_flows)

        benchmark_returns = None
        if benchmark_ticker:
            benchmark_prices = self.market_data.get_close_price_series(benchmark_ticker, start, end)
            benchmark_returns = daily_returns(benchmark_prices)

        return compute_portfolio_metrics_from_returns(returns, risk_free_rate, benchmark_returns)

    def get_portfolio_history(
        self,
        portfolio_id: int,
        start: date,
        end: date,
        benchmark_ticker: str | None = None,
    ) -> tuple[pd.Series, pd.Series | None]:
        """Value series for the evolution chart, plus an optional raw benchmark price
        series (the frontend indexes both to a common base so they can share one axis)."""
        value_series = self.get_portfolio_value_history(portfolio_id, start, end)
        if value_series.empty:
            raise PortfolioNotFoundError(f"No price history available for portfolio {portfolio_id}")

        benchmark_series = None
        if benchmark_ticker:
            benchmark_series = self.market_data.get_close_price_series(benchmark_ticker, start, end)
            benchmark_series = benchmark_series.reindex(value_series.index).ffill().dropna()

        return value_series, benchmark_series
