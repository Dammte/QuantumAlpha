import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'
import { api } from './api'
import { colorScaleFor } from './palette'
import { formatCurrency, formatPercent } from './format'
import { rangeFor } from './timeframes'
import PortfolioSelect from './components/PortfolioSelect'
import PositionsTable from './components/PositionsTable'
import AllocationDonut from './components/AllocationDonut'
import MetricsPanel from './components/MetricsPanel'
import TimeframeTabs from './components/TimeframeTabs'
import PortfolioEvolutionChart from './components/PortfolioEvolutionChart'
import DrawdownChart from './components/DrawdownChart'
import TransactionForm from './components/TransactionForm'
import TransactionsList from './components/TransactionsList'
import MarketView from './components/market/MarketView'
import Sidebar from './components/Sidebar'
import RefreshBar from './components/RefreshBar'
import SwapSuggestions from './components/SwapSuggestions'
import { DEFAULT_REGION } from './regions'

function App() {
  const [view, setView] = useState('portfolio')
  const [marketSection, setMarketSection] = useState('analysis')
  const [presetTicker, setPresetTicker] = useState(null)
  const [region, setRegion] = useState(DEFAULT_REGION)
  const [portfolios, setPortfolios] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [summary, setSummary] = useState(null)
  const [transactions, setTransactions] = useState([])
  const [risk, setRisk] = useState(null)
  const [riskLoading, setRiskLoading] = useState(false)
  const [riskRefreshing, setRiskRefreshing] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [history, setHistory] = useState({ points: [], benchmarkPoints: null })
  const [timeframe, setTimeframe] = useState('6M')
  const [benchmarkTicker, setBenchmarkTicker] = useState('')
  const [compareMode, setCompareMode] = useState(false)
  const [loading, setLoading] = useState(true)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [error, setError] = useState(null)
  const [submittingTx, setSubmittingTx] = useState(false)

  const loadPortfolios = useCallback(async () => {
    const list = await api.listPortfolios()
    setPortfolios(list)
    setSelectedId((current) => current ?? list[0]?.id ?? null)
    return list
  }, [])

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        await loadPortfolios()
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [loadPortfolios])

  const loadSummary = useCallback(async (id) => {
    const [summaryData, txData] = await Promise.all([api.getSummary(id), api.listTransactions(id)])
    setError(null)
    setSummary(summaryData)
    setTransactions(txData)
    // Runs the full quant suite per holding server-side, so it can take a while
    // on a cold cache - never blocks the rest of the dashboard on it, but does
    // clear any stale risk data from a previously selected portfolio and track
    // its own loading state so the UI can show "calculando…" instead of just
    // making the whole column disappear while this is in flight.
    setRisk(null)
    setRiskLoading(true)
    api
      .getPortfolioRisk(id)
      .then(setRisk)
      .catch(() => setRisk(null))
      .finally(() => setRiskLoading(false))
  }, [])

  const loadAnalysis = useCallback(async (id, tf, benchmark) => {
    setAnalysisLoading(true)
    const { start, end } = rangeFor(tf)
    try {
      const [metricsData, historyData] = await Promise.all([
        api.getMetrics(id, { start, end, benchmarkTicker: benchmark || undefined }),
        api.getHistory(id, { start, end, benchmarkTicker: benchmark || undefined }),
      ])
      setMetrics(metricsData)
      setHistory({ points: historyData.points, benchmarkPoints: historyData.benchmark_points })
    } catch {
      // Expected right after creating a portfolio (no transactions yet) or when the
      // selected date range has no price history — the empty states below already
      // communicate that, so this isn't worth a page-level error banner.
      setMetrics(null)
      setHistory({ points: [], benchmarkPoints: null })
    } finally {
      setAnalysisLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!selectedId) return
    async function load() {
      setError(null)
      await loadSummary(selectedId).catch((err) => setError(err.message))
      await loadAnalysis(selectedId, timeframe, benchmarkTicker)
    }
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId])

  useEffect(() => {
    if (!selectedId) return
    async function load() {
      await loadAnalysis(selectedId, timeframe, benchmarkTicker)
    }
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeframe, benchmarkTicker])

  const handleCreatePortfolio = async (name) => {
    const created = await api.createPortfolio(name)
    await loadPortfolios()
    setSelectedId(created.id)
  }

  const refreshAfterMutation = async () => {
    await loadSummary(selectedId)
    await loadAnalysis(selectedId, timeframe, benchmarkTicker)
  }

  const handleRefreshRisk = async () => {
    if (!selectedId) return
    setRiskRefreshing(true)
    try {
      setRisk(await api.getPortfolioRisk(selectedId, { refresh: true }))
    } catch {
      // Keep showing whatever was already on screen rather than blanking it out
      // over a failed manual refresh.
    } finally {
      setRiskRefreshing(false)
    }
  }

  const handleAddTransaction = async (transaction) => {
    setSubmittingTx(true)
    try {
      await api.addTransaction(selectedId, transaction)
      await refreshAfterMutation()
    } finally {
      setSubmittingTx(false)
    }
  }

  const handleDeleteTransaction = async (transactionId) => {
    await api.deleteTransaction(selectedId, transactionId)
    await refreshAfterMutation()
  }

  // Jump straight into a full "Analizar activo" deep dive for a specific
  // ticker from anywhere else in the app (premium watchlist, portfolio
  // positions) instead of duplicating its cards in those views. `key` forces
  // TickerAnalysisPanel's effect to re-fire even for the same ticker twice in
  // a row.
  const navigateToAnalysis = (ticker) => {
    setPresetTicker({ ticker, key: Date.now() })
    setView('market')
    setMarketSection('analysis')
  }

  const colorScale = useMemo(
    () => colorScaleFor(summary?.positions.map((p) => p.ticker) ?? []),
    [summary],
  )

  const riskByTicker = useMemo(
    () => (risk ? new Map(risk.positions.map((p) => [p.ticker, p])) : null),
    [risk],
  )

  const currencyByTicker = useMemo(
    () => new Map(summary?.positions.map((p) => [p.ticker, p.currency]) ?? []),
    [summary],
  )

  const totalPnlIsUp = (summary?.total_pnl ?? 0) >= 0
  const dayChangeIsUp = (summary?.total_day_change ?? 0) >= 0
  const periodIsUp = (metrics?.cumulative_return ?? 0) >= 0
  const toneOf = (isUp) => (isUp ? 'up' : 'down')
  const cashSharePct =
    summary && summary.total_portfolio_value > 0 ? summary.cash_balance / summary.total_portfolio_value : null

  return (
    <div className="app">
      <Sidebar
        view={view}
        marketSection={marketSection}
        region={region}
        onSelectPortfolio={() => setView('portfolio')}
        onSelectMarketSection={(key) => {
          setView('market')
          setMarketSection(key)
        }}
        onRegionChange={setRegion}
      />
      <div className="app-content">
        {view === 'portfolio' && portfolios.length > 0 && (
          <div className="app-content__toolbar">
            <PortfolioSelect
              portfolios={portfolios}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onCreate={handleCreatePortfolio}
            />
          </div>
        )}

        {error && <div className="banner banner--error">{error}</div>}

        {view === 'market' ? (
          <MarketView
            section={marketSection}
            presetTicker={presetTicker}
            onNavigateToTicker={navigateToAnalysis}
            region={region}
          />
        ) : loading ? (
          <p className="empty-state">Cargando…</p>
        ) : portfolios.length === 0 ? (
          <section className="panel">
            <h2>Crea tu primera cartera</h2>
            <p className="empty-state">
              Añade tus posiciones reales (ticker, cantidad, precio y fecha de compra) para ver tu cartera
              analizada con datos de mercado en vivo.
            </p>
            <PortfolioSelect portfolios={[]} selectedId={null} onSelect={() => {}} onCreate={handleCreatePortfolio} />
          </section>
        ) : (
          <main className="dashboard">
          {summary && (
            <section className="hero-row">
              <div className="hero-card hero-card--primary">
                <p className="hero-card__label">Valor total de la cartera</p>
                <p className="hero-card__value">
                  {formatCurrency(summary.total_portfolio_value, summary.base_currency)}
                </p>
                <p className={`hero-card__pct hero-card__pct--${toneOf(dayChangeIsUp)}`}>
                  {dayChangeIsUp ? '▲' : '▼'} {formatCurrency(Math.abs(summary.total_day_change), summary.base_currency)}
                  {summary.total_day_change_pct !== null && (
                    <> ({formatPercent(summary.total_day_change_pct, { signed: true })})</>
                  )}{' '}
                  hoy
                </p>
                <p className="hero-card__hint">
                  Invertido: {formatCurrency(summary.total_market_value, summary.base_currency)}
                </p>
              </div>

              <div className="hero-card">
                <p className="hero-card__label">Liquidez disponible</p>
                <p className="hero-card__value">{formatCurrency(summary.cash_balance, summary.base_currency)}</p>
                <p className="hero-card__hint">
                  {cashSharePct !== null && `${formatPercent(cashSharePct)} de tu cartera total · `}
                  no invertida ahora mismo
                </p>
              </div>

              <div className="hero-card">
                <p className="hero-card__label">Ganancia acumulada total</p>
                <p className={`hero-card__value hero-card__value--${toneOf(totalPnlIsUp)}`}>
                  {formatCurrency(summary.total_pnl, summary.base_currency)}
                  <span className="hero-card__pct">
                    {' '}
                    ({formatPercent(summary.total_cost_basis ? summary.total_pnl / summary.total_cost_basis : 0, { signed: true })})
                  </span>
                </p>
                <p className="hero-card__hint">
                  No realizado: {formatCurrency(summary.total_unrealized_pnl, summary.base_currency)} · Realizado:{' '}
                  {formatCurrency(summary.total_realized_pnl, summary.base_currency)}
                </p>
              </div>

              <div className="hero-card">
                <p className="hero-card__label">Retorno del periodo ({timeframe})</p>
                <p className={`hero-card__value hero-card__value--${toneOf(periodIsUp)}`}>
                  {metrics ? formatPercent(metrics.cumulative_return, { signed: true }) : '—'}
                </p>
              </div>
            </section>
          )}

          <section className="panel evolution-panel">
            <div className="panel__header">
              <h2>Evolución de la cartera</h2>
              <TimeframeTabs value={timeframe} onChange={setTimeframe} />
            </div>
            <label className="filter filter--inline benchmark-input">
              <span>Comparar con</span>
              <input
                type="text"
                placeholder="^GSPC"
                value={benchmarkTicker}
                onChange={(e) => setBenchmarkTicker(e.target.value.toUpperCase())}
              />
            </label>
            {analysisLoading ? (
              <p className="empty-state">Cargando histórico…</p>
            ) : (
              <PortfolioEvolutionChart
                points={history.points}
                benchmarkPoints={history.benchmarkPoints}
                benchmarkTicker={benchmarkTicker}
                currency={summary?.base_currency ?? 'USD'}
                compareMode={compareMode}
                onToggleCompare={() => setCompareMode((v) => !v)}
              />
            )}
          </section>

          <section className="grid-2">
            <div className="panel">
              <h2>Drawdown</h2>
              {analysisLoading ? <p className="empty-state">Calculando…</p> : <DrawdownChart points={history.points} />}
            </div>
            <div className="panel">
              {summary && (
                <AllocationDonut
                  positions={summary.positions}
                  totalMarketValue={summary.total_market_value}
                  cashBalance={summary.cash_balance}
                  currency={summary.base_currency}
                  colorScale={colorScale}
                />
              )}
            </div>
          </section>

          <section className="panel">
            <h2>Posiciones {summary && `(${summary.positions.length})`}</h2>
            {risk && (
              <>
                <RefreshBar computedAt={risk.computed_at} onRefresh={handleRefreshRisk} refreshing={riskRefreshing} />
                <SwapSuggestions suggestions={risk.swap_suggestions} onNavigateToTicker={navigateToAnalysis} />
              </>
            )}
            {summary && (
              <PositionsTable
                positions={summary.positions}
                colorScale={colorScale}
                totalMarketValue={summary.total_market_value}
                riskByTicker={riskByTicker}
                riskLoading={riskLoading}
                onNavigateToTicker={navigateToAnalysis}
              />
            )}
          </section>

          <section className="panel">
            <h2>Métricas de riesgo y rendimiento</h2>
            {analysisLoading ? <p className="empty-state">Calculando métricas…</p> : <MetricsPanel metrics={metrics} />}
          </section>

          <section className="panel">
            <h2>Transacciones</h2>
            <TransactionForm
              onSubmit={handleAddTransaction}
              submitting={submittingTx}
              baseCurrency={summary?.base_currency ?? 'USD'}
            />
            <div className="transactions-list">
              <TransactionsList
                transactions={transactions}
                currency={summary?.base_currency ?? 'USD'}
                currencyByTicker={currencyByTicker}
                onDelete={handleDeleteTransaction}
              />
            </div>
          </section>
        </main>
        )}
      </div>
    </div>
  )
}

export default App
