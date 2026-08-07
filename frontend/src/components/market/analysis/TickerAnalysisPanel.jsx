import { useEffect, useState } from 'react'
import { api } from '../../../api'
import { formatCurrency, formatPercent, formatRatio } from '../../../format'
import StatTile from '../../StatTile'
import TrendBadge from '../TrendBadge'
import PriceChart from './PriceChart'
import VolumeChart from './VolumeChart'
import RsiMacdChart from './RsiMacdChart'
import RecommendationCard from './RecommendationCard'
import FundamentalsCard from './FundamentalsCard'
import HoldersCard from './HoldersCard'
import NewsList from './NewsList'
import SeasonalityChart from './SeasonalityChart'
import HistoricalAnalogsCard from './HistoricalAnalogsCard'
import MarkovChainCard from './MarkovChainCard'
import VolatilityCard from './VolatilityCard'
import MonteCarloChart from './MonteCarloChart'
import BacktestCard from './BacktestCard'
import PositionSizingCard from './PositionSizingCard'

const STAGE_LABELS = { stage1: 'Fase 1 (base)', stage2: 'Fase 2 (avance)', stage3: 'Fase 3 (techo)', stage4: 'Fase 4 (declive)' }
const HORIZON_OPTIONS = [
  { value: '1m', label: '1 mes' },
  { value: '3m', label: '3 meses' },
  { value: '6m', label: '6 meses' },
]
const TABS = [
  { key: 'summary', label: 'Resumen' },
  { key: 'quant', label: 'Cuantitativo' },
  { key: 'charts', label: 'Gráficos' },
  { key: 'fundamentals', label: 'Fundamentales' },
  { key: 'seasonality', label: 'Estacionalidad' },
]

function TickerAnalysisPanel({ presetTicker } = {}) {
  const [ticker, setTicker] = useState('')
  const [horizon, setHorizon] = useState('3m')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('summary')

  const search = async (value, selectedHorizon = horizon) => {
    const symbol = value.trim().toUpperCase()
    if (!symbol) return
    setTicker(symbol)
    setLoading(true)
    setError(null)
    try {
      setAnalysis(await api.getTickerAnalysis(symbol, { horizon: selectedHorizon }))
      setTab('summary')
    } catch (err) {
      setError(err.message)
      setAnalysis(null)
    } finally {
      setLoading(false)
    }
  }

  // Lets other views ("A revisar" premium list, portfolio positions) jump
  // straight into a full deep-dive for a specific ticker instead of duplicating
  // this whole panel's cards elsewhere. `presetTicker.key` changes on every
  // navigation request (even re-clicking the same ticker), so the effect fires
  // again even when `presetTicker.ticker` is unchanged.
  useEffect(() => {
    if (!presetTicker?.ticker) return
    // Fetching in response to a prop change is exactly the sanctioned exception
    // to this rule (see the "Fetching data" example in the React docs this rule
    // links to) - there's no external-system subscription to synchronize here,
    // just a one-shot search triggered by another view's navigation request.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    search(presetTicker.ticker)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetTicker?.key])

  const handleSubmit = (e) => {
    e.preventDefault()
    search(ticker)
  }

  const handleClear = () => {
    setTicker('')
    setAnalysis(null)
    setError(null)
  }

  const handleHorizonChange = (value) => {
    setHorizon(value)
    if (analysis) search(analysis.ticker, value)
  }

  const nearestSupport = analysis?.support_resistance.filter((l) => l.kind === 'support').sort((a, b) => Math.abs(a.distance_pct) - Math.abs(b.distance_pct))[0]
  const nearestResistance = analysis?.support_resistance.filter((l) => l.kind === 'resistance').sort((a, b) => Math.abs(a.distance_pct) - Math.abs(b.distance_pct))[0]

  return (
    <div>
      <form className="filters-row analysis-search" onSubmit={handleSubmit}>
        <label className="filter">
          <span>Analizar un activo</span>
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="AAPL, NVDA, ^GSPC…"
            autoFocus
          />
        </label>
        <button type="submit" className="button-primary" disabled={loading} style={{ alignSelf: 'flex-end' }}>
          {loading ? 'Analizando…' : 'Analizar'}
        </button>
        {(ticker || analysis || error) && (
          <button
            type="button"
            className="button-secondary"
            onClick={handleClear}
            disabled={loading}
            style={{ alignSelf: 'flex-end' }}
          >
            Limpiar
          </button>
        )}
      </form>

      {error && <div className="banner banner--error">{error}</div>}

      {!analysis && !loading && (
        <p className="empty-state">
          Escribe un ticker para ver su análisis cuantitativo completo: gráfico con Bollinger/Gann/soportes, RSI,
          MACD, fundamentales, noticias, estacionalidad, análogos históricos, cadena de Markov, volatilidad GARCH,
          simulación Monte Carlo, validación histórica del sistema (backtest) y una recomendación con stop-loss,
          objetivo y tamaño de posición sugerido (Kelly).
        </p>
      )}

      {analysis && (
        <div className="ticker-analysis">
          <div className="ticker-analysis__header">
            <div>
              <h3>
                {analysis.ticker} <span className="ticker-analysis__name">{analysis.name}</span>
              </h3>
              <p className="ticker-analysis__meta">
                {analysis.sector} {analysis.industry ? `· ${analysis.industry}` : ''}
              </p>
            </div>
            <div className="ticker-analysis__price">
              <span className="ticker-analysis__price-value">{formatCurrency(analysis.price, analysis.currency ?? 'USD')}</span>
              <span className={analysis.change_1d >= 0 ? 'delta-up' : 'delta-down'}>
                {formatPercent(analysis.change_1d, { signed: true })} hoy
              </span>
            </div>
          </div>

          <div className="stat-grid" style={{ marginBottom: 20 }}>
            <StatTile label="Tendencia" value={<TrendBadge trend={analysis.trend} />} />
            <StatTile label="Fase (Weinstein)" value={analysis.stage ? STAGE_LABELS[analysis.stage] : '—'} />
            <StatTile label="RS Rating" value={analysis.rs_rating ?? '—'} hint={analysis.rs_rating === null ? 'fuera de nuestro universo curado' : undefined} />
            <StatTile label="Mansfield RS" value={analysis.mansfield_rs !== null ? formatRatio(analysis.mansfield_rs) : '—'} />
            <StatTile label="RSI (14)" value={analysis.rsi14 !== null ? analysis.rsi14.toFixed(1) : '—'} />
            <StatTile label="ADX (14)" value={analysis.adx14 !== null ? analysis.adx14.toFixed(1) : '—'} />
            <StatTile label="Volumen relativo" value={analysis.relative_volume !== null ? `${analysis.relative_volume.toFixed(2)}x` : '—'} />
            <StatTile label="Minervini" value={`${analysis.minervini_score}/8`} tone={analysis.minervini_pass ? 'up' : 'neutral'} />
          </div>

          <div className="sub-toggle" role="tablist" aria-label="Sección del análisis">
            {TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                role="tab"
                aria-selected={tab === t.key}
                className={`sub-toggle__item ${tab === t.key ? 'sub-toggle__item--active' : ''}`}
                onClick={() => setTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {tab === 'summary' && (
            <>
              <section className="panel panel--nested">
                <h3>Recomendación</h3>
                <RecommendationCard recommendation={analysis.recommendation} currency={analysis.currency ?? 'USD'} />
                <PositionSizingCard
                  sizing={analysis.position_sizing}
                  currency={analysis.currency ?? 'USD'}
                  price={analysis.price}
                  stopLoss={analysis.recommendation.stop_loss}
                />
              </section>

              <section className="panel panel--nested">
                <h3>Validación histórica del sistema (backtest walk-forward)</h3>
                <BacktestCard backtest={analysis.backtest} />
              </section>
            </>
          )}

          {tab === 'quant' && (
            <>
              <div className="grid-2">
                <section className="panel panel--nested">
                  <h3>Cadena de Markov (pronóstico probabilístico)</h3>
                  <MarkovChainCard markov={analysis.markov} />
                </section>
                <section className="panel panel--nested">
                  <h3>Volatilidad condicional (GARCH)</h3>
                  <VolatilityCard garch={analysis.garch} />
                </section>
              </div>

              <section className="panel panel--nested">
                <div className="ticker-analysis__section-header">
                  <h3>Simulación Monte Carlo (bandas de precio y probabilidad de stop/objetivo)</h3>
                  <div className="horizon-switch" role="tablist" aria-label="Horizonte de la proyección">
                    {HORIZON_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        role="tab"
                        aria-selected={horizon === opt.value}
                        className={`horizon-switch__item ${horizon === opt.value ? 'horizon-switch__item--active' : ''}`}
                        onClick={() => handleHorizonChange(opt.value)}
                        disabled={loading}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>
                <p className="ticker-analysis__section-hint">
                  El horizonte elegido también actualiza la proyección en el gráfico de precio (pestaña Gráficos).
                </p>
                <MonteCarloChart
                  monteCarlo={analysis.monte_carlo}
                  currentPrice={analysis.price}
                  currency={analysis.currency ?? 'USD'}
                  stopLoss={analysis.recommendation.stop_loss}
                  takeProfit={analysis.recommendation.take_profit}
                />
              </section>
            </>
          )}

          {tab === 'charts' && (
            <section className="panel panel--nested">
              <h3>Gráfico (precio, Bollinger, MA50/200, Gann 1x1, soportes/resistencias, proyección Monte Carlo)</h3>
              <PriceChart
                data={analysis.price_history}
                currency={analysis.currency ?? 'USD'}
                nearestSupport={nearestSupport}
                nearestResistance={nearestResistance}
                monteCarlo={analysis.monte_carlo}
                currentPrice={analysis.price}
              />
              <VolumeChart data={analysis.price_history} />
              <RsiMacdChart data={analysis.price_history} />
            </section>
          )}

          {tab === 'fundamentals' && (
            <>
              <div className="grid-2">
                <section className="panel panel--nested">
                  <h3>Fundamentales y consenso de analistas</h3>
                  <FundamentalsCard fundamentals={analysis.fundamentals} price={analysis.price} currency={analysis.currency ?? 'USD'} />
                </section>
                <section className="panel panel--nested">
                  <h3>Noticias recientes</h3>
                  <NewsList news={analysis.news} />
                </section>
              </div>
              <section className="panel panel--nested">
                <h3>Tenedores institucionales e iniciados</h3>
                <HoldersCard holders={analysis.holders} currency={analysis.currency ?? 'USD'} />
              </section>
            </>
          )}

          {tab === 'seasonality' && (
            <div className="grid-2">
              <section className="panel panel--nested">
                <h3>Estacionalidad mensual (histórico)</h3>
                <SeasonalityChart seasonality={analysis.seasonality} />
              </section>
              <section className="panel panel--nested">
                <h3>Análogos históricos</h3>
                <HistoricalAnalogsCard analogs={analysis.historical_analogs} />
              </section>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default TickerAnalysisPanel
