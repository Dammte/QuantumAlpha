import { useEffect, useState } from 'react'
import { api } from '../../../api'
import { formatCurrency, formatPercent, formatRatio } from '../../../format'
import { stageLabel } from '../../../marketFormat'
import StatTile from '../../StatTile'
import TrendBadge from '../TrendBadge'
import MultiTimeframeSemaphore from '../../MultiTimeframeSemaphore'
import PriceChart from './PriceChart'
import VolumeChart from './VolumeChart'
import RsiMacdChart from './RsiMacdChart'
import RecommendationCard from './RecommendationCard'
import GateNarrative from './GateNarrative'
import FundamentalsCard from './FundamentalsCard'
import NewsList from './NewsList'
import TripleBarrierBacktestCard from './TripleBarrierBacktestCard'
import RelationshipMapCard from './RelationshipMapCard'

const TABS = [
  { key: 'summary', label: 'Resumen' },
  { key: 'charts', label: 'Gráficos' },
  { key: 'fundamentals', label: 'Fundamentales' },
  { key: 'relationships', label: 'Relaciones' },
]

const HORIZON = '3m' // se persiste en el snapshot histórico - ya no cambia qué se calcula (2026-09)

function TickerAnalysisPanel({ presetTicker } = {}) {
  const [ticker, setTicker] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('summary')
  const [relationshipMap, setRelationshipMap] = useState(null)
  const [relLoading, setRelLoading] = useState(false)
  const [relError, setRelError] = useState(null)

  const search = async (value, selectedHorizon = HORIZON) => {
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

  // Lazy, per-ticker: the relationship map has its own endpoint, so it's
  // only fetched once the user actually opens that tab, not on every
  // "Analizar" click - and cached per ticker so flipping tabs back and
  // forth doesn't refetch.
  useEffect(() => {
    if (tab !== 'relationships' || !analysis) return
    if (relationshipMap && relationshipMap.ticker === analysis.ticker) return
    let cancelled = false
    async function loadRelationships() {
      setRelLoading(true)
      setRelError(null)
      try {
        const map = await api.getRelationshipMap(analysis.ticker)
        if (!cancelled) setRelationshipMap(map)
      } catch (err) {
        if (!cancelled) setRelError(err.message)
      } finally {
        if (!cancelled) setRelLoading(false)
      }
    }
    loadRelationships()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, analysis?.ticker])

  const handleSubmit = (e) => {
    e.preventDefault()
    search(ticker)
  }

  const handleClear = () => {
    setTicker('')
    setAnalysis(null)
    setError(null)
    setRelationshipMap(null)
    setRelError(null)
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
          Escribe un ticker para ver su análisis cuantitativo completo: gráfico con EMA21/55, Bollinger,
          soportes/resistencias, RSI, MACD, semáforo semanal/diario, fundamentales, noticias, backtest de
          barrera triple y el gate de entrada con stop y objetivo sugeridos.
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
              {analysis.is_intraday_snapshot && (
                <span
                  className="ticker-analysis__intraday-badge"
                  title="La sesión de hoy sigue en curso: precio e indicadores son en vivo, no un cierre confirmado - pueden variar antes del cierre."
                >
                  ● Sesión en curso
                </span>
              )}
            </div>
          </div>

          <div className="stat-grid" style={{ marginBottom: 20 }}>
            <StatTile label="Tendencia" value={<TrendBadge trend={analysis.trend} />} />
            <StatTile label="Fase (Weinstein)" value={stageLabel(analysis.stage)} />
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
                <h3>Gate de entrada</h3>
                <MultiTimeframeSemaphore multiTimeframe={analysis.multi_timeframe} />
                <RecommendationCard
                  gate={analysis.gate}
                  imminentCross={analysis.imminent_cross}
                  imminentCrossShortTerm={analysis.imminent_cross_short_term}
                  candlestickPattern={analysis.candlestick_pattern}
                  currency={analysis.currency ?? 'USD'}
                />
                <GateNarrative narrative={analysis.llm_narrative} />
              </section>

              <section className="panel panel--nested">
                <h3>Backtest de triple-barrera (stop/objetivo/trailing real, neto de costes)</h3>
                <TripleBarrierBacktestCard backtest={analysis.triple_barrier_backtest} />
              </section>
            </>
          )}

          {tab === 'charts' && (
            <section className="panel panel--nested">
              <h3>Gráfico (precio, EMA21/55, MA50/200, Bollinger, soportes/resistencias)</h3>
              <PriceChart
                data={analysis.price_history}
                currency={analysis.currency ?? 'USD'}
                nearestSupport={nearestSupport}
                nearestResistance={nearestResistance}
              />
              <VolumeChart data={analysis.price_history} />
              <RsiMacdChart data={analysis.price_history} />
            </section>
          )}

          {tab === 'fundamentals' && (
            <div className="grid-2">
              <section className="panel panel--nested">
                <h3>Fundamentales</h3>
                <FundamentalsCard fundamentals={analysis.fundamentals} />
              </section>
              <section className="panel panel--nested">
                <h3>Noticias recientes</h3>
                <NewsList news={analysis.news} />
              </section>
            </div>
          )}

          {tab === 'relationships' && (
            <section className="panel panel--nested">
              <h3>Mapa de relaciones</h3>
              <p className="ticker-analysis__section-hint">
                Activos relacionados con {analysis.ticker}, en dos capas ordenadas de más a menos fiable - cada
                nombre es un candidato en el que se puede entrar directamente, no solo una etiqueta.
              </p>
              <RelationshipMapCard
                relationshipMap={relationshipMap}
                loading={relLoading}
                error={relError}
                onSelectTicker={search}
              />
            </section>
          )}
        </div>
      )}
    </div>
  )
}

export default TickerAnalysisPanel
