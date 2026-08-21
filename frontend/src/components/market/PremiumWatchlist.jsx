import { useEffect, useState } from 'react'
import { api } from '../../api'
import { SETUP_LABELS, formatCurrency, formatPercent, garchRegimeLabel, isExceptionalScore } from '../../format'
import CandlestickPatternBadge from './CandlestickPatternBadge'
import EntryTimingBadge from './EntryTimingBadge'
import ImminentCrossBadge from './ImminentCrossBadge'
import RefreshBar from '../RefreshBar'

// No "todas" option on purpose: computing all three tiers at once means the
// full quant suite (GARCH, Markov, Monte Carlo, backtest, Kelly) over up to
// 45 tickers in one request - exactly the kind of avoidable load this is
// meant to prevent. Defaults to "daily" below; weekly/monthly only get
// computed if you actually click them.
const TIERS = [
  { key: 'daily', label: 'Diaria' },
  { key: 'weekly', label: 'Semanal' },
  { key: 'monthly', label: 'Mensual' },
]

const TIER_META = {
  daily: { label: 'Revisión diaria', hint: 'Setups de corto plazo - se re-evalúa una vez al día' },
  weekly: { label: 'Revisión semanal', hint: 'Confirmaciones de medio plazo - se re-evalúa una vez a la semana' },
  monthly: { label: 'Revisión mensual', hint: 'Tendencias de fondo - se re-evalúa una vez al mes' },
}
const TIER_ORDER = ['daily', 'weekly', 'monthly']

function backtestBadge(backtest) {
  if (!backtest) return { label: 'Historial insuficiente', tone: 'neutral' }
  const test = backtest.significance_tests.find((t) => t.comparison === 'comprar vs evitar')
  if (!test) return { label: 'Historial insuficiente', tone: 'neutral' }
  if (test.significant_at_5pct && test.mean_difference > 0) return { label: 'Ventaja confirmada', tone: 'up' }
  if (test.significant_at_5pct && test.mean_difference <= 0) return { label: 'Sin ventaja aquí', tone: 'down' }
  return { label: 'Sin diferencia significativa', tone: 'neutral' }
}

function PremiumWatchlistCard({ item, onNavigateToTicker }) {
  const { signals } = item
  const badge = backtestBadge(signals.backtest)
  const exceptional = isExceptionalScore(signals.recommendation.score)

  return (
    <div className={`watchlist-card premium-watchlist-card ${exceptional ? 'premium-watchlist-card--exceptional' : ''}`}>
      {exceptional && <span className="premium-watchlist-card__exceptional-tag">★ Señal excepcional</span>}
      <div className="watchlist-card__header">
        <div>
          <span className="positions-table__ticker">{item.ticker}</span>
          <span className="watchlist-card__industry">{item.industry ?? item.sector}</span>
        </div>
        <span className="badge badge--buy">COMPRAR ({signals.recommendation.score >= 0 ? '+' : ''}{signals.recommendation.score})</span>
      </div>

      {item.setup && <span className="setup-badge">{SETUP_LABELS[item.setup] ?? item.setup}</span>}

      <div className="watchlist-card__stats">
        <span>{formatCurrency(signals.price, item.currency)}</span>
        <span className={signals.change_1d >= 0 ? 'delta-up' : 'delta-down'}>
          {formatPercent(signals.change_1d, { signed: true })} (1D)
        </span>
        {signals.rs_rating !== null && <span>RS {signals.rs_rating}</span>}
        {signals.garch && <span>{garchRegimeLabel(signals.garch.regime)}</span>}
      </div>

      <EntryTimingBadge entryTiming={signals.entry_timing} />
      <ImminentCrossBadge imminentCross={signals.imminent_cross} />
      <ImminentCrossBadge imminentCross={signals.imminent_cross_short_term} shortTerm />
      <CandlestickPatternBadge pattern={signals.candlestick_pattern} />

      <p className="premium-watchlist-card__backtest">
        <span className={`sector-tier-badge sector-tier-badge--${badge.tone}`}>{badge.label}</span>
        {signals.backtest && <span className="premium-watchlist-card__backtest-text"> {signals.backtest.interpretation}</span>}
      </p>

      {signals.position_sizing && (
        <p className="premium-watchlist-card__kelly">
          Tamaño de posición sugerido: <strong>{formatPercent(signals.position_sizing.recommended_position_pct)}</strong> de la cartera
        </p>
      )}

      <ul className="watchlist-card__reasons">
        {item.reasons.map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>

      {onNavigateToTicker && (
        <button type="button" className="premium-watchlist-card__cta" onClick={() => onNavigateToTicker(item.ticker)}>
          Ver análisis completo →
        </button>
      )}
    </div>
  )
}

function PremiumWatchlist({ onNavigateToTicker, region }) {
  const [tier, setTier] = useState('daily')
  const [items, setItems] = useState([])
  const [discardStats, setDiscardStats] = useState([])
  const [computedAt, setComputedAt] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const body = await api.getPremiumWatchlist({ region, tier })
        setItems(body.items)
        setDiscardStats(body.discard_stats ?? [])
        setComputedAt(body.computed_at)
        setError(null)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region, tier])

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      const body = await api.getPremiumWatchlist({ region, tier, refresh: true })
      setItems(body.items)
      setDiscardStats(body.discard_stats ?? [])
      setComputedAt(body.computed_at)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setRefreshing(false)
    }
  }

  const sections = TIER_ORDER.filter((t) => t === tier)
    .map((t) => ({ tier: t, items: items.filter((i) => i.tier === t) }))
    .filter((section) => section.items.length > 0)

  return (
    <div>
      <p className="empty-state" style={{ marginBottom: 16 }}>
        Una selección reducida (hasta 10 por horizonte) de activos que no solo cumplen una regla técnica, sino que
        pasaron el mismo análisis completo de "Analizar activo" - GARCH, cadena de Markov, Monte Carlo, backtest
        walk-forward y tamaño de posición Kelly - y salieron respaldados. No es lo mismo revisar 150 activos que
        revisar 10 excelentes. Las marcadas con <strong>★ Señal excepcional</strong> tienen una puntuación de 10 o
        más - muy pocas llegan ahí, y son las que más factores independientes confirman a la vez.
      </p>

      <div className="filters-row">
        {TIERS.map((t) => (
          <button
            key={t.key}
            type="button"
            className={`timeframe-tab ${tier === t.key ? 'timeframe-tab--active' : ''}`}
            onClick={() => setTier(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {!loading && (
        <RefreshBar computedAt={computedAt} onRefresh={handleRefresh} refreshing={refreshing} />
      )}

      {loading ? (
        <p className="empty-state">Analizando candidatos con el motor cuantitativo completo…</p>
      ) : error ? (
        <div className="banner banner--error">{error}</div>
      ) : sections.length === 0 ? (
        <p className="empty-state">
          Ningún activo pasó la barra "premium" ahora mismo para este horizonte - eso es normal y esperado: la lista
          es selectiva a propósito.
        </p>
      ) : (
        sections.map((section) => {
          const stats = discardStats.find((s) => s.tier === section.tier)
          return (
          <section className="watchlist-section" key={section.tier}>
            <h4 className="watchlist-section__title">
              {TIER_META[section.tier].label}
              <span className="premium-watchlist__tier-hint"> · {TIER_META[section.tier].hint}</span>
            </h4>
            {stats && stats.prefilter_matches > stats.analyzed && (
              <p className="premium-watchlist__discard-note">
                {stats.analyzed} de {stats.prefilter_matches} candidatos analizados en detalle
                {stats.approved < stats.analyzed && ` · ${stats.approved} superaron la barra "premium"`}
              </p>
            )}
            <div className="watchlist-grid">
              {section.items.map((item) => (
                <PremiumWatchlistCard key={`${item.tier}-${item.ticker}`} item={item} onNavigateToTicker={onNavigateToTicker} />
              ))}
            </div>
          </section>
          )
        })
      )}
    </div>
  )
}

export default PremiumWatchlist
