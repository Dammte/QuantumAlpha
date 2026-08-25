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

function PremiumWatchlistCard({ item, onNavigateToTicker }) {
  const { signals } = item
  // Tercera auditoría, Bloque F-10: the card used to show
  // signals.recommendation.score (the raw checklist score) - premium_score
  // is what actually ranks and cuts this list (setup percentile/sector/
  // entry-timing adjustments folded in), so it's what should be on the
  // badge. isExceptionalScore's own threshold is calibrated against the raw
  // score, not the adjusted one - kept as-is on purpose.
  const exceptional = isExceptionalScore(signals.recommendation.score)

  return (
    <div className={`watchlist-card premium-watchlist-card ${exceptional ? 'premium-watchlist-card--exceptional' : ''}`}>
      {exceptional && <span className="premium-watchlist-card__exceptional-tag">★ Señal excepcional</span>}
      <div className="watchlist-card__header">
        <div>
          <span className="positions-table__ticker">{item.ticker}</span>
          <span className="watchlist-card__industry">{item.industry ?? item.sector}</span>
        </div>
        <span className="badge badge--buy">
          COMPRAR ({item.premium_score >= 0 ? '+' : ''}{item.premium_score.toFixed(1)})
        </span>
      </div>

      {item.days_to_earnings !== null && item.days_to_earnings !== undefined && item.days_to_earnings >= 0 && (
        <p className="premium-watchlist-card__earnings-warning">
          ⚠️ Resultados en {item.days_to_earnings} sesiones - riesgo de evento dentro de la ventana del trade
        </p>
      )}

      {item.setup && (
        <span className="setup-badge">
          {item.setup_label ?? SETUP_LABELS[item.setup] ?? item.setup}
        </span>
      )}
      {item.also_matched_setups?.length > 0 && (
        <span className="setup-badge setup-badge--secondary">
          también:{' '}
          {item.also_matched_setups
            .map((s, i) => item.also_matched_setup_labels?.[i] ?? SETUP_LABELS[s] ?? s)
            .join(', ')}
        </span>
      )}
      {item.setup_outcome_stats && (
        <p className="watchlist-card__setup-outcome">
          Histórico de este setup ({item.setup_outcome_stats.n} casos):{' '}
          {formatPercent(item.setup_outcome_stats.win_rate)} aciertos, expectancy{' '}
          {item.setup_outcome_stats.expectancy_r >= 0 ? '+' : ''}
          {item.setup_outcome_stats.expectancy_r.toFixed(2)}R, duración mediana{' '}
          {item.setup_outcome_stats.median_bars_held} sesiones, MAE p80 −
          {item.setup_outcome_stats.mae_p80_pct.toFixed(1)}%
        </p>
      )}

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
          Ver análisis completo → (incluye el backtest de barrera triple)
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
        pasaron el mismo análisis completo de "Analizar activo" (GARCH, cadena de Markov, Monte Carlo) y superaron el
        veredicto "comprar". El orden y el corte los decide <strong>premium_score</strong> (el badge superior): la
        puntuación del checklist ajustada por el percentil del setup, el sector y el momento de entrada - no el
        tamaño de posición Kelly (que se muestra igual, pero ya no puntúa esta selección) ni el backtest walk-forward
        (retirado de aquí - ver "Analizar activo" para el backtest de barrera triple). Las marcadas con{' '}
        <strong>★ Señal excepcional</strong> tienen una puntuación base de 10 o más - muy pocas llegan ahí, y son las
        que más factores independientes confirman a la vez.
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
        <div>
          <p className="empty-state">
            Ningún activo pasó la barra "premium" ahora mismo para este horizonte - eso es normal y esperado: la
            lista es selectiva a propósito.
          </p>
          {(() => {
            // Tercera auditoría, Bloque F-10: exactly the case where knowing
            // how many were even looked at matters most - "0 aprobados" reads
            // very differently next to "0 de 0 candidatos" than next to
            // "0 de 15 candidatos".
            const stats = discardStats.find((s) => s.tier === tier)
            if (!stats) return null
            return (
              <p className="premium-watchlist__discard-note">
                {stats.analyzed} de {stats.prefilter_matches} candidatos del prefiltro fueron analizados en
                detalle - ninguno superó la barra "premium".
              </p>
            )
          })()}
        </div>
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
                <PremiumWatchlistCard
                  key={`${item.tier}-${item.ticker}-${item.setup ?? 'none'}`}
                  item={item}
                  onNavigateToTicker={onNavigateToTicker}
                />
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
