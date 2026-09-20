import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api'
import { formatCurrency, formatPercent, formatRelativeTime } from '../../format'
import { gradeTone, setupStageLabel } from '../../marketFormat'
import TrendBadge from './TrendBadge'

// Auditoria del Radar, bloque 10 (texto literal completo del encargo): dos
// listas por horizonte (corto/medio plazo, hasta 10 cada una, is_primary
// marcando "el principal a entrar" si supera el umbral), ficha ampliada del
// primario, y las 5 subsecciones del bloque G - cabecera de régimen de
// mercado (con consecuencia real sobre tamaño/umbral, no decorativa),
// "a punto de disparar", "rompiendo por abajo", mapa de sectores y nota de
// cobertura al pie. "El Radar tiene que caber en una pantalla" (literal):
// el flat `items` (hasta 25, superconjunto de las dos listas) pasa a ser una
// sección secundaria y colapsada por defecto - "todos los candidatos" - no
// la vista principal como en la reconstrucción anterior a este bloque.
// Chips podados de 8 a 4 (bloque G, literal: "elimina... todo chip que no
// tenga consumidor real... menos superficie, más señal") - las dos listas
// curadas ya hacen el trabajo de "qué setup es cuál"; los 4 que quedan son
// los que aportan un corte que ninguna otra parte de la vista ya muestra.

function leadingSetup(item) {
  // El primero de `setups` ya es el ganador de `arbitration.order_by_rank`
  // ("el mejor gana, los demás son contexto") - esta vista solo LEE esa
  // posición, nunca vuelve a elegir entre setups.
  return item.setups && item.setups.length > 0 ? item.setups[0] : null
}

const CHIPS = [
  { id: 'grade_a', label: 'Solo grado A', test: (item) => item.grade?.grade === 'A' },
  { id: 'triggered', label: 'Solo disparados', test: (item) => leadingSetup(item)?.stage === 'triggered' },
  { id: 'strong_sector', label: 'Sectores fuertes', test: (item) => (item.sector_rs_percentile ?? 0) >= 70 },
  {
    id: 'measured',
    label: 'Con muestra medida',
    test: (item) => leadingSetup(item)?.confidence === 'measured',
  },
]

function timeframeCellTone(cell) {
  if (!cell) return ''
  if (cell.bias === 'bullish') return 'radar-row__timeframe-cell--bullish'
  if (cell.bias === 'bearish') return 'radar-row__timeframe-cell--bearish'
  return ''
}

function TimeframeStrip({ strip }) {
  if (!strip) return null
  return (
    <span className="radar-row__timeframe-strip">
      <span
        className={`radar-row__timeframe-cell radar-row__timeframe-cell--monthly ${timeframeCellTone(strip.monthly)}`}
        title={`Mensual: ${strip.monthly.note} — Contexto de largo plazo, no participa en la decisión.`}
      >
        M
      </span>
      <span
        className={`radar-row__timeframe-cell ${timeframeCellTone(strip.weekly)}`}
        title={`Semanal: ${strip.weekly.note}`}
      >
        S
      </span>
      <span
        className={`radar-row__timeframe-cell ${timeframeCellTone(strip.daily)}`}
        title={`Diario: ${strip.daily.note}`}
      >
        D
      </span>
    </span>
  )
}

function SetupBadge({ setup }) {
  if (!setup) return null
  return (
    <span className={`radar-row__setup radar-row__setup--${setup.stage}`} title={setup.narrative_es || undefined}>
      {setup.label_es} · {setupStageLabel(setup.stage)}
    </span>
  )
}

function SetupStatBadge({ stats }) {
  // Parte 10.3/12.1: solo se muestra con muestra real detrás
  // (n_observations viene siempre que exista una fila en setup_performance,
  // pero win_rate/expectancy_r solo se rellenan una vez algo ha disparado -
  // ver setup_replay.aggregate_setup_performance) - "sin muestra" se queda
  // sin badge, nunca un "85% de probabilidad" fabricado.
  if (!stats || stats.win_rate == null || stats.expectancy_r == null) return null
  const sign = stats.expectancy_r >= 0 ? '+' : ''
  return (
    <span
      className="watchlist-card__industry"
      title={`Medido sobre ${stats.n_observations} disparos históricos - no es una promesa para el próximo trade.`}
    >
      {(stats.win_rate * 100).toFixed(0)}% · {sign}
      {stats.expectancy_r.toFixed(2)}R
    </span>
  )
}

// Parte 11.2, literal: "este valor ha formado 4 VCP en 5 años; 3 dispararon
// y 2 alcanzaron objetivo". Plantilla determinista sobre `ticker_history`
// (conteos de ESTE ticker, sin Gemini) - nunca una probabilidad (Parte 15).
function tickerHistorySentence(setup) {
  const h = setup.ticker_history
  const years = Math.max(
    1,
    Math.round((new Date(h.last_ready_date) - new Date(h.first_ready_date)) / (365.25 * 24 * 3600 * 1000))
  )
  const yearsLabel = years === 1 ? '1 año' : `${years} años`
  const timesLabel = h.n_observations === 1 ? 'una vez' : `${h.n_observations} veces`
  return (
    `Este valor ha formado ${setup.label_es} ${timesLabel} en ${yearsLabel}; ` +
    `${h.n_triggered} dispararon y ${h.n_target_hit} alcanzaron objetivo.`
  )
}

// Bloque E2, literal: "el score y el desglose por componente viajan en la
// respuesta... sin desglose, esto vuelve a ser una caja negra". Cada
// componente ya llega multiplicado por su peso, y las penalizaciones ya en
// negativo - esta tabla solo las lista, ningún cálculo nuevo en el cliente.
const SCORE_COMPONENT_LABELS = [
  ['setup_quality', 'Calidad del setup'],
  ['relative_strength', 'Fuerza relativa'],
  ['trigger_proximity', 'Proximidad al disparador'],
  ['geometry_quality', 'Calidad de la geometría'],
  ['volume_confirmation', 'Confirmación de volumen'],
]
const SCORE_PENALTY_LABELS = [
  ['earnings_penalty', 'Earnings próximos'],
  ['high_atr_penalty', 'ATR elevado vs. universo'],
]

function ScoreBreakdown({ score }) {
  if (!score) return null
  return (
    <div className="radar-row__detail-section">
      <p className="radar-row__detail-title">Score: {score.total.toFixed(0)}/100 - por qué está aquí</p>
      <div className="radar-row__trigger-invalidation">
        {SCORE_COMPONENT_LABELS.map(([key, label]) => (
          <span key={key} className="radar-row__numeric">
            {label}: {score[key].toFixed(1)}
          </span>
        ))}
        {SCORE_PENALTY_LABELS.filter(([key]) => score[key] < 0).map(([key, label]) => (
          <span key={key} className="radar-row__numeric radar-row__numeric--penalty">
            {label}: {score[key].toFixed(1)}
          </span>
        ))}
      </div>
    </div>
  )
}

function RadarRow({ item, onNavigateToTicker }) {
  const [expanded, setExpanded] = useState(false)
  const setup = leadingSetup(item)
  const otherSetups = (item.setups ?? []).slice(1)
  const classicPatterns = (item.setups ?? []).filter((s) => s.family === 'classic_pattern')
  const grade = item.grade?.grade ?? null
  const distanceAtr = item.grade?.distance_atr ?? null
  const geometry = item.entry_geometry
  const reasons = item.grade?.reasons ?? []
  const evidenceEntries = setup ? Object.entries(setup.evidence ?? {}) : []

  return (
    <div className={`radar-row ${item.gate_passes ? 'radar-row--pass' : 'radar-row--pending'}`}>
      <div className="radar-row__summary">
        <div className="radar-row__identity">
          {onNavigateToTicker ? (
            <button
              type="button"
              className="positions-table__ticker positions-table__ticker--link"
              onClick={() => onNavigateToTicker(item.ticker)}
              title="Ver análisis completo"
            >
              {item.ticker}
            </button>
          ) : (
            <span className="positions-table__ticker">{item.ticker}</span>
          )}
          {grade && (
            <span
              className={`badge badge--${gradeTone(grade)} radar-row__grade`}
              title="Grado: geometría, no probabilidad"
            >
              {grade}
            </span>
          )}
          {item.score && (
            <span className="watchlist-card__industry" title="Score compuesto (bloque E2) - ver detalle">
              {item.score.total.toFixed(0)}/100
            </span>
          )}
          <SetupBadge setup={setup} />
        </div>

        <div className="radar-row__stats">
          {distanceAtr != null && <span className="radar-row__numeric">{distanceAtr.toFixed(2)} ATR</span>}
          {setup?.trigger_price != null && (
            <span className="radar-row__numeric" title="Precio que confirma el setup">
              disp. {formatCurrency(setup.trigger_price, item.currency)}
            </span>
          )}
          {setup?.invalidation_price != null && (
            <span className="radar-row__numeric" title="Precio que invalida el setup, tan visible como el disparo">
              inv. {formatCurrency(setup.invalidation_price, item.currency)}
            </span>
          )}
          {geometry?.stop_price != null && (
            <span className="radar-row__numeric">
              stop {formatCurrency(geometry.stop_price, item.currency)}
              {geometry.entry_price
                ? ` (${formatPercent((geometry.stop_price - geometry.entry_price) / geometry.entry_price, { signed: true })})`
                : ''}
            </span>
          )}
          {geometry?.risk_reward_net != null && (
            <span className="radar-row__numeric">R/R {geometry.risk_reward_net.toFixed(1)}</span>
          )}
          <TimeframeStrip strip={item.timeframe_strip} />
          {item.sector && (
            <span className="watchlist-card__industry">
              {item.sector}
              {item.sector_rs_percentile != null && ` · RS ${item.sector_rs_percentile}`}
            </span>
          )}
          <TrendBadge trend={item.trend} />
          <SetupStatBadge stats={setup?.measured_stats} />
          <button type="button" className="timeframe-tab" onClick={() => setExpanded((v) => !v)}>
            {expanded ? 'Ocultar detalle' : 'Ver detalle'}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="radar-row__detail">
          {setup?.narrative_es && <p className="radar-row__narrative">{setup.narrative_es}</p>}

          <ScoreBreakdown score={item.score} />

          {evidenceEntries.length > 0 && (
            <div className="radar-row__detail-section">
              <p className="radar-row__detail-title">Evidencia</p>
              <dl className="radar-row__evidence">
                {evidenceEntries.map(([key, value]) => (
                  <span key={key}>
                    <dt>{key}</dt>
                    <dd>{String(value)}</dd>
                  </span>
                ))}
              </dl>
            </div>
          )}

          {setup && (
            <div className="radar-row__trigger-invalidation">
              <span>
                <strong>Gatillo:</strong> {setup.trigger_condition || '—'}
              </span>
              <span>
                <strong>Anulación:</strong> {setup.invalidation_condition || '—'}
              </span>
            </div>
          )}

          {geometry && (
            <div className="radar-row__detail-section">
              <p className="radar-row__detail-title">Geometría completa</p>
              <div className="radar-row__trigger-invalidation">
                <span className="radar-row__numeric">Entrada {formatCurrency(geometry.entry_price, item.currency)}</span>
                <span className="radar-row__numeric">
                  Stop {formatCurrency(geometry.stop_price, item.currency)}
                  {geometry.stop_basis ? ` (${geometry.stop_basis})` : ''}
                </span>
                <span className="radar-row__numeric">
                  Objetivo {formatCurrency(geometry.target_price, item.currency)}
                </span>
                {geometry.shares_for_risk_budget != null && (
                  <span className="radar-row__numeric">
                    {Math.floor(geometry.shares_for_risk_budget)} acciones
                  </span>
                )}
                {geometry.pct_of_portfolio != null && (
                  <span className="radar-row__numeric">{formatPercent(geometry.pct_of_portfolio)} de cartera</span>
                )}
              </div>
            </div>
          )}

          {reasons.length > 0 && (
            <div className="radar-row__detail-section">
              <p className="radar-row__detail-title">Modificadores de contexto activos</p>
              <div className="radar-row__reason-badges">
                {reasons.map((reason) => (
                  <span key={reason} className="radar-row__reason-badge">
                    {reason}
                  </span>
                ))}
              </div>
            </div>
          )}

          {otherSetups.length > 0 && (
            <div className="radar-row__detail-section">
              <p className="radar-row__detail-title">También cumple</p>
              <div className="radar-row__other-setups">
                {otherSetups.map((s) => (
                  <span key={s.name} className="watchlist-card__industry">
                    {s.label_es} ({setupStageLabel(s.stage)})
                  </span>
                ))}
              </div>
            </div>
          )}

          {classicPatterns.length > 0 && (
            <div className="radar-row__detail-section">
              <p className="radar-row__detail-title">Patrones clásicos detectados</p>
              <div className="radar-row__other-setups">
                {classicPatterns.map((s) => (
                  <span key={s.name} className="watchlist-card__industry">
                    {s.label_es}
                    {s.trigger_price == null && ' · no dispara por sí solo'}
                  </span>
                ))}
              </div>
            </div>
          )}

          {(setup?.measured_stats || setup?.ticker_history) && (
            <div className="radar-row__detail-section">
              <p className="radar-row__detail-title">Estadística del setup e historial en este valor</p>
              {setup.measured_stats && (
                <div className="radar-row__trigger-invalidation">
                  <span className="radar-row__numeric">{setup.measured_stats.n_observations} observaciones</span>
                  {setup.measured_stats.trigger_rate != null && (
                    <span className="radar-row__numeric">
                      {(setup.measured_stats.trigger_rate * 100).toFixed(0)}% llegó a disparar
                    </span>
                  )}
                  {setup.measured_stats.win_rate != null && (
                    <span className="radar-row__numeric">
                      {(setup.measured_stats.win_rate * 100).toFixed(0)}% tocó objetivo antes que stop
                    </span>
                  )}
                  {setup.measured_stats.expectancy_r != null && (
                    <span className="radar-row__numeric">
                      Expectancy {setup.measured_stats.expectancy_r >= 0 ? '+' : ''}
                      {setup.measured_stats.expectancy_r.toFixed(2)}R
                    </span>
                  )}
                </div>
              )}
              {setup.ticker_history && (
                <p className="radar-row__narrative">{tickerHistorySentence(setup)}</p>
              )}
              <p className="radar-row__narrative">Historia medida, no una promesa para el próximo trade.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Bloque E4/H3, literal: "el primario lleva una ficha ampliada: entrada,
// disparador exacto, stop con su anclaje en texto, objetivo, R:R, tamaño de
// posición sugerido en euros... y dos o tres frases de tesis". `thesis` ya
// llega generada (plantilla determinista hasta que el bloque 12 conecte
// Gemini por encima - el LLM redacta, nunca decide qué es primario).
function PrimaryCard({ item }) {
  const geometry = item.entry_geometry
  const setup = leadingSetup(item)
  return (
    <div className="radar-primary-card">
      <div className="radar-primary-card__header">
        <span className="badge badge--buy">Principal a entrar</span>
        <span className="positions-table__ticker">{item.ticker}</span>
        {item.grade?.grade && (
          <span className={`badge badge--${gradeTone(item.grade.grade)}`}>{item.grade.grade}</span>
        )}
        <span className="watchlist-card__industry">{item.score.total.toFixed(0)}/100</span>
      </div>
      {item.thesis && <p className="radar-primary-card__thesis">{item.thesis}</p>}
      {geometry && (
        <div className="radar-primary-card__geometry">
          <span>
            <strong>Entrada</strong> {formatCurrency(geometry.entry_price, item.currency)}
          </span>
          <span>
            <strong>Stop</strong> {formatCurrency(geometry.stop_price, item.currency)}
            {geometry.stop_basis ? ` — ${geometry.stop_basis}` : ''}
          </span>
          <span>
            <strong>Objetivo</strong> {formatCurrency(geometry.target_price, item.currency)}
          </span>
          {geometry.risk_reward_net != null && (
            <span>
              <strong>R:R neto</strong> {geometry.risk_reward_net.toFixed(1)}
            </span>
          )}
          {geometry.position_value != null && (
            <span>
              <strong>Tamaño sugerido</strong> {formatCurrency(geometry.position_value, item.currency)}
              {geometry.shares_for_risk_budget != null && ` (${Math.floor(geometry.shares_for_risk_budget)} acc.)`}
            </span>
          )}
        </div>
      )}
      {setup?.trigger_condition && (
        <p className="radar-primary-card__trigger">
          <strong>Gatillo:</strong> {setup.trigger_condition}
        </p>
      )}
    </div>
  )
}

// Bloque E4, literal: "si el mejor candidato del día no llega al umbral,
// ninguno es primario... un sistema que cada día me señala obligatoriamente
// un principal me empuja a operar por operar."
function NoPrimaryNotice() {
  return (
    <p className="radar-primary-card__no-primary">
      Hoy ningún candidato alcanza el nivel de convicción para entrada principal.
    </p>
  )
}

// Bloque G/10, literal: "cabecera de régimen de mercado (una sola línea)...
// el contexto tiene que tener consecuencia, no ser decorado."
function RegimeHeader({ regime }) {
  if (!regime || regime.status === 'desconocido') return null
  const tone = regime.status === 'bajista' ? 'banner--warning' : ''
  return (
    <div className={`banner radar-regime-banner ${tone}`}>
      {regime.headline}
      {regime.breadth_change_5d != null && (
        <span className="radar-regime-banner__change">
          {' '}
          Amplitud {regime.breadth_change_5d >= 0 ? '+' : ''}
          {(regime.breadth_change_5d * 100).toFixed(0)} pts en 5 sesiones.
        </span>
      )}
    </div>
  )
}

// Bloque G/10, literal: "'a punto de disparar' (máximo 5): valores a menos
// de 0.3 ATR de su disparador que no están todavía en las listas. Es la
// lista de alarmas para mañana."
function AboutToTriggerSection({ items, onNavigateToTicker }) {
  if (items.length === 0) return null
  return (
    <section className="radar-subsection">
      <h3 className="radar-subsection__title">A punto de disparar</h3>
      <div className="radar-list">
        {items.map((item) => (
          <RadarRow key={item.ticker} item={item} onNavigateToTicker={onNavigateToTicker} />
        ))}
      </div>
    </section>
  )
}

function brokenLevelLabel(level) {
  const kindLabel = { ema21: 'EMA21', ema55: 'EMA55', pivot_support: 'soporte' }[level.kind] ?? level.kind
  const sessionsLabel = level.bars_since_loss === 1 ? '1 sesión' : `${level.bars_since_loss} sesiones`
  return `perdió ${kindLabel} hace ${sessionsLabel}`
}

// Bloque G/10, literal: "'rompiendo por abajo' (máximo 5): valores que han
// perdido un soporte, la EMA21 o la EMA55 en las últimas 3 sesiones...
// marca visualmente los que están en mi cartera."
function BreakingDownSection({ items, onNavigateToTicker }) {
  if (items.length === 0) return null
  return (
    <section className="radar-subsection">
      <h3 className="radar-subsection__title">Rompiendo por abajo</h3>
      <div className="radar-breaking-down">
        {items.map((item) => (
          <div key={item.ticker} className="radar-breaking-down__row">
            {onNavigateToTicker ? (
              <button
                type="button"
                className="positions-table__ticker positions-table__ticker--link"
                onClick={() => onNavigateToTicker(item.ticker)}
              >
                {item.ticker}
              </button>
            ) : (
              <span className="positions-table__ticker">{item.ticker}</span>
            )}
            {item.held && (
              <span className="badge badge--warn" title="Ya en tu cartera">
                En cartera
              </span>
            )}
            {item.sector && <span className="watchlist-card__industry">{item.sector}</span>}
            <span className="radar-row__numeric">{formatCurrency(item.price, item.currency)}</span>
            {item.broken_levels.map((level) => (
              <span key={level.kind} className="radar-row__numeric radar-row__numeric--penalty">
                {brokenLevelLabel(level)}
              </span>
            ))}
          </div>
        ))}
      </div>
    </section>
  )
}

// Bloque G/10, literal: "mapa de sectores (compacto): fuerza relativa por
// sector en semanal y diario, con la columna mensual visible pero
// explícitamente excluida del ranking. Ordenados de más fuerte a más
// débil." `RadarItemResponse` hoy solo trae un `sector_rs_percentile`
// (sin separar semanal/diario en un campo propio) - se muestra el único
// disponible, con nota honesta, en vez de fabricar una segunda columna que
// no existe todavía (ver la lista de fuera de alcance al cierre del bloque).
function SectorMap({ items }) {
  const sectors = useMemo(() => {
    const bySector = new Map()
    for (const item of items) {
      if (!item.sector || bySector.has(item.sector)) continue
      bySector.set(item.sector, item.sector_rs_percentile ?? null)
    }
    return [...bySector.entries()].sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1))
  }, [items])

  if (sectors.length === 0) return null
  return (
    <section className="radar-subsection">
      <h3 className="radar-subsection__title">Mapa de sectores</h3>
      <p className="radar-subsection__note">
        Percentil de fuerza relativa por sector - la mensual es contexto y nunca entra en la ordenación.
      </p>
      <div className="radar-sector-map">
        {sectors.map(([sector, percentile]) => (
          <div key={sector} className="radar-sector-map__row">
            <span className="radar-sector-map__name">{sector}</span>
            <span className="radar-sector-map__bar-track">
              <span className="radar-sector-map__bar" style={{ width: `${Math.max(percentile ?? 0, 4)}%` }} />
            </span>
            <span className="radar-row__numeric">{percentile != null ? `RS ${percentile}` : '—'}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

// Bloque G/10, literal: "nota de cobertura, al pie y siempre visible:
// cuántos valores se han analizado, sobre cuántos del universo, con qué
// fecha de datos y por qué vía."
function CoverageNote({ source, coverage, partial, computedAt }) {
  const relative = formatRelativeTime(computedAt)
  const sourceLabel = source === 'live_fallback' ? 'cálculo en vivo (cierre diario no disponible)' : 'cierre diario'
  return (
    <footer className="radar-coverage-note">
      {coverage.analyzed} de {coverage.universe} valores del universo analizados · vía {sourceLabel}
      {relative && ` · actualizado ${relative}`}
      {partial && ' · resultado parcial, el cómputo en vivo no terminó a tiempo'}
    </footer>
  )
}

function HorizonList({ title, items, message, onNavigateToTicker }) {
  return (
    <section className="radar-subsection">
      <h3 className="radar-subsection__title">{title}</h3>
      {message && <p className="radar-subsection__note">{message}</p>}
      {items.length === 0 && !message ? (
        <p className="empty-state">Sin candidatos en este horizonte hoy.</p>
      ) : (
        <div className="radar-list">
          {items.map((item) => (
            <RadarRow key={item.ticker} item={item} onNavigateToTicker={onNavigateToTicker} />
          ))}
        </div>
      )}
    </section>
  )
}

function readStoredAllExpanded() {
  try {
    return localStorage.getItem('radar-all-candidates-expanded') === 'true'
  } catch {
    return false
  }
}

function RadarView({ onNavigateToTicker, region, portfolioId }) {
  const [body, setBody] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeChips, setActiveChips] = useState(() => new Set())
  const [allExpanded, setAllExpanded] = useState(readStoredAllExpanded)

  useEffect(() => {
    // `ignore` evita que una respuesta vieja (p. ej. la primera invocación
    // de StrictMode en desarrollo, o la región anterior si el usuario
    // cambia de región dos veces seguidas) pise el estado de una petición
    // más nueva que resolvió antes.
    let ignore = false
    async function load() {
      setLoading(true)
      try {
        const response = await api.getRadar({ region, portfolioId })
        if (ignore) return
        setBody(response)
        setError(null)
      } catch (err) {
        if (!ignore) setError(err.message)
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [region, portfolioId])

  function toggleChip(id) {
    setActiveChips((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function toggleAllExpanded() {
    setAllExpanded((prev) => {
      const next = !prev
      try {
        localStorage.setItem('radar-all-candidates-expanded', String(next))
      } catch {
        // Preferencia de presentación únicamente.
      }
      return next
    })
  }

  const items = useMemo(() => body?.items ?? [], [body])
  const filteredItems = useMemo(() => {
    if (activeChips.size === 0) return items
    const activeTests = CHIPS.filter((c) => activeChips.has(c.id)).map((c) => c.test)
    return items.filter((item) => activeTests.every((test) => test(item)))
  }, [items, activeChips])

  if (loading) return <p className="empty-state">Cargando radar…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (!body) return null

  const shortTerm = body.short_term ?? []
  const mediumTerm = body.medium_term ?? []
  const primary = shortTerm.find((item) => item.is_primary) ?? null

  if (body.computed_at == null || items.length === 0) {
    return (
      <div>
        <RegimeHeader regime={body.regime} />
        <p className="empty-state">
          {body.message ??
            'Sin candidatos todavía - el radar se rellena con el cierre diario (`daily_close.py`); si esta región nunca ha corrido, vuelve más tarde.'}
        </p>
      </div>
    )
  }

  return (
    <div className="radar-view">
      <RegimeHeader regime={body.regime} />

      {shortTerm.length > 0 && (primary ? <PrimaryCard item={primary} /> : <NoPrimaryNotice />)}

      <div className="radar-horizon-lists">
        <HorizonList
          title="Corto plazo (2-10 sesiones)"
          items={shortTerm}
          message={body.short_term_message}
          onNavigateToTicker={onNavigateToTicker}
        />
        <HorizonList
          title="Medio plazo (3-10 semanas, para vigilar)"
          items={mediumTerm}
          message={body.medium_term_message}
          onNavigateToTicker={onNavigateToTicker}
        />
      </div>

      <AboutToTriggerSection items={body.about_to_trigger ?? []} onNavigateToTicker={onNavigateToTicker} />
      <BreakingDownSection items={body.breaking_down ?? []} onNavigateToTicker={onNavigateToTicker} />
      <SectorMap items={items} />

      <section className="radar-subsection">
        <button type="button" className="timeframe-tab" onClick={toggleAllExpanded}>
          {allExpanded ? 'Ocultar todos los candidatos' : `Ver todos los candidatos analizados (${items.length})`}
        </button>
        {allExpanded && (
          <>
            <div className="radar-chips">
              {CHIPS.map((chip) => (
                <button
                  key={chip.id}
                  type="button"
                  className={`radar-chip ${activeChips.has(chip.id) ? 'radar-chip--active' : ''}`}
                  onClick={() => toggleChip(chip.id)}
                >
                  {chip.label}
                </button>
              ))}
            </div>
            {filteredItems.length === 0 ? (
              <p className="empty-state">Ningún candidato cumple los filtros activos.</p>
            ) : (
              <div className="radar-list">
                {filteredItems.map((item) => (
                  <RadarRow key={item.ticker} item={item} onNavigateToTicker={onNavigateToTicker} />
                ))}
              </div>
            )}
          </>
        )}
      </section>

      <CoverageNote
        source={body.source}
        coverage={body.coverage}
        partial={body.partial}
        computedAt={body.computed_at}
      />
    </div>
  )
}

export default RadarView
