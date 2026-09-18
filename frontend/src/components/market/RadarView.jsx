import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api'
import { formatCurrency, formatPercent, formatRelativeTime } from '../../format'
import { gradeTone, setupStageLabel } from '../../marketFormat'
import TrendBadge from './TrendBadge'

// Reconstruction (2026-09), Fase 5, reescrita en la Fase 10 (Parte 12,
// biblioteca de setups del Radar - docs/quant_methodology.md §28.x) sobre
// GET /market/radar, que desde la Fase 9 ya llega ordenado
// (lexicográficamente, Parte 9.1), agrupable por sector y recortado
// (Parte 9.2) - esta vista solo PRESENTA ese orden, nunca decide uno
// propio ni descarta nada por su cuenta. El modo agrupado/lista (Parte 8)
// y los chips de filtro (Parte 12.1) son, a propósito, presentación pura
// en el cliente: `sector`/`sector_rs_percentile`/`setups` ya viajan en
// cada fila, agrupar o filtrar sobre eso no justifica una segunda forma de
// servir el mismo endpoint.

const VIEW_MODE_STORAGE_KEY = 'radar-view-mode'
const SECTOR_COLLAPSE_PERCENTILE = 30

function leadingSetup(item) {
  // El primero de `setups` ya es el ganador de `arbitration.order_by_rank`
  // ("el mejor gana, los demás son contexto") - esta vista solo LEE esa
  // posición, nunca vuelve a elegir entre setups.
  return item.setups && item.setups.length > 0 ? item.setups[0] : null
}

// Chips de la Parte 12.1 con datos reales detrás. "Sin correlación con mi
// cartera" se queda fuera a propósito: no tiene hoy un campo estructurado en
// RadarItemResponse (`apply_portfolio_grade_modifiers` la calcula, pero solo
// como texto libre dentro de `grade.reasons` - ver CLAUDE.md, "modificadores
// de cartera... siguen sin consumidor" - un subsistema distinto de esta
// biblioteca, no fabricado aquí). "Con muestra medida" SÍ tiene datos reales
// desde la Fase 13 (`setup_performance` vía `measured_stats`) - filtra por
// `confidence === 'measured'`, el mismo criterio binario de la Parte 10.3
// (no "algo de historial", el umbral n>=30 ya decidido en setup_replay.py).
const CHIPS = [
  { id: 'triggered', label: 'Solo disparados', test: (item) => leadingSetup(item)?.stage === 'triggered' },
  { id: 'grade_a', label: 'Solo grado A', test: (item) => item.grade?.grade === 'A' },
  {
    id: 'stage_transition',
    label: 'Etapa 1→2',
    test: (item) => (item.setups ?? []).some((s) => s.family === 'stage_transition'),
  },
  { id: 'vcp', label: 'VCP', test: (item) => (item.setups ?? []).some((s) => s.family === 'vcp') },
  { id: 'breakout', label: 'Rupturas', test: (item) => (item.setups ?? []).some((s) => s.family === 'breakout') },
  { id: 'pullback', label: 'Retrocesos', test: (item) => (item.setups ?? []).some((s) => s.family === 'pullback') },
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
                <span className="radar-row__numeric">Stop {formatCurrency(geometry.stop_price, item.currency)}</span>
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

function readStoredViewMode() {
  try {
    return localStorage.getItem(VIEW_MODE_STORAGE_KEY) === 'list' ? 'list' : 'grouped'
  } catch {
    return 'grouped'
  }
}

function RadarView({ onNavigateToTicker, region, portfolioId }) {
  const [items, setItems] = useState([])
  const [computedAt, setComputedAt] = useState(null)
  const [totalAnalyzed, setTotalAnalyzed] = useState(0)
  const [message, setMessage] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [viewMode, setViewMode] = useState(readStoredViewMode)
  const [activeChips, setActiveChips] = useState(() => new Set())
  const [collapsedOverrides, setCollapsedOverrides] = useState(() => new Map())

  useEffect(() => {
    // `ignore` evita que una respuesta vieja (p. ej. la primera invocación
    // de StrictMode en desarrollo, o la región anterior si el usuario
    // cambia de región dos veces seguidas) pise el estado de una petición
    // más nueva que resolvió antes - encontrado probando esta vista a mano
    // (cambiar de región rápido dejaba viendo datos de la región anterior).
    let ignore = false
    async function load() {
      setLoading(true)
      try {
        const body = await api.getRadar({ region, portfolioId })
        if (ignore) return
        setItems(body.items)
        setComputedAt(body.computed_at)
        setTotalAnalyzed(body.total_analyzed ?? 0)
        setMessage(body.message ?? null)
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

  function setMode(mode) {
    setViewMode(mode)
    try {
      localStorage.setItem(VIEW_MODE_STORAGE_KEY, mode)
    } catch {
      // Preferencia de presentación únicamente - modo privado o cuota
      // agotada no debe romper el Radar, solo no recordar la elección.
    }
  }

  function toggleChip(id) {
    setActiveChips((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const filtered = useMemo(() => {
    if (activeChips.size === 0) return items
    const activeTests = CHIPS.filter((c) => activeChips.has(c.id)).map((c) => c.test)
    return items.filter((item) => activeTests.every((test) => test(item)))
  }, [items, activeChips])

  const groups = useMemo(() => {
    const bySector = new Map()
    for (const item of filtered) {
      const key = item.sector ?? 'Sin sector identificado'
      if (!bySector.has(key)) bySector.set(key, [])
      bySector.get(key).push(item)
    }
    return [...bySector.entries()]
      .map(([sector, sectorItems]) => ({
        sector,
        percentile: sectorItems[0]?.sector_rs_percentile ?? null,
        items: sectorItems,
      }))
      .sort((a, b) => (b.percentile ?? -1) - (a.percentile ?? -1))
  }, [filtered])

  function isCollapsed(sector, defaultCollapsed) {
    return collapsedOverrides.has(sector) ? collapsedOverrides.get(sector) : defaultCollapsed
  }

  function toggleSector(sector, defaultCollapsed) {
    setCollapsedOverrides((prev) => {
      const next = new Map(prev)
      next.set(sector, !isCollapsed(sector, defaultCollapsed))
      return next
    })
  }

  const relative = formatRelativeTime(computedAt)

  return (
    <div>
      <div className="radar-header">
        <div className="radar-header__meta">
          {relative && <span>Calculado por el cierre diario · actualizado {relative}</span>}
          {computedAt && (
            <span className="radar-count">
              {filtered.length} de {totalAnalyzed} analizados
            </span>
          )}
        </div>
        <div className="timeframe-tabs">
          <button
            type="button"
            className={`timeframe-tab ${viewMode === 'grouped' ? 'timeframe-tab--active' : ''}`}
            onClick={() => setMode('grouped')}
          >
            Agrupado
          </button>
          <button
            type="button"
            className={`timeframe-tab ${viewMode === 'list' ? 'timeframe-tab--active' : ''}`}
            onClick={() => setMode('list')}
          >
            Lista
          </button>
        </div>
      </div>

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

      {loading ? (
        <p className="empty-state">Cargando radar…</p>
      ) : error ? (
        <div className="banner banner--error">{error}</div>
      ) : items.length === 0 ? (
        <p className="empty-state">
          {message ??
            'Sin candidatos todavía - el radar se rellena con el cierre diario (`daily_close.py`); si esta región nunca ha corrido, vuelve más tarde.'}
        </p>
      ) : filtered.length === 0 ? (
        <p className="empty-state">Ningún candidato cumple los filtros activos.</p>
      ) : viewMode === 'list' ? (
        <div className="radar-list">
          {filtered.map((item) => (
            <RadarRow key={item.ticker} item={item} onNavigateToTicker={onNavigateToTicker} />
          ))}
        </div>
      ) : (
        <div className="radar-list">
          {groups.map(({ sector, percentile, items: sectorItems }) => {
            const defaultCollapsed = percentile != null && percentile <= SECTOR_COLLAPSE_PERCENTILE
            const collapsed = isCollapsed(sector, defaultCollapsed)
            return (
              <div key={sector} className="radar-sector-group">
                <button
                  type="button"
                  className="radar-sector-group__header"
                  onClick={() => toggleSector(sector, defaultCollapsed)}
                  aria-expanded={!collapsed}
                >
                  <span className="radar-sector-group__caret">{collapsed ? '▹' : '▸'}</span>
                  <span className="radar-sector-group__title">{sector}</span>
                  <span className="radar-sector-group__meta">
                    {percentile != null && `RS ${percentile} · `}
                    {sectorItems.length} candidato{sectorItems.length === 1 ? '' : 's'}
                  </span>
                </button>
                {!collapsed && (
                  <div className="radar-sector-group__body">
                    {sectorItems.map((item) => (
                      <RadarRow key={item.ticker} item={item} onNavigateToTicker={onNavigateToTicker} />
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default RadarView
