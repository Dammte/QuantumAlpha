import { SETUP_LABELS, formatPercent, formatRatio } from '../../../format'
import TrendBadge from '../TrendBadge'

// Tercera auditoría, Bloque G: "el mapa debe terminar en candidatos
// accionables, no en un diagrama bonito" - every related ticker is a button
// that jumps straight into its own full analysis (`onSelectTicker`), not just
// a label. Three layers, ordered and labeled by decreasing reliability - see
// `relationship_map_service.py`'s module docstring for the full reasoning.

function SetupTag({ setup, percentileScore }) {
  if (!setup) return <span className="relationship-map__no-setup">—</span>
  return (
    <span className="setup-badge">
      {SETUP_LABELS[setup] ?? setup}
      {percentileScore !== null && percentileScore !== undefined && <> · p{Math.round(percentileScore)}</>}
    </span>
  )
}

function leadLagLabel(ticker, otherTicker, lagDays) {
  if (lagDays === null || lagDays === undefined) return '—'
  if (lagDays === 0) return 'Simultáneo'
  return lagDays > 0 ? `${ticker} lidera por ${lagDays}d` : `${otherTicker} lidera por ${Math.abs(lagDays)}d`
}

function StatisticalRelationsTable({ ticker, relations, onSelectTicker }) {
  if (relations.length === 0) {
    return (
      <p className="empty-state">
        No hay suficiente historial de retornos en común con otros activos del universo para calcular
        correlaciones fiables todavía.
      </p>
    )
  }
  return (
    <div className="table-scroll table-scroll--capped">
      <table className="positions-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Sector</th>
            <th className="num">Corr. 60d</th>
            <th className="num">Corr. 250d</th>
            <th className="num">Beta relativa</th>
            <th>Desfase</th>
            <th className="num">Co-mov. días extremos</th>
            <th>Setup hoy</th>
          </tr>
        </thead>
        <tbody>
          {relations.map((r) => (
            <tr key={r.ticker}>
              <td>
                <button
                  type="button"
                  className="positions-table__ticker positions-table__ticker--link"
                  onClick={() => onSelectTicker(r.ticker)}
                  title="Ver análisis completo"
                >
                  {r.ticker}
                </button>
                {r.is_diverging && (
                  <span
                    className="badge badge--sell relationship-map__divergence-badge"
                    title="Correlacionado históricamente (≥250 sesiones), pero desacoplado en las últimas 60 sesiones"
                  >
                    Divergiendo
                  </span>
                )}
              </td>
              <td>{r.sector ?? '—'}</td>
              <td className="num">{formatRatio(r.correlation_60d)}</td>
              <td className="num">{formatRatio(r.correlation_250d)}</td>
              <td className="num">{formatRatio(r.relative_beta)}</td>
              <td>{leadLagLabel(ticker, r.ticker, r.lead_lag_days)}</td>
              <td className="num">{formatPercent(r.comovement_extreme_days_pct)}</td>
              <td>
                <SetupTag setup={r.setup} percentileScore={r.percentile_score} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SectorPeersTable({ peers, onSelectTicker }) {
  if (peers.length === 0) {
    return (
      <p className="empty-state">
        Este activo no pertenece a ninguna industria curada del universo (o no tiene más miembros en nuestra región).
      </p>
    )
  }
  return (
    <div className="table-scroll table-scroll--capped">
      <table className="positions-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Industria</th>
            <th className="num">RS Rating</th>
            <th>Tendencia</th>
            <th>Setup hoy</th>
          </tr>
        </thead>
        <tbody>
          {peers.map((p) => (
            <tr key={p.ticker}>
              <td>
                <button
                  type="button"
                  className="positions-table__ticker positions-table__ticker--link"
                  onClick={() => onSelectTicker(p.ticker)}
                  title="Ver análisis completo"
                >
                  {p.ticker}
                </button>
              </td>
              <td>{p.industry}</td>
              <td className="num">{p.rs_rating ?? '—'}</td>
              <td>
                <TrendBadge trend={p.trend} />
              </td>
              <td>
                <SetupTag setup={p.setup} percentileScore={p.percentile_score} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DisclosedRelationsList({ available, relations }) {
  if (!available) {
    return (
      <p className="empty-state">
        No disponible para este activo: EDGAR solo cubre emisores domiciliados en EE.UU., o no se pudo resolver el
        nombre exacto de la empresa.
      </p>
    )
  }
  if (relations.length === 0) {
    return (
      <p className="empty-state">
        Ninguna otra empresa menciona a este activo por su nombre en un 10-K/10-Q propio de los últimos 2 años.
      </p>
    )
  }
  return (
    <ul className="relationship-map__disclosed-list">
      {relations.map((d, i) => (
        // No stable id from EDGAR's own search index for a single hit - filer
        // name + form + date is unique enough within one ticker's result set.
        <li key={`${d.filer_name}-${d.form}-${d.filing_date}-${i}`}>
          <span className="relationship-map__disclosed-filer">{d.filer_name}</span>
          <span className="relationship-map__disclosed-meta">
            {d.form} · {d.filing_date}
          </span>
        </li>
      ))}
    </ul>
  )
}

function RelationshipMapCard({ relationshipMap, loading, error, onSelectTicker }) {
  if (loading) return <p className="empty-state">Buscando activos relacionados…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (!relationshipMap) return null

  const { ticker, statistical, sector_peers: sectorPeers, disclosed, disclosed_available: disclosedAvailable } = relationshipMap

  return (
    <div className="relationship-map">
      <section className="relationship-map__layer relationship-map__layer--statistical">
        <div className="relationship-map__layer-header">
          <h4>
            1. Estadística <span className="relationship-map__reliability relationship-map__reliability--high">más fiable</span>
          </h4>
          <p className="relationship-map__hint">
            Correlación de retornos, beta relativa, desfase (quién se mueve primero) y co-movimiento en días
            extremos - calculado sobre el histórico del universo ya descargado, sin llamadas de red nuevas.
          </p>
        </div>
        <StatisticalRelationsTable ticker={ticker} relations={statistical} onSelectTicker={onSelectTicker} />
      </section>

      <section className="relationship-map__layer relationship-map__layer--sector">
        <div className="relationship-map__layer-header">
          <h4>
            2. Pares de sector / industria <span className="relationship-map__reliability relationship-map__reliability--mid">contexto</span>
          </h4>
          <p className="relationship-map__hint">
            Otros activos del universo curado en la misma industria, con su fuerza relativa y estado técnico
            actual. Sin cuadrante RRG todavía (fuera de alcance de esta ronda).
          </p>
        </div>
        <SectorPeersTable peers={sectorPeers} onSelectTicker={onSelectTicker} />
      </section>

      <section className="relationship-map__layer relationship-map__layer--disclosed">
        <div className="relationship-map__layer-header">
          <h4>
            3. Relaciones declaradas en SEC (10-K/10-Q){' '}
            <span className="relationship-map__reliability relationship-map__reliability--low">más especulativa</span>
          </h4>
          <p className="relationship-map__hint">
            Otras empresas que mencionan a este activo por su nombre en su propio 10-K/10-Q reciente - la señal
            más literal de una relación comercial real, y también la menos verificada (una coincidencia de texto,
            no un hecho confirmado).
          </p>
        </div>
        <DisclosedRelationsList available={disclosedAvailable} relations={disclosed ?? []} />
      </section>
    </div>
  )
}

export default RelationshipMapCard
