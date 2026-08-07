import { formatCurrency, formatPercent } from '../../../format'

function formatShares(value) {
  if (value === null || value === undefined) return '—'
  return value.toLocaleString('en-US', { maximumFractionDigits: 0 })
}

function formatDateReported(value) {
  if (!value) return '—'
  return value.split(' ')[0].split('T')[0]
}

function HoldersCard({ holders, currency }) {
  if (!holders) {
    return <p className="empty-state">No hay datos de tenedores institucionales disponibles para este ticker.</p>
  }

  return (
    <div>
      <div className="stat-grid" style={{ marginBottom: 16 }}>
        <div className="stat-tile">
          <p className="stat-tile__label">% en manos de instituciones</p>
          <p className="stat-tile__value">
            {holders.pct_held_by_institutions !== null ? formatPercent(holders.pct_held_by_institutions) : '—'}
          </p>
        </div>
        <div className="stat-tile">
          <p className="stat-tile__label">% en manos de insiders</p>
          <p className="stat-tile__value">
            {holders.pct_held_by_insiders !== null ? formatPercent(holders.pct_held_by_insiders) : '—'}
          </p>
        </div>
      </div>

      {holders.top_institutional_holders.length > 0 ? (
        <div className="table-scroll">
          <table className="positions-table">
            <thead>
              <tr>
                <th>Institución</th>
                <th className="num">Acciones</th>
                <th className="num">Valor</th>
                <th className="num">% en cartera</th>
                <th>Reportado</th>
              </tr>
            </thead>
            <tbody>
              {holders.top_institutional_holders.map((h) => (
                <tr key={h.holder}>
                  <td className="positions-table__ticker">{h.holder}</td>
                  <td className="num">{formatShares(h.shares)}</td>
                  <td className="num">{h.value !== null ? formatCurrency(h.value, currency) : '—'}</td>
                  <td className="num">{h.pct_held !== null ? formatPercent(h.pct_held) : '—'}</td>
                  <td>{formatDateReported(h.date_reported)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="empty-state">Sin tenedores institucionales individuales disponibles.</p>
      )}

      <p className="holders-card__disclaimer">
        Datos de Yahoo Finance (reportes 13F trimestrales, con hasta 45 días de rezago) - directriz de contexto
        sobre el interés institucional, no una fuente autoritativa; puede contener imprecisiones puntuales.
      </p>
    </div>
  )
}

export default HoldersCard
