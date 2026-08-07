import { formatPercent } from '../../../format'

function HistoricalAnalogsCard({ analogs }) {
  if (!analogs) {
    return (
      <p className="empty-state">
        No hay suficiente histórico (se necesitan varios años de datos) para buscar análogos.
      </p>
    )
  }

  return (
    <div>
      <p className="analogs-card__intro">
        De los {analogs.n_analogs} momentos históricos de este mismo activo más parecidos al estado técnico actual
        (por momentum y volatilidad reciente), esto es lo que pasó en los {analogs.forward_horizon_days} días
        siguientes:
      </p>
      <div className="stat-grid">
        <div className="stat-tile">
          <p className="stat-tile__label">Retorno medio</p>
          <p className={`stat-tile__value ${analogs.avg_forward_return >= 0 ? 'stat-tile__value--up' : 'stat-tile__value--down'}`}>
            {formatPercent(analogs.avg_forward_return, { signed: true })}
          </p>
        </div>
        <div className="stat-tile">
          <p className="stat-tile__label">Retorno mediano</p>
          <p className={`stat-tile__value ${analogs.median_forward_return >= 0 ? 'stat-tile__value--up' : 'stat-tile__value--down'}`}>
            {formatPercent(analogs.median_forward_return, { signed: true })}
          </p>
        </div>
        <div className="stat-tile">
          <p className="stat-tile__label">% de veces positivo</p>
          <p className="stat-tile__value">{formatPercent(analogs.win_rate)}</p>
        </div>
      </div>
      <p className="analogs-card__disclaimer">
        Comparación estadística simple (k-vecinos más cercanos) sobre el propio historial del activo - no es una
        predicción, solo contexto de qué ha tendido a pasar en condiciones similares.
      </p>
    </div>
  )
}

export default HistoricalAnalogsCard
