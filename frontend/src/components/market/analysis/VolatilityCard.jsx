import { formatPercent } from '../../../format'
import StatTile from '../../StatTile'

const REGIME_META = {
  baja: { label: 'Baja', tone: 'up' },
  normal: { label: 'Normal', tone: 'neutral' },
  elevada: { label: 'Elevada', tone: 'warn' },
  alta: { label: 'Alta', tone: 'down' },
}

function VolatilityCard({ garch }) {
  if (!garch) {
    return (
      <p className="empty-state">
        No hay suficiente histórico (se necesita más de un año de datos) para ajustar un modelo GARCH(1,1) de
        volatilidad.
      </p>
    )
  }

  const meta = REGIME_META[garch.regime] ?? { label: garch.regime, tone: 'neutral' }

  return (
    <div>
      <div className="volatility-card__regime">
        <span className={`volatility-card__badge volatility-card__badge--${meta.tone}`}>Régimen {meta.label}</span>
        <div className="volatility-card__gauge">
          <div className="volatility-card__gauge-track">
            <div
              className={`volatility-card__gauge-fill volatility-card__gauge-fill--${meta.tone}`}
              style={{ width: `${Math.round(garch.vol_percentile * 100)}%` }}
            />
          </div>
          <span className="volatility-card__gauge-label">
            Percentil {Math.round(garch.vol_percentile * 100)} de su propio historial de volatilidad
          </span>
        </div>
      </div>

      <div className="stat-grid">
        <StatTile label="Volatilidad actual (anualizada)" value={formatPercent(garch.current_vol_annualized)} />
        <StatTile label="Pronóstico a 21 días" value={formatPercent(garch.forecast_vol_21d_annualized)} />
        <StatTile label="Volatilidad de largo plazo" value={formatPercent(garch.unconditional_vol_annualized)} />
        <StatTile
          label="Persistencia (α+β)"
          value={garch.persistence.toFixed(2)}
          hint={garch.persistence > 0.97 ? 'los shocks de volatilidad tardan mucho en disiparse' : undefined}
        />
      </div>
      <p className="volatility-card__disclaimer">
        Modelo GARCH(1,1) ajustado por máxima verosimilitud sobre el historial propio del activo - la volatilidad se
        agrupa en el tiempo (los días de alta volatilidad tienden a seguirse de más), a diferencia de asumir una
        volatilidad constante.
      </p>
    </div>
  )
}

export default VolatilityCard
