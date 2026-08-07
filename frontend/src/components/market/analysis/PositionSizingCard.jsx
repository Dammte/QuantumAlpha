import { useState } from 'react'
import { formatCurrency, formatPercent, formatRatio } from '../../../format'

function PositionSizingCard({ sizing, currency, price, stopLoss }) {
  const [capital, setCapital] = useState('')

  if (!sizing) return null

  const capitalValue = parseFloat(capital)
  const hasCapital = !Number.isNaN(capitalValue) && capitalValue > 0
  const suggestedAmount = hasCapital ? capitalValue * sizing.recommended_position_pct : null
  const shares = suggestedAmount && price ? Math.floor(suggestedAmount / price) : null
  const dollarRisk = shares && stopLoss ? shares * (price - stopLoss) : null

  return (
    <div className="position-sizing-card">
      <div className="position-sizing-card__headline">
        <span className="position-sizing-card__label">Tamaño de posición sugerido (Kelly fraccional)</span>
        <span className="position-sizing-card__value">{formatPercent(sizing.recommended_position_pct)}</span>
        <span className="position-sizing-card__hint">del capital destinado a este tipo de operación</span>
      </div>

      <label className="position-sizing-card__capital">
        <span className="position-sizing-card__capital-label">
          ¿Cuánto capital destinas a este tipo de operación?
        </span>
        <input
          type="number"
          min="0"
          step="any"
          inputMode="decimal"
          value={capital}
          onChange={(e) => setCapital(e.target.value)}
          placeholder="10000"
        />
      </label>

      {hasCapital && (
        <div className="stat-grid" style={{ marginBottom: 14 }}>
          <div className="stat-tile">
            <p className="stat-tile__label">Monto sugerido</p>
            <p className="stat-tile__value">{formatCurrency(suggestedAmount, currency)}</p>
          </div>
          {shares !== null && shares > 0 && (
            <div className="stat-tile">
              <p className="stat-tile__label">Acciones aproximadas</p>
              <p className="stat-tile__value">{shares.toLocaleString('es')}</p>
            </div>
          )}
          {dollarRisk !== null && dollarRisk > 0 && (
            <div className="stat-tile">
              <p className="stat-tile__label">Riesgo en $ si toca el stop-loss</p>
              <p className="stat-tile__value stat-tile__value--down">{formatCurrency(dollarRisk, currency)}</p>
            </div>
          )}
        </div>
      )}

      <div className="stat-grid">
        <div className="stat-tile">
          <p className="stat-tile__label">Probabilidad de éxito estimada</p>
          <p className="stat-tile__value">{formatPercent(sizing.win_probability)}</p>
        </div>
        <div className="stat-tile">
          <p className="stat-tile__label">Ratio riesgo/beneficio</p>
          <p className="stat-tile__value">{formatRatio(sizing.reward_risk_ratio)} : 1</p>
        </div>
        <div className="stat-tile">
          <p className="stat-tile__label">Kelly completo</p>
          <p className="stat-tile__value">{formatPercent(sizing.full_kelly_fraction)}</p>
        </div>
      </div>
      <p className="position-sizing-card__rationale">{sizing.rationale}</p>
    </div>
  )
}

export default PositionSizingCard
