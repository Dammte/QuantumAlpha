import { useState } from 'react'
import { formatCurrency, formatPercent, formatRatio } from '../../../format'

function PositionSizingCard({ sizing, currency, price, stopLoss, verdict, monteCarlo }) {
  const [capital, setCapital] = useState('')

  if (!sizing) return null

  // Tercera auditoría, Bloque H: `sizing` (Kelly) sólo existe cuando el
  // veredicto ya era "comprar" (ver ticker_analysis_service.py), y su
  // `win_probability` viene directamente de estas mismas dos probabilidades
  // de Monte Carlo - pero no son siempre la misma tensión: con un ratio
  // riesgo/beneficio grande, P(stop) > P(objetivo) puede convivir con un
  // Kelly todavía positivo (compensado por ese ratio) - dos casos reales y
  // distintos, no uno solo, cada uno con su propio texto.
  const isBuy = verdict === 'comprar'
  const kellyUnfavorable = sizing.full_kelly_fraction <= 0
  const hasBarriers =
    monteCarlo?.probability_stop_before_target !== null &&
    monteCarlo?.probability_stop_before_target !== undefined &&
    monteCarlo?.probability_target_before_stop !== null &&
    monteCarlo?.probability_target_before_stop !== undefined
  const monteCarloUnfavorable = hasBarriers && monteCarlo.probability_stop_before_target > monteCarlo.probability_target_before_stop
  const showConflictWarning = isBuy && (kellyUnfavorable || monteCarloUnfavorable)

  const capitalValue = parseFloat(capital)
  const hasCapital = !Number.isNaN(capitalValue) && capitalValue > 0
  const suggestedAmount = hasCapital ? capitalValue * sizing.recommended_position_pct : null
  const shares = suggestedAmount && price ? Math.floor(suggestedAmount / price) : null
  const dollarRisk = shares && stopLoss ? shares * (price - stopLoss) : null

  return (
    <div className="position-sizing-card">
      {showConflictWarning && kellyUnfavorable && (
        <div className="position-sizing-card__conflict-warning">
          <strong>⚠️ El veredicto es COMPRAR por puntuación técnica, pero el criterio de Kelly no recomienda
          abrir posición</strong> con la probabilidad de éxito y el ratio riesgo/beneficio estimados aquí
          {monteCarloUnfavorable && (
            <>
              {' '}
              - la simulación Monte Carlo de abajo da más probabilidad de tocar el stop-loss que el objetivo
              dentro de este horizonte
            </>
          )}
          . Son preguntas distintas (la puntuación técnica no exige ningún tamaño de posición concreto), pero
          el tamaño sugerido aquí es 0%.
        </div>
      )}
      {showConflictWarning && !kellyUnfavorable && (
        <div className="position-sizing-card__conflict-warning">
          <strong>⚠️ La simulación Monte Carlo da más probabilidad de tocar el stop-loss que el objetivo</strong>{' '}
          dentro de este horizonte, aunque el ratio riesgo/beneficio todavía compensa esa asimetría lo bastante
          para que Kelly siga sugiriendo un tamaño de posición positivo abajo - vale la pena mirar ambos números
          antes de dimensionar la entrada.
        </div>
      )}
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
