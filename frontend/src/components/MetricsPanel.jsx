import StatTile from './StatTile'
import { formatPercent, formatRatio } from '../format'

function toneFor(value) {
  if (value === null || value === undefined) return 'neutral'
  return value >= 0 ? 'up' : 'down'
}

function MetricsPanel({ metrics }) {
  if (!metrics) return null

  const hasBenchmark = metrics.beta !== null && metrics.beta !== undefined

  return (
    <div className="metrics-groups">
      <div className="metrics-group">
        <p className="metrics-group__title">Rendimiento</p>
        <div className="stat-grid">
          <StatTile label="Retorno acumulado" value={formatPercent(metrics.cumulative_return, { signed: true })} tone={toneFor(metrics.cumulative_return)} />
          <StatTile label="CAGR" value={formatPercent(metrics.cagr, { signed: true })} tone={toneFor(metrics.cagr)} />
          <StatTile label="Mejor día" value={formatPercent(metrics.best_day, { signed: true })} tone="up" />
          <StatTile label="Peor día" value={formatPercent(metrics.worst_day, { signed: true })} tone="down" />
          <StatTile label="Win rate" value={formatPercent(metrics.win_rate)} />
        </div>
      </div>

      <div className="metrics-group">
        <p className="metrics-group__title">Riesgo</p>
        <div className="stat-grid">
          <StatTile label="Volatilidad anualizada" value={formatPercent(metrics.annualized_volatility)} />
          <StatTile label="Ratio de Sharpe" value={formatRatio(metrics.sharpe_ratio)} tone={toneFor(metrics.sharpe_ratio)} />
          <StatTile label="Ratio de Sortino" value={formatRatio(metrics.sortino_ratio)} tone={toneFor(metrics.sortino_ratio)} />
          <StatTile label="Ratio de Calmar" value={formatRatio(metrics.calmar_ratio)} tone={toneFor(metrics.calmar_ratio)} />
          <StatTile label="Máximo drawdown" value={formatPercent(metrics.max_drawdown)} tone="down" />
          <StatTile label="Drawdown actual" value={formatPercent(metrics.current_drawdown)} tone={metrics.current_drawdown < 0 ? 'down' : 'neutral'} />
          <StatTile label="VaR (95%, diario)" value={formatPercent(metrics.var_95)} tone="down" />
          <StatTile label="CVaR (95%, diario)" value={formatPercent(metrics.cvar_95)} tone="down" />
        </div>
      </div>

      {hasBenchmark && (
        <div className="metrics-group">
          <p className="metrics-group__title">Vs. benchmark</p>
          <div className="stat-grid">
            <StatTile label="Beta" value={formatRatio(metrics.beta)} hint="Sensibilidad frente al benchmark" />
            <StatTile label="Alfa de Jensen (anual)" value={formatPercent(metrics.alpha, { signed: true })} tone={toneFor(metrics.alpha)} hint="Rendimiento no explicado por el mercado" />
          </div>
        </div>
      )}
    </div>
  )
}

export default MetricsPanel
