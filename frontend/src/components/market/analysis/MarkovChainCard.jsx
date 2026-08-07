import { formatPercent } from '../../../format'
import StatTile from '../../StatTile'

const SHORT_LABELS = ['F.Baj', 'Bajista', 'Lateral', 'Alcista', 'F.Alc']

function cellColor(p) {
  // 0 -> transparent, 1 -> full accent - a plain intensity heatmap, no red/green
  // semantics here since a transition matrix row isn't inherently good/bad.
  const alpha = Math.min(1, p * 1.8)
  return `rgba(var(--accent-rgb), ${alpha.toFixed(3)})`
}

function MarkovChainCard({ markov }) {
  if (!markov) {
    return (
      <p className="empty-state">
        No hay suficiente histórico (se necesita más de un año de datos) para estimar una cadena de Markov confiable.
      </p>
    )
  }

  const bullish = markov.prob_bullish_21d >= 0.55
  const bearish = markov.prob_bullish_21d <= 0.45

  return (
    <div>
      <p className="markov-card__state">
        Estado actual: <strong>{markov.current_state_label}</strong>
      </p>

      <div className="stat-grid">
        <StatTile
          label="Prob. alcista a 21 días"
          value={formatPercent(markov.prob_bullish_21d)}
          tone={bullish ? 'up' : bearish ? 'down' : 'neutral'}
        />
        <StatTile label="Retorno esperado 5 días" value={formatPercent(markov.forecast_5d_return, { signed: true })} />
        <StatTile label="Retorno esperado 21 días" value={formatPercent(markov.forecast_21d_return, { signed: true })} />
      </div>

      <p className="markov-card__subheading">Matriz de transición (probabilidad de pasar de un estado a otro)</p>
      <div className="markov-heatmap">
        <div className="markov-heatmap__row markov-heatmap__row--header">
          <span className="markov-heatmap__corner" />
          {SHORT_LABELS.map((label) => (
            <span key={label} className="markov-heatmap__col-label">
              {label}
            </span>
          ))}
        </div>
        {markov.transition_matrix.map((row, i) => (
          <div key={SHORT_LABELS[i]} className="markov-heatmap__row">
            <span className="markov-heatmap__row-label">{SHORT_LABELS[i]}</span>
            {row.map((p, j) => (
              <span
                key={j}
                className="markov-heatmap__cell"
                style={{ background: cellColor(p) }}
                title={`${SHORT_LABELS[i]} → ${SHORT_LABELS[j]}: ${formatPercent(p)}`}
              >
                {Math.round(p * 100)}
              </span>
            ))}
          </div>
        ))}
      </div>

      {markov.sequence_looks_random && (
        <p className="markov-card__caution">
          La secuencia de subidas/bajadas de este activo no se distingue estadísticamente de un paseo aleatorio (test
          de rachas) - el pronóstico anterior no se usa para puntuar la recomendación por falta de respaldo
          estadístico.
        </p>
      )}
      {!markov.sequence_looks_random && markov.order2_justified && (
        <p className="markov-card__note">
          Hay evidencia de que más de un paso anterior influye en el siguiente movimiento (test de orden, p=
          {markov.order2_p_value.toFixed(3)}) - una dependencia más rica que un simple modelo de un paso.
        </p>
      )}
    </div>
  )
}

export default MarkovChainCard
