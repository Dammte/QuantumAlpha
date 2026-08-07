import { formatPercent } from '../../../format'

const VERDICT_LABELS = { comprar: 'Comprar', esperar: 'Esperar', evitar: 'Evitar' }

function interpretationTone(text) {
  if (text.includes('insuficiente')) return 'neutral'
  if (text.startsWith('Advertencia')) return 'down'
  if (text.startsWith('El sistema muestra ventaja')) return 'up'
  return 'neutral'
}

function BacktestCard({ backtest }) {
  if (!backtest) {
    return (
      <p className="empty-state">
        No hay suficiente histórico (o muy pocas señales de "comprar"/"evitar" pasadas) para correr el backtest
        walk-forward en este activo.
      </p>
    )
  }

  const tone = interpretationTone(backtest.interpretation)

  return (
    <div>
      <p className={`backtest-card__interpretation backtest-card__interpretation--${tone}`}>
        {backtest.interpretation}
      </p>

      <table className="backtest-card__table">
        <thead>
          <tr>
            <th>Veredicto</th>
            <th>Casos</th>
            <th>% positivos</th>
            <th>Retorno medio</th>
            <th>Retorno mediano</th>
          </tr>
        </thead>
        <tbody>
          {backtest.bucket_stats.map((b) => (
            <tr key={b.verdict}>
              <td>{VERDICT_LABELS[b.verdict] ?? b.verdict}</td>
              <td>{b.n}</td>
              <td>{b.win_rate !== null ? formatPercent(b.win_rate) : '—'}</td>
              <td className={b.mean_return !== null && b.mean_return >= 0 ? 'delta-up' : 'delta-down'}>
                {b.mean_return !== null ? formatPercent(b.mean_return, { signed: true }) : '—'}
              </td>
              <td className={b.median_return !== null && b.median_return >= 0 ? 'delta-up' : 'delta-down'}>
                {b.median_return !== null ? formatPercent(b.median_return, { signed: true }) : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {backtest.significance_tests.length > 0 && (
        <ul className="backtest-card__tests">
          {backtest.significance_tests.map((t) => (
            <li key={t.comparison}>
              <strong>{t.comparison}</strong>: diferencia {formatPercent(t.mean_difference, { signed: true })} · p
              (Bonferroni) = {t.p_value_bonferroni.toFixed(3)} · p (permutación) = {t.permutation_p_value.toFixed(3)}{' '}
              {t.significant_at_5pct ? '· significativo (95%)' : '· no significativo'}
            </li>
          ))}
        </ul>
      )}

      <p className="backtest-card__disclaimer">
        Replay punto-en-el-tiempo de la lógica de recomendación sobre {backtest.n_samples} ventanas no solapadas de{' '}
        {backtest.horizon_days} días de este activo (sin usar RS Rating ni soportes/resistencias, no disponibles
        retroactivamente sin recalcular todo el universo). Contraste estadístico con prueba t de Welch y prueba de
        permutación, corregido por comparaciones múltiples (Bonferroni).
      </p>
    </div>
  )
}

export default BacktestCard
