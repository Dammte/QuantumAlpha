import { formatNumber, formatPercent } from '../../../format'

// Segunda auditoría, Bloque 2/4: la lectura honesta del backtest -
// etiquetado con barrera triple (stop/objetivo/vertical, consciente de
// huecos de apertura), trailing Chandelier real, y neto de costes (ver
// backtest_engine.py). Distinto de BacktestCard.jsx (el walk-forward legacy,
// que ignora el stop/objetivo) - se muestran juntos, cada uno etiquetado por
// lo que realmente es, nunca uno sustituyendo al otro en silencio.
const STRATEGY_LABELS = {
  strategy_fixed: 'Estrategia (stop/objetivo fijos)',
  strategy_trailing: 'Estrategia (trailing Chandelier real)',
  buy_and_hold: 'Comprar y mantener (referencia)',
  random_entries: 'Entradas aleatorias (referencia sin ventaja)',
}
const STRATEGY_ORDER = ['strategy_fixed', 'strategy_trailing', 'buy_and_hold', 'random_entries']

function MetricsRow({ label, metrics }) {
  if (!metrics || metrics.n_trades === 0) {
    return (
      <tr>
        <td>{label}</td>
        <td colSpan={6} className="empty-state">Sin operaciones en este historial</td>
      </tr>
    )
  }
  return (
    <tr>
      <td>{label}</td>
      <td className="num">{metrics.n_trades}</td>
      <td className="num">{metrics.win_rate !== null ? formatPercent(metrics.win_rate) : '—'}</td>
      <td className={`num ${(metrics.expectancy_pct ?? 0) >= 0 ? 'delta-up' : 'delta-down'}`}>
        {metrics.expectancy_pct !== null ? formatPercent(metrics.expectancy_pct, { signed: true }) : '—'}
      </td>
      <td className="num">{metrics.profit_factor !== null ? formatNumber(metrics.profit_factor) : '—'}</td>
      <td className="num delta-down">
        {metrics.max_drawdown_pct !== null ? formatPercent(metrics.max_drawdown_pct, { signed: true }) : '—'}
      </td>
      <td className="num">
        {metrics.avg_mae_pct !== null ? formatPercent(metrics.avg_mae_pct, { signed: true }) : '—'} /{' '}
        {metrics.avg_mfe_pct !== null ? formatPercent(metrics.avg_mfe_pct, { signed: true }) : '—'}
      </td>
    </tr>
  )
}

function TripleBarrierBacktestCard({ backtest }) {
  if (!backtest) {
    return (
      <p className="empty-state">
        No hay suficiente histórico para correr el backtest de triple-barrera en este activo.
      </p>
    )
  }

  return (
    <div>
      <p className="backtest-card__disclaimer">
        Etiquetado con barrera triple (stop, objetivo o vencimiento a {backtest.horizon_days} sesiones - lo que se
        toque primero, consciente de huecos de apertura) sobre {backtest.n_signals_evaluated} señales "comprar"
        replayadas en una rejilla no solapada. <strong>Todas las cifras de P&amp;L son netas de costes</strong>{' '}
        (MAE/MFE se muestran brutos a propósito - describen el recorrido intrabarra, no la ganancia/pérdida real).
      </p>
      <div className="table-scroll">
        <table className="backtest-card__table">
          <thead>
            <tr>
              <th>Estrategia</th>
              <th className="num">Operaciones</th>
              <th className="num">% acierto</th>
              <th className="num">Expectativa</th>
              <th className="num">Factor de beneficio</th>
              <th className="num">Máx. drawdown</th>
              <th className="num">MAE / MFE (bruto)</th>
            </tr>
          </thead>
          <tbody>
            {STRATEGY_ORDER.map((key) => (
              <MetricsRow key={key} label={STRATEGY_LABELS[key]} metrics={backtest[key]} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default TripleBarrierBacktestCard
