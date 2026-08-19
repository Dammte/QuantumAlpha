import {
  EXIT_URGENCY_LABELS,
  EXIT_URGENCY_TONE,
  SCALED_EXIT_ACTION_LABELS,
  formatCurrency,
  formatPercent,
  formatRMultiple,
} from '../format'
import MultiTimeframeSemaphore from './MultiTimeframeSemaphore'
import StatTile from './StatTile'

// The "ficha de posición" from the Fase 7 brief: everything about *this*
// specific open trade the positions table's one-line signal badge can't
// show - the plan it's being judged against, how far along it is, and
// exactly why the exit engine currently rates it the way it does. `risk` is
// the PositionRisk for this ticker (never null when this renders - see
// PositionsTable's expand guard); `currency` comes from the summary
// position, not `risk.currency`, so it's never out of sync with the rest of
// the row it expands from.
function PositionDetailPanel({ risk, currency }) {
  const plan = risk.trade_plan
  const atr14 = risk.signals?.atr14 ?? null

  const distanceToStopPct =
    plan?.current_stop != null && risk.price ? (risk.price - plan.current_stop) / risk.price : null
  const distanceToStopAtr =
    plan?.current_stop != null && atr14 ? (risk.price - plan.current_stop) / atr14 : null

  const urgencyTone = risk.exit_urgency ? (EXIT_URGENCY_TONE[risk.exit_urgency] ?? 'neutral') : 'neutral'

  return (
    <div className="position-detail">
      {!plan ? (
        <p className="empty-state">
          Todavía no hay un plan de operación para esta posición (se genera en la próxima actualización de riesgo).
        </p>
      ) : (
        <>
          <div className="stat-grid position-detail__stats">
            <StatTile label="Precio de entrada" value={formatCurrency(plan.entry_price, currency)} />
            <StatTile label="Fecha de entrada" value={plan.entry_date} />
            <StatTile
              label="Stop vigente"
              value={plan.current_stop != null ? formatCurrency(plan.current_stop, currency) : '—'}
              hint={
                plan.initial_stop != null && plan.current_stop !== plan.initial_stop
                  ? `Inicial: ${formatCurrency(plan.initial_stop, currency)}`
                  : undefined
              }
              tone="down"
            />
            <StatTile
              label="Objetivo inicial"
              value={plan.initial_target != null ? formatCurrency(plan.initial_target, currency) : '—'}
              tone="up"
            />
            <StatTile
              label="R actual"
              value={formatRMultiple(risk.r_multiple)}
              tone={risk.r_multiple != null ? (risk.r_multiple >= 0 ? 'up' : 'down') : 'neutral'}
            />
            <StatTile label="Sesiones mantenidas" value={risk.bars_held ?? '—'} />
            <StatTile
              label="Distancia al stop"
              value={distanceToStopPct != null ? formatPercent(distanceToStopPct) : '—'}
              hint={distanceToStopAtr != null ? `${distanceToStopAtr.toFixed(1)}x ATR` : undefined}
              tone={distanceToStopPct != null && distanceToStopPct < 0 ? 'down' : 'neutral'}
            />
            {risk.exit_urgency && (
              <StatTile label="Urgencia de salida" value={EXIT_URGENCY_LABELS[risk.exit_urgency]} tone={urgencyTone} />
            )}
          </div>

          <p className="position-detail__thesis">{plan.thesis}</p>

          {risk.scaled_exit && risk.scaled_exit.action !== 'none' && (
            <p className="position-detail__scaled-exit">
              <strong>{SCALED_EXIT_ACTION_LABELS[risk.scaled_exit.action] ?? risk.scaled_exit.action}:</strong>{' '}
              {risk.scaled_exit.description}
            </p>
          )}
        </>
      )}

      {risk.exit_reasons?.length > 0 && (
        <div className="position-detail__reasons">
          <p className="position-detail__section-label">Motivos (todos los niveles, no solo el que ganó)</p>
          <ul>
            {risk.exit_reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </div>
      )}

      {risk.multi_timeframe && (
        <div className="position-detail__mtf">
          <p className="position-detail__section-label">Lectura multi-temporalidad</p>
          <MultiTimeframeSemaphore multiTimeframe={risk.multi_timeframe} />
        </div>
      )}

      <p className="position-detail__footnote">
        Basado en el cierre de la última sesión confirmada, no en la cotización intradía en vivo.
      </p>
    </div>
  )
}

export default PositionDetailPanel
