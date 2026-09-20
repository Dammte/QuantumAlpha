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

// Auditoria del Radar, bloque H3, literal: "el stop deja de ser un número
// suelto" - mismas cuatro etiquetas que `trade_geometry.classify_volatility_profile`.
const VOLATILITY_PROFILE_LABELS = {
  tranquilo: 'Tranquilo',
  normal: 'Normal',
  volatil: 'Volátil',
  extremo: 'Extremo',
}

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
  // Auditoria del Radar, bloque H2/H3: el perfil de volatilidad y el techo
  // de riesgo son del ESTADO ACTUAL del valor (la lectura fresca del gate,
  // `risk.signals.gate.entry_geometry`), no del momento de entrada - son
  // contexto de hoy, complementan el anclaje persistido, no lo sustituyen.
  const liveGeometry = risk.signals?.gate?.entry_geometry ?? null

  // El anclaje EN TEXTO del stop vigente - `current_stop_basis` una vez el
  // Chandelier toma el relevo, si no, el mismo `initial_stop_basis` con el
  // que se abrió la posición (bloque H2: "el trailing nunca debe machacar
  // el registro del inicial").
  const stopBasis = plan?.current_stop_basis ?? plan?.initial_stop_basis ?? null
  const stopHasMoved =
    plan?.initial_stop != null && plan?.current_stop != null && plan.current_stop !== plan.initial_stop

  const distanceToStopPct =
    plan?.current_stop != null && risk.price ? (risk.price - plan.current_stop) / risk.price : null
  const distanceToStopEur = plan?.current_stop != null && risk.price ? risk.price - plan.current_stop : null
  const distanceToStopAtr = plan?.current_stop != null && atr14 ? (risk.price - plan.current_stop) / atr14 : null
  const distanceHint =
    [
      distanceToStopEur != null ? formatCurrency(distanceToStopEur, currency) : null,
      distanceToStopAtr != null ? `${distanceToStopAtr.toFixed(1)}x ATR` : null,
    ]
      .filter(Boolean)
      .join(' · ') || undefined

  const urgencyTone = risk.exit_urgency ? (EXIT_URGENCY_TONE[risk.exit_urgency] ?? 'neutral') : 'neutral'

  return (
    <div className="position-detail">
      {!plan ? (
        <p className="empty-state">
          Todavía no hay un plan de operación para esta posición (se genera en la próxima actualización de riesgo).
        </p>
      ) : (
        <>
          {plan.current_stop == null && (
            <p className="banner banner--warning">
              Sin nivel estructural defendible para esta posición hoy - considera no mantenerla con un stop técnico.
            </p>
          )}

          <div className="stat-grid position-detail__stats">
            <StatTile label="Precio de entrada" value={formatCurrency(plan.entry_price, currency)} />
            <StatTile label="Fecha de entrada" value={plan.entry_date} />
            {/* Bloque H3, literal: "el precio del stop y su anclaje en texto...
                distancia en % y en euros - el % como consecuencia, presentado
                después del nivel, no antes" - el nivel (esta ficha) precede a
                la distancia (la siguiente). */}
            <StatTile
              label="Stop vigente"
              value={plan.current_stop != null ? formatCurrency(plan.current_stop, currency) : '—'}
              hint={stopBasis ?? undefined}
              tone="down"
            />
            <StatTile
              label="Distancia al stop"
              value={distanceToStopPct != null ? formatPercent(distanceToStopPct) : '—'}
              hint={distanceHint}
              tone={distanceToStopPct != null && distanceToStopPct < 0 ? 'down' : 'neutral'}
            />
            {stopHasMoved && (
              <StatTile
                label="Stop inicial (para saber cuánto se ha movido)"
                value={formatCurrency(plan.initial_stop, currency)}
                hint={plan.initial_stop_basis ?? undefined}
              />
            )}
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
            {liveGeometry?.volatility_profile && (
              <StatTile
                label="Perfil de volatilidad"
                value={VOLATILITY_PROFILE_LABELS[liveGeometry.volatility_profile] ?? liveGeometry.volatility_profile}
                hint={
                  liveGeometry.risk_ceiling_pct != null
                    ? `Techo de riesgo: ${formatPercent(liveGeometry.risk_ceiling_pct)}`
                    : undefined
                }
              />
            )}
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
