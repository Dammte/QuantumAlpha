import { formatCurrency, formatPercent, formatRatio } from '../../../format'
import CandlestickPatternBadge from '../CandlestickPatternBadge'
import ImminentCrossBadge from '../ImminentCrossBadge'

// 2026-09 (reconstruction, Fase 4): shows GateResult (levels_engine.py) - a
// boolean AND of 6 conditions, not the retired weighted checklist's
// comprar/esperar/evitar verdict + score. Every condition is always shown,
// passed or not (levels_engine.py's own docstring: "never collapsed to a
// bare yes/no") - there's no more "only show what triggered" filtering,
// since which condition(s) failed *is* the primary diagnostic value of a
// gate that didn't pass.

// Parte 7 (2026-09, later pass): `GateResult.entry_geometry` is the real
// stop-cascade/adaptive-risk-ceiling/net-of-costs design (trade_geometry.py)
// - strictly richer than `stop_and_target` below (which stays for engines
// that never compute it, e.g. a backtest replay). Shown whenever present,
// in place of the simpler block - `viable=False` is a normal, informative
// result on its own (the gate's simpler reward:risk check can pass while
// the real net-of-costs one doesn't), never hidden as if it didn't exist.
function EntryGeometryBlock({ geometry, currency }) {
  if (!geometry) return null
  if (!geometry.viable) {
    return (
      <div className="recommendation-card__levels">
        <p className="recommendation-card__level-hint">
          Geometría de entrada real (Parte 7) no viable: {geometry.rejection_reason}
        </p>
      </div>
    )
  }
  return (
    <div className="recommendation-card__levels">
      <div>
        <p className="recommendation-card__level-label">Stop-loss sugerido</p>
        <p className="recommendation-card__level-value delta-down">
          {formatCurrency(geometry.stop_price, currency)}
        </p>
        <p className="recommendation-card__level-hint">{geometry.stop_basis}</p>
      </div>
      <div>
        <p className="recommendation-card__level-label">Objetivo estimado</p>
        <p className="recommendation-card__level-value delta-up">
          {formatCurrency(geometry.target_price, currency)}
        </p>
        <p className="recommendation-card__level-hint">{geometry.target_basis}</p>
      </div>
      <div>
        <p className="recommendation-card__level-label">Ratio riesgo/beneficio (neto de costes)</p>
        <p className="recommendation-card__level-value">{formatRatio(geometry.risk_reward_net)} : 1</p>
      </div>
      {geometry.shares_for_risk_budget != null && (
        <div>
          <p className="recommendation-card__level-label">Tamaño sugerido</p>
          <p className="recommendation-card__level-value">
            {Math.floor(geometry.shares_for_risk_budget)} acc. · {formatCurrency(geometry.position_value, currency)}
          </p>
          <p className="recommendation-card__level-hint">{formatPercent(geometry.pct_of_portfolio)} de la cartera</p>
        </div>
      )}
    </div>
  )
}

function GateTrigger({ entryTrigger, currency }) {
  if (!entryTrigger) return null
  const label = entryTrigger.trigger_type === 'breakout' ? 'Disparador de ruptura' : 'Rebote en soporte'
  return (
    <div className="recommendation-card__trigger">
      <p className="recommendation-card__level-label">{label}</p>
      <p className="recommendation-card__level-value">
        {formatCurrency(entryTrigger.trigger_price, currency)}
        {entryTrigger.already_triggered && <span className="delta-up"> · ya disparado</span>}
      </p>
    </div>
  )
}

function RecommendationCard({
  gate,
  imminentCross,
  imminentCrossShortTerm,
  candlestickPattern,
  currency,
}) {
  const passingCount = gate.conditions.filter((c) => c.passed).length
  const tone = gate.passes ? 'up' : 'neutral'

  return (
    <div className={`recommendation-card recommendation-card--${tone}`}>
      <div className="recommendation-card__verdict">
        <span className={`recommendation-card__badge recommendation-card__badge--${tone}`}>
          {gate.passes ? 'GATE APROBADO' : 'GATE NO APROBADO'}
        </span>
        <span className="recommendation-card__score">
          {passingCount}/{gate.conditions.length} condiciones
        </span>
      </div>

      <ImminentCrossBadge imminentCross={imminentCross} gatePasses={gate.passes} />
      <ImminentCrossBadge imminentCross={imminentCrossShortTerm} shortTerm gatePasses={gate.passes} />
      <CandlestickPatternBadge pattern={candlestickPattern} />

      <GateTrigger entryTrigger={gate.entry_trigger} currency={currency} />

      {gate.passes && gate.entry_geometry && (
        <EntryGeometryBlock geometry={gate.entry_geometry} currency={currency} />
      )}
      {gate.passes && !gate.entry_geometry && gate.stop_and_target && (
        <div className="recommendation-card__levels">
          <div>
            <p className="recommendation-card__level-label">Stop-loss sugerido</p>
            <p className="recommendation-card__level-value delta-down">
              {formatCurrency(gate.stop_and_target.stop_loss, currency)}
            </p>
          </div>
          <div>
            <p className="recommendation-card__level-label">Objetivo estimado</p>
            <p className="recommendation-card__level-value delta-up">
              {formatCurrency(gate.stop_and_target.take_profit, currency)}
            </p>
            <p className="recommendation-card__level-hint">{gate.stop_and_target.take_profit_method}</p>
          </div>
          <div>
            <p className="recommendation-card__level-label">Ratio riesgo/beneficio</p>
            <p className="recommendation-card__level-value">{formatRatio(gate.stop_and_target.risk_reward)} : 1</p>
          </div>
        </div>
      )}

      <ul className="recommendation-card__factors">
        {gate.conditions.map((c) => (
          <li key={c.label} className={c.passed ? 'delta-up' : 'delta-down'}>
            <span>{c.passed ? '✓' : '✗'}</span> {c.label}
          </li>
        ))}
      </ul>
      <p className="recommendation-card__disclaimer">
        Gate transparente de reglas técnicas propias: tendencia o Fase 2 de Weinstein, sin extensión parabólica, sin
        sobrecompra extrema fuera de tendencia fuerte, sin divergencia bajista de volumen (OBV), sin veto del par
        rápido EMA21/55, y una relación beneficio:riesgo mínima - las 6 condiciones deben cumplirse a la vez, no una
        suma de puntos. RS Rating y Minervini se muestran aparte, como contexto, sin condicionar el gate. No es
        asesoramiento financiero.
      </p>
    </div>
  )
}

export default RecommendationCard
