// `imminentCross` is CoreSignalsResponse.imminent_cross / .imminent_cross_short_term
// (see technical_analysis.detect_imminent_cross on the backend) - a *projected*
// MA crossover, not yet confirmed. Shown separately from `ma_cross` (which only
// reports a cross that already happened) precisely because by the time a cross
// is confirmed, a real part of the move behind it has already played out - this
// is the earlier heads-up. `shortTerm` switches the wording/pair from
// SMA50/SMA200 (medium/long-term) to SMA20/SMA50 (short-term) - same underlying
// projection, different moving-average pair, relevant for a position managed on
// a shorter horizon.
function ImminentCrossBadge({ imminentCross, shortTerm = false }) {
  if (!imminentCross) return null
  const isDeath = imminentCross.direction === 'death'
  const tone = isDeath ? 'warn' : 'up'
  const pairLabel = shortTerm ? 'corto plazo (SMA20/SMA50)' : 'SMA50/SMA200'
  const label = isDeath
    ? `Posible cruce bajista de ${pairLabel} en ~${imminentCross.bars_until} sesiones`
    : `Posible cruce alcista de ${pairLabel} en ~${imminentCross.bars_until} sesiones`

  return (
    <p className="entry-timing">
      <span className={`sector-tier-badge sector-tier-badge--${tone}`} title="Proyección, no una confirmación">
        {isDeath ? '⚠️ ' : ''}
        {label}
      </span>
    </p>
  )
}

export default ImminentCrossBadge
