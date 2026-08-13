// `imminentCross` is CoreSignalsResponse.imminent_cross (see
// technical_analysis.detect_imminent_cross on the backend) - a *projected*
// MA50/MA200 crossover, not yet confirmed. Shown separately from `ma_cross`
// (which only reports a cross that already happened) precisely because by
// the time a cross is confirmed, a real part of the move behind it has
// already played out - this is the earlier heads-up.
function ImminentCrossBadge({ imminentCross }) {
  if (!imminentCross) return null
  const isDeath = imminentCross.direction === 'death'
  const tone = isDeath ? 'warn' : 'up'
  const label = isDeath
    ? `Posible cruce de medias bajista en ~${imminentCross.bars_until} sesiones`
    : `Posible cruce de medias alcista en ~${imminentCross.bars_until} sesiones`

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
