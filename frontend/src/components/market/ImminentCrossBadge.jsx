// `imminentCross` is CoreSignalsResponse.imminent_cross / .imminent_cross_short_term
// (see technical_analysis.detect_imminent_cross on the backend) - a *projected*
// MA crossover, not yet confirmed. Shown separately from `ma_cross` (which only
// reports a cross that already happened) precisely because by the time a cross
// is confirmed, a real part of the move behind it has already played out - this
// is the earlier heads-up. `shortTerm` switches the wording/pair from
// SMA50/SMA200 (medium/long-term) to SMA21/SMA50 (short-term) - same underlying
// projection, different moving-average pair, relevant for a position managed on
// a shorter horizon.
//
// Mirrors exit_engine.py's own confidence bar for this *exact* signal -
// IMMINENT_CROSS_50_200_MIN_R2/IMMINENT_CROSS_20_50_MIN_R2 - so the badge can
// honestly say whether this projection already clears the bar the exit engine
// itself would act on for a held position, not just repeat the raw number.
// This never affects the buy-side score shown alongside it (see D2 in
// docs/quant_methodology.md: a projected cross is an exit-engine trigger, not
// a scored factor) - the badge says so explicitly rather than implying it does.
const MIN_R2_BY_TERM = { long: 0.7, short: 0.6 }

function ImminentCrossBadge({ imminentCross, shortTerm = false }) {
  if (!imminentCross) return null
  const isDeath = imminentCross.direction === 'death'
  const tone = isDeath ? 'warn' : 'up'
  const pairLabel = shortTerm ? 'corto plazo (SMA21/SMA50)' : 'SMA50/SMA200'
  const label = isDeath
    ? `Posible cruce bajista de ${pairLabel} en ~${imminentCross.bars_until} sesiones`
    : `Posible cruce alcista de ${pairLabel} en ~${imminentCross.bars_until} sesiones`

  const minR2 = MIN_R2_BY_TERM[shortTerm ? 'short' : 'long']
  const clearsActionBar = imminentCross.r_squared >= minR2
  const confidenceHint = clearsActionBar
    ? `confianza suficiente (R²=${imminentCross.r_squared.toFixed(2)}) para que el motor de salida actuase si ya mantuvieras esta posición`
    : `confianza todavía baja (R²=${imminentCross.r_squared.toFixed(2)}, por debajo de ${minR2.toFixed(2)}) - proyección a vigilar, no a actuar todavía`

  return (
    <p className="entry-timing imminent-cross-badge">
      <span
        className={`sector-tier-badge sector-tier-badge--${tone}`}
        title={`Proyección, no una confirmación - no afecta a la puntuación de compra. ${confidenceHint}.`}
      >
        {isDeath ? '⚠️ ' : ''}
        {label}
        <span className="imminent-cross-badge__r2"> (R²={imminentCross.r_squared.toFixed(2)})</span>
      </span>
      {clearsActionBar && (
        <span className="imminent-cross-badge__action-hint" title={confidenceHint}>
          {isDeath ? '· ya bastaría para subir el stop en una posición abierta' : '· proyección ya fiable'}
        </span>
      )}
    </p>
  )
}

export default ImminentCrossBadge
