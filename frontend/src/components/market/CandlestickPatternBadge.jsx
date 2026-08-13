// `pattern` is CoreSignalsResponse.candlestick_pattern (see
// technical_analysis.detect_engulfing_pattern on the backend) - a two-candle
// reversal pattern read off the last two *closed* sessions, informational
// only (never scored - see recommendation_engine.py's own factor list).
function CandlestickPatternBadge({ pattern }) {
  if (!pattern) return null
  const isBearish = pattern === 'bearish_engulfing'
  const tone = isBearish ? 'warn' : 'up'
  const label = isBearish ? 'Vela envolvente bajista (última sesión)' : 'Vela envolvente alcista (última sesión)'

  return (
    <p className="entry-timing">
      <span className={`sector-tier-badge sector-tier-badge--${tone}`}>
        {isBearish ? '⚠️ ' : ''}
        {label}
      </span>
    </p>
  )
}

export default CandlestickPatternBadge
