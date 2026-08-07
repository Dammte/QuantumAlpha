import { TREND_TONE, trendLabel } from '../../marketFormat'

function TrendBadge({ trend }) {
  const tone = TREND_TONE[trend] ?? 'neutral'
  return <span className={`trend-badge trend-badge--${tone}`}>{trendLabel(trend)}</span>
}

export default TrendBadge
