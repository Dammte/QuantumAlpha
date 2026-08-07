export const TREND_LABELS = {
  uptrend: 'Alcista',
  downtrend: 'Bajista',
  sideways: 'Lateral',
}

export const TREND_TONE = {
  uptrend: 'up',
  downtrend: 'down',
  sideways: 'neutral',
}

export function trendLabel(trend) {
  return TREND_LABELS[trend] ?? trend
}
