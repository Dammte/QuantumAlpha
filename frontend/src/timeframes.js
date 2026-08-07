const toInputDate = (d) => d.toISOString().slice(0, 10)

function monthsAgo(n, from) {
  const d = new Date(from)
  d.setMonth(d.getMonth() - n)
  return d
}

function daysAgo(n, from) {
  const d = new Date(from)
  d.setDate(d.getDate() - n)
  return d
}

export const TIMEFRAMES = [
  { key: '1W', label: '1S', start: (today) => daysAgo(7, today) },
  { key: '1M', label: '1M', start: (today) => monthsAgo(1, today) },
  { key: '3M', label: '3M', start: (today) => monthsAgo(3, today) },
  { key: '6M', label: '6M', start: (today) => monthsAgo(6, today) },
  {
    key: 'YTD',
    label: 'YTD',
    start: (today) => new Date(today.getFullYear(), 0, 1),
  },
  { key: '1Y', label: '1A', start: (today) => monthsAgo(12, today) },
  { key: '3Y', label: '3A', start: (today) => monthsAgo(36, today) },
  { key: 'ALL', label: 'Todo', start: () => new Date(2000, 0, 1) },
]

const DEFAULT_TIMEFRAME_KEY = '6M'

export function rangeFor(timeframeKey, today = new Date()) {
  const timeframe =
    TIMEFRAMES.find((t) => t.key === timeframeKey) ?? TIMEFRAMES.find((t) => t.key === DEFAULT_TIMEFRAME_KEY)
  return { start: toInputDate(timeframe.start(today)), end: toInputDate(today) }
}
