// Fixed hue order backed by CSS custom properties (see App.css `.viz-root` block for the
// validated light/dark hex behind each slot — dataviz skill references/palette.md).
// Assigned by ticker identity (alphabetical), never by rank/weight, so a color never
// jumps between tickers as the portfolio's weights change.
const SLOTS = ['--cat-1', '--cat-2', '--cat-3', '--cat-4', '--cat-5', '--cat-6', '--cat-7', '--cat-8']
const OTHER = '--cat-other'

export function colorScaleFor(tickers) {
  const sorted = [...new Set(tickers)].sort()
  const scale = new Map()
  sorted.forEach((ticker, index) => {
    const varName = index < SLOTS.length ? SLOTS[index] : OTHER
    scale.set(ticker, { css: `var(${varName})` })
  })
  return scale
}
