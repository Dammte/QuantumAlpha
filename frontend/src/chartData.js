/** Value series for the plain $ evolution chart. */
export function buildValueSeries(points) {
  return points.map((p) => ({ date: p.date, value: p.value }))
}

/** Indexes portfolio + benchmark to 0% at the first shared date, so both series can share
 * one axis (never dual-axis: two different-scale measures get indexed to a common base
 * instead, per the "one axis" rule). */
export function buildIndexedSeries(points, benchmarkPoints) {
  if (!points.length) return []
  const benchmarkByDate = new Map((benchmarkPoints ?? []).map((p) => [p.date, p.value]))

  const firstPortfolioValue = points[0].value
  const firstBenchmarkDate = points.find((p) => benchmarkByDate.has(p.date))?.date
  const firstBenchmarkValue = firstBenchmarkDate ? benchmarkByDate.get(firstBenchmarkDate) : null

  return points.map((p) => {
    const benchmarkValue = benchmarkByDate.get(p.date)
    return {
      date: p.date,
      portfolio: firstPortfolioValue ? (p.value / firstPortfolioValue - 1) * 100 : 0,
      benchmark:
        benchmarkValue !== undefined && firstBenchmarkValue
          ? (benchmarkValue / firstBenchmarkValue - 1) * 100
          : null,
    }
  })
}

/** Underwater/drawdown series (%) from a plain value series. */
export function buildDrawdownSeries(points) {
  let runningMax = -Infinity
  return points.map((p) => {
    runningMax = Math.max(runningMax, p.value)
    const drawdown = runningMax > 0 ? (p.value / runningMax - 1) * 100 : 0
    return { date: p.date, drawdown }
  })
}

export function formatAxisDate(dateStr) {
  const d = new Date(`${dateStr}T00:00:00`)
  return d.toLocaleDateString('es-ES', { day: '2-digit', month: 'short' })
}

export function formatFullDate(dateStr) {
  const d = new Date(`${dateStr}T00:00:00`)
  return d.toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' })
}
