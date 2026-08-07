import { useMemo } from 'react'
import { Area, CartesianGrid, Line, ComposedChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatCurrency } from '../../../format'
import ChartTooltip from '../../ChartTooltip'

const SERIES = [
  { key: 'close', label: 'Precio', color: 'var(--series-1)', width: 2, dash: undefined },
  { key: 'sma50', label: 'MA50', color: 'var(--cat-2)', width: 1.5, dash: undefined },
  { key: 'sma200', label: 'MA200', color: 'var(--cat-4)', width: 1.5, dash: undefined },
  { key: 'bb_upper', label: 'Bollinger sup.', color: 'var(--chart-muted)', width: 1, dash: '3 3' },
  { key: 'bb_lower', label: 'Bollinger inf.', color: 'var(--chart-muted)', width: 1, dash: '3 3' },
  { key: 'gann_1x1', label: 'Gann 1x1', color: 'var(--accent)', width: 1.5, dash: '4 2' },
  { key: 'forecastMedian', label: 'Mediana proyectada (Monte Carlo)', color: 'var(--series-1)', width: 2, dash: '2 3' },
]

// Fields the Y-axis should actually scale to. Gann fan lines are deliberately
// excluded: they diverge to extreme levels over a long time span by design (that's
// how a Gann fan works), and letting the axis stretch to fit them would squash all
// the real price action into a sliver at the bottom of the chart.
const PRICE_RELEVANT_FIELDS = ['close', 'sma50', 'sma200', 'bb_upper', 'bb_lower']

function toTs(dateStr) {
  return new Date(`${dateStr}T00:00:00`).getTime()
}

function formatAxisTs(ts) {
  return new Date(ts).toLocaleDateString('es-ES', { day: '2-digit', month: 'short' })
}

function formatFullTs(ts) {
  return new Date(ts).toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' })
}

function priceDomain(data, nearestSupport, nearestResistance, monteCarlo) {
  const values = data.flatMap((row) => PRICE_RELEVANT_FIELDS.map((f) => row[f]).filter((v) => v !== null && v !== undefined))
  if (nearestSupport) values.push(nearestSupport.price)
  if (nearestResistance) values.push(nearestResistance.price)
  if (monteCarlo) {
    for (const p of monteCarlo.percentiles) values.push(p.p5, p.p95)
  }
  if (values.length === 0) return ['auto', 'auto']
  const min = Math.min(...values)
  const max = Math.max(...values)
  const padding = (max - min) * 0.08
  return [min - padding, max + padding]
}

// The Monte Carlo horizon is expressed in trading days; converting to calendar
// days (roughly 7/5 ratio) is only meant to place the projection at a plausible
// future date on a real time-based axis shared with history, not a precise
// trading calendar (holidays aren't modeled).
function tradingToCalendarDays(tradingDays) {
  return Math.round((tradingDays * 7) / 5)
}

function buildForecastRows(lastTs, currentPrice, percentiles) {
  const dayMs = 24 * 60 * 60 * 1000
  const anchor = {
    ts: lastTs,
    forecastBase: currentPrice,
    forecastLower: 0,
    forecastMid: 0,
    forecastUpper: 0,
    forecastMedian: currentPrice,
  }
  const rows = percentiles.map((p) => ({
    ts: lastTs + tradingToCalendarDays(p.day) * dayMs,
    forecastBase: p.p5,
    forecastLower: p.p25 - p.p5,
    forecastMid: p.p75 - p.p25,
    forecastUpper: p.p95 - p.p75,
    forecastMedian: p.p50,
  }))
  return [anchor, ...rows]
}

function buildChartData(data, monteCarlo, currentPrice) {
  const historical = data.map((row) => ({
    ...row,
    ts: toTs(row.date),
    forecastBase: null,
    forecastLower: null,
    forecastMid: null,
    forecastUpper: null,
    forecastMedian: null,
  }))
  if (!monteCarlo || !historical.length) return historical

  const lastTs = historical[historical.length - 1].ts
  const forecastRows = buildForecastRows(lastTs, currentPrice, monteCarlo.percentiles)
  historical[historical.length - 1] = { ...historical[historical.length - 1], ...forecastRows[0] }
  return [...historical, ...forecastRows.slice(1)]
}

function PriceChart({ data, currency, nearestSupport, nearestResistance, monteCarlo, currentPrice }) {
  const chartData = useMemo(() => buildChartData(data, monteCarlo, currentPrice), [data, monteCarlo, currentPrice])
  const domain = useMemo(
    () => priceDomain(chartData, nearestSupport, nearestResistance, monteCarlo),
    [chartData, nearestSupport, nearestResistance, monteCarlo]
  )
  const lastHistoricalTs = data.length ? toTs(data[data.length - 1].date) : null

  return (
    <div className="viz-root">
      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
          <XAxis
            dataKey="ts"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={formatAxisTs}
            stroke="var(--chart-axis)"
            tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: 'var(--chart-axis)' }}
            minTickGap={50}
          />
          <YAxis
            domain={domain}
            allowDataOverflow
            tickFormatter={(v) => formatCurrency(v, currency)}
            stroke="var(--chart-axis)"
            tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={72}
          />
          <Tooltip
            cursor={{ stroke: 'var(--chart-axis)', strokeWidth: 1 }}
            content={
              <ChartTooltip
                labelFormatter={formatFullTs}
                rows={(payload) =>
                  payload
                    .filter((p) => p.value !== null && p.value !== undefined)
                    .filter((p) => !['forecastBase', 'forecastLower', 'forecastMid', 'forecastUpper'].includes(p.dataKey))
                    .map((p) => {
                      const series = SERIES.find((s) => s.key === p.dataKey)
                      return {
                        label: series?.label ?? p.dataKey,
                        color: series?.color ?? p.color,
                        value: formatCurrency(p.value, currency),
                      }
                    })
                }
              />
            }
          />
          {nearestSupport && (
            <ReferenceLine
              y={nearestSupport.price}
              stroke="var(--series-good)"
              strokeDasharray="4 3"
              label={{ value: `Soporte ${formatCurrency(nearestSupport.price, currency)}`, fill: 'var(--series-good)', fontSize: 11, position: 'insideBottomLeft' }}
            />
          )}
          {nearestResistance && (
            <ReferenceLine
              y={nearestResistance.price}
              stroke="var(--series-critical)"
              strokeDasharray="4 3"
              label={{ value: `Resistencia ${formatCurrency(nearestResistance.price, currency)}`, fill: 'var(--series-critical)', fontSize: 11, position: 'insideTopLeft' }}
            />
          )}
          {monteCarlo && lastHistoricalTs && (
            <ReferenceLine
              x={lastHistoricalTs}
              stroke="var(--chart-axis)"
              strokeDasharray="2 2"
              label={{ value: 'Hoy →', fill: 'var(--chart-muted)', fontSize: 11, position: 'insideTopLeft' }}
            />
          )}
          {monteCarlo && (
            <>
              <Area dataKey="forecastBase" stackId="forecast" stroke="none" fill="transparent" isAnimationActive={false} />
              <Area
                dataKey="forecastLower"
                stackId="forecast"
                stroke="none"
                fill="var(--series-1)"
                fillOpacity={0.12}
                isAnimationActive={false}
              />
              <Area
                dataKey="forecastMid"
                stackId="forecast"
                stroke="none"
                fill="var(--series-1)"
                fillOpacity={0.25}
                isAnimationActive={false}
              />
              <Area
                dataKey="forecastUpper"
                stackId="forecast"
                stroke="none"
                fill="var(--series-1)"
                fillOpacity={0.12}
                isAnimationActive={false}
              />
            </>
          )}
          {SERIES.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={s.color}
              strokeWidth={s.width}
              strokeDasharray={s.dash}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
      <ul className="legend-inline chart-legend-wrap">
        {SERIES.filter((s) => s.key !== 'forecastMedian' || monteCarlo).map((s) => (
          <li key={s.key} className="legend-inline__item">
            <span className="ticker-swatch" style={{ '--swatch': s.color }} />
            {s.label}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default PriceChart
