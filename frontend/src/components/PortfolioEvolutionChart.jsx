import { useMemo } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { buildIndexedSeries, buildValueSeries, formatAxisDate } from '../chartData'
import { formatCurrency, formatPercent } from '../format'
import ChartTooltip from './ChartTooltip'

function PortfolioEvolutionChart({ points, benchmarkPoints, benchmarkTicker, currency, compareMode, onToggleCompare }) {
  const valueData = useMemo(() => buildValueSeries(points), [points])
  const indexedData = useMemo(() => buildIndexedSeries(points, benchmarkPoints), [points, benchmarkPoints])

  const showCompare = compareMode && benchmarkTicker

  if (points.length === 0) {
    return <p className="empty-state">No hay histórico de precios para este periodo todavía.</p>
  }

  return (
    <div className="evolution-chart">
      <div className="evolution-chart__toolbar">
        <div className="legend-inline">
          <span className="legend-inline__item">
            <span className="ticker-swatch" style={{ '--swatch': 'var(--series-1)', '--swatch-dark': 'var(--series-1)' }} />
            Cartera
          </span>
          {showCompare && (
            <span className="legend-inline__item">
              <span className="ticker-swatch" style={{ '--swatch': 'var(--series-2)', '--swatch-dark': 'var(--series-2)' }} />
              {benchmarkTicker}
            </span>
          )}
        </div>
        {benchmarkTicker && (
          <button type="button" className="button-secondary button-secondary--pill" onClick={onToggleCompare}>
            {compareMode ? 'Ver valor ($)' : `Comparar vs ${benchmarkTicker}`}
          </button>
        )}
      </div>

      <div className="viz-root evolution-chart__plot">
        <ResponsiveContainer width="100%" height={320}>
          {showCompare ? (
            <LineChart data={indexedData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={formatAxisDate}
                stroke="var(--chart-axis)"
                tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: 'var(--chart-axis)' }}
                minTickGap={40}
              />
              <YAxis
                tickFormatter={(v) => `${v.toFixed(0)}%`}
                stroke="var(--chart-axis)"
                tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
                tickLine={false}
                axisLine={false}
                width={48}
              />
              <Tooltip
                cursor={{ stroke: 'var(--chart-axis)', strokeWidth: 1 }}
                content={
                  <ChartTooltip
                    rows={(payload) =>
                      payload
                        .filter((p) => p.value !== null && p.value !== undefined)
                        .map((p) => ({
                          label: p.dataKey === 'portfolio' ? 'Cartera' : benchmarkTicker,
                          color: p.dataKey === 'portfolio' ? 'var(--series-1)' : 'var(--series-2)',
                          value: formatPercent(p.value / 100, { signed: true }),
                        }))
                    }
                  />
                }
              />
              <Line type="monotone" dataKey="portfolio" stroke="var(--series-1)" strokeWidth={2} dot={false} />
              <Line
                type="monotone"
                dataKey="benchmark"
                stroke="var(--series-2)"
                strokeWidth={2}
                dot={false}
                strokeDasharray="4 3"
              />
            </LineChart>
          ) : (
            <AreaChart data={valueData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="portfolioValueGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--series-1)" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="var(--series-1)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={formatAxisDate}
                stroke="var(--chart-axis)"
                tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
                tickLine={false}
                axisLine={{ stroke: 'var(--chart-axis)' }}
                minTickGap={40}
              />
              <YAxis
                tickFormatter={(v) => formatCurrency(v, currency)}
                stroke="var(--chart-axis)"
                tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
                tickLine={false}
                axisLine={false}
                width={72}
                domain={['auto', 'auto']}
              />
              <Tooltip
                cursor={{ stroke: 'var(--chart-axis)', strokeWidth: 1 }}
                content={
                  <ChartTooltip
                    rows={(payload) =>
                      payload.map((p) => ({
                        label: 'Valor de la cartera',
                        color: 'var(--series-1)',
                        value: formatCurrency(p.value, currency),
                      }))
                    }
                  />
                }
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="var(--series-1)"
                strokeWidth={2}
                fill="url(#portfolioValueGradient)"
              />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default PortfolioEvolutionChart
