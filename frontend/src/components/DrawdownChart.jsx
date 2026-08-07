import { useMemo } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { buildDrawdownSeries, formatAxisDate } from '../chartData'
import { formatPercent } from '../format'
import ChartTooltip from './ChartTooltip'

function DrawdownChart({ points }) {
  const data = useMemo(() => buildDrawdownSeries(points), [points])

  if (data.length === 0) return null

  return (
    <div className="viz-root">
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={data} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--series-critical)" stopOpacity={0.05} />
              <stop offset="100%" stopColor="var(--series-critical)" stopOpacity={0.4} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={formatAxisDate}
            stroke="var(--chart-axis)"
            tick={{ fill: 'var(--chart-muted)', fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: 'var(--chart-axis)' }}
            minTickGap={40}
          />
          <YAxis
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            stroke="var(--chart-axis)"
            tick={{ fill: 'var(--chart-muted)', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={44}
            domain={['dataMin', 0]}
          />
          <Tooltip
            cursor={{ stroke: 'var(--chart-axis)', strokeWidth: 1 }}
            content={
              <ChartTooltip
                rows={(payload) =>
                  payload.map((p) => ({
                    label: 'Drawdown',
                    color: 'var(--series-critical)',
                    value: formatPercent(p.value / 100),
                  }))
                }
              />
            }
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="var(--series-critical)"
            strokeWidth={1.5}
            fill="url(#drawdownGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export default DrawdownChart
