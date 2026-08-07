import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatPercent } from '../../../format'
import ChartTooltip from '../../ChartTooltip'

const MONTH_LABELS = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
const CURRENT_MONTH = new Date().getMonth() + 1

function SeasonalityChart({ seasonality }) {
  const data = seasonality.map((s) => ({
    month: MONTH_LABELS[s.month - 1],
    avg_return: s.avg_return * 100,
    win_rate: s.win_rate,
    n: s.n_observations,
    isCurrent: s.month === CURRENT_MONTH,
  }))

  return (
    <div className="viz-root">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="month"
            stroke="var(--chart-axis)"
            tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: 'var(--chart-axis)' }}
          />
          <YAxis
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            stroke="var(--chart-axis)"
            tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          <ReferenceLine y={0} stroke="var(--chart-axis)" />
          <Tooltip
            cursor={{ fill: 'var(--chart-grid)' }}
            content={
              <ChartTooltip
                labelFormatter={(label) => label}
                rows={(payload) =>
                  payload.map((p) => ({
                    label: `Retorno medio (${p.payload.n} años) · win rate ${(p.payload.win_rate * 100).toFixed(0)}%${p.payload.isCurrent ? ' · mes actual' : ''}`,
                    color: p.value >= 0 ? 'var(--series-1)' : 'var(--series-critical)',
                    value: formatPercent(p.value / 100, { signed: true }),
                  }))
                }
              />
            }
          />
          <Bar dataKey="avg_return" radius={[3, 3, 3, 3]} isAnimationActive={false}>
            {data.map((row) => (
              <Cell
                key={row.month}
                fill={row.avg_return >= 0 ? 'var(--series-1)' : 'var(--series-critical)'}
                stroke={row.isCurrent ? 'var(--accent)' : 'none'}
                strokeWidth={row.isCurrent ? 2 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="seasonality-chart__hint">
        <span className="seasonality-chart__current-swatch" /> mes actual ({MONTH_LABELS[CURRENT_MONTH - 1]})
      </p>
    </div>
  )
}

export default SeasonalityChart
