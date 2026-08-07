import { Bar, Cell, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatAxisDate } from '../../../chartData'
import ChartTooltip from '../../ChartTooltip'

function RsiMacdChart({ data }) {
  return (
    <div className="viz-root rsi-macd-grid">
      <div>
        <p className="subchart-title">RSI (14)</p>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={data} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
            <XAxis dataKey="date" tickFormatter={formatAxisDate} hide />
            <YAxis
              domain={[0, 100]}
              ticks={[30, 50, 70]}
              stroke="var(--chart-axis)"
              tick={{ fill: 'var(--chart-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={32}
            />
            <ReferenceLine y={70} stroke="var(--series-critical)" strokeDasharray="3 3" />
            <ReferenceLine y={30} stroke="var(--series-good)" strokeDasharray="3 3" />
            <Tooltip
              content={<ChartTooltip rows={(payload) => payload.map((p) => ({ label: 'RSI', color: 'var(--series-1)', value: p.value?.toFixed(1) ?? '—' }))} />}
            />
            <Line
              type="monotone"
              dataKey="rsi14"
              stroke="var(--series-1)"
              dot={false}
              strokeWidth={1.5}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div>
        <p className="subchart-title">MACD (histograma)</p>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={data} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
            <XAxis dataKey="date" tickFormatter={formatAxisDate} hide />
            <YAxis
              stroke="var(--chart-axis)"
              tick={{ fill: 'var(--chart-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <ReferenceLine y={0} stroke="var(--chart-axis)" />
            <Tooltip
              content={<ChartTooltip rows={(payload) => payload.map((p) => ({ label: 'MACD hist.', color: p.value >= 0 ? 'var(--series-good)' : 'var(--series-critical)', value: p.value?.toFixed(2) ?? '—' }))} />}
            />
            <Bar dataKey="macd_histogram" isAnimationActive={false}>
              {data.map((row) => (
                <Cell key={row.date} fill={row.macd_histogram >= 0 ? 'var(--series-good)' : 'var(--series-critical)'} />
              ))}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default RsiMacdChart
