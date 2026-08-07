import { useMemo } from 'react'
import { Bar, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatAxisDate } from '../../../chartData'
import ChartTooltip from '../../ChartTooltip'

const SMA_WINDOW = 20

function formatVolume(v) {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`
  return `${v}`
}

// Cheap rolling-sum SMA (no per-point Cell coloring - with 500+ bars, one
// React element per data point is a real rendering cost, so volume spikes
// are left to the existing "Volumen relativo" stat tile instead).
function withVolumeSma(data) {
  let sum = 0
  return data.map((row, i) => {
    sum += row.volume
    if (i >= SMA_WINDOW) sum -= data[i - SMA_WINDOW].volume
    const count = Math.min(i + 1, SMA_WINDOW)
    return { ...row, volumeSma: i >= SMA_WINDOW - 1 ? sum / count : null }
  })
}

function VolumeChart({ data }) {
  const chartData = useMemo(() => withVolumeSma(data), [data])

  return (
    <div className="viz-root">
      <p className="subchart-title">Volumen (línea: media 20 días)</p>
      <ResponsiveContainer width="100%" height={100}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
          <XAxis dataKey="date" tickFormatter={formatAxisDate} hide />
          <YAxis
            tickFormatter={formatVolume}
            stroke="var(--chart-axis)"
            tick={{ fill: 'var(--chart-muted)', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
          />
          <Tooltip
            cursor={{ fill: 'var(--chart-grid)' }}
            content={
              <ChartTooltip
                rows={(payload) =>
                  payload
                    .filter((p) => p.value !== null && p.value !== undefined)
                    .map((p) => ({
                      label: p.dataKey === 'volumeSma' ? 'Media 20 días' : 'Volumen',
                      color: p.dataKey === 'volumeSma' ? 'var(--chart-muted)' : 'var(--series-1)',
                      value: formatVolume(p.value),
                    }))
                }
              />
            }
          />
          <Bar dataKey="volume" fill="var(--series-1)" opacity={0.6} isAnimationActive={false} />
          <Line
            type="monotone"
            dataKey="volumeSma"
            stroke="var(--chart-muted)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export default VolumeChart
