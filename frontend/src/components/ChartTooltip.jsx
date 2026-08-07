import { formatFullDate } from '../chartData'

function ChartTooltip({ active, payload, label, rows, labelFormatter = formatFullDate }) {
  if (!active || !payload?.length) return null

  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__date">{labelFormatter(label)}</p>
      {rows(payload).map((row) => (
        <p key={row.label} className="chart-tooltip__row">
          <span className="chart-tooltip__swatch" style={{ background: row.color }} />
          <span className="chart-tooltip__label">{row.label}</span>
          <span className="chart-tooltip__value">{row.value}</span>
        </p>
      ))}
    </div>
  )
}

export default ChartTooltip
