import { useMemo } from 'react'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { formatCurrency, formatPercent } from '../format'
import ChartTooltip from './ChartTooltip'

function AllocationDonut({ positions, totalMarketValue, currency, colorScale }) {
  const segments = useMemo(
    () =>
      [...positions]
        // A position whose quote (or FX rate, for a non-base-currency ticker)
        // couldn't be fetched right now has no known base-currency value to
        // draw a slice for - excluded here rather than plotted as a misleading
        // $0 sliver or, worse, mixing unconverted currencies in one chart.
        .filter((position) => position.market_value_base !== null)
        .sort((a, b) => a.ticker.localeCompare(b.ticker))
        .map((position) => ({
          ticker: position.ticker,
          value: position.market_value_base,
          weight: totalMarketValue ? position.market_value_base / totalMarketValue : 0,
          color: colorScale.get(position.ticker),
        })),
    [positions, totalMarketValue, colorScale],
  )

  if (segments.length === 0 || totalMarketValue <= 0) return null

  return (
    <div className="allocation">
      <p className="allocation__title">Asignación por posición</p>
      <div className="allocation__layout">
        <div className="viz-root allocation__donut">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={segments}
                dataKey="value"
                nameKey="ticker"
                innerRadius={62}
                outerRadius={92}
                paddingAngle={2}
                stroke="var(--surface-1)"
                strokeWidth={2}
              >
                {segments.map((segment) => (
                  <Cell key={segment.ticker} fill={segment.color.css} />
                ))}
              </Pie>
              <Tooltip
                content={
                  <ChartTooltip
                    rows={(payload) =>
                      payload.map((p) => ({
                        label: p.payload.ticker,
                        color: p.payload.color.css,
                        value: `${formatCurrency(p.payload.value, currency)} (${formatPercent(p.payload.weight)})`,
                      }))
                    }
                  />
                }
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="allocation__donut-center">
            <span className="allocation__donut-total">{formatCurrency(totalMarketValue, currency)}</span>
            <span className="allocation__donut-label">Total</span>
          </div>
        </div>
        <ul className="allocation__legend">
          {segments.map((segment) => (
            <li key={segment.ticker}>
              <span className="ticker-swatch" style={{ '--swatch': segment.color.css, '--swatch-dark': segment.color.css }} />
              <span className="allocation__legend-ticker">{segment.ticker}</span>
              <span className="allocation__legend-weight">{formatPercent(segment.weight)}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default AllocationDonut
