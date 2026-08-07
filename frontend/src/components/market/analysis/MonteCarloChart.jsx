import { useMemo } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatCurrency, formatPercent } from '../../../format'
import StatTile from '../../StatTile'

function buildFanData(percentiles, currentPrice) {
  const rows = [
    {
      day: 0,
      base: currentPrice,
      lowerBand: 0,
      midBand: 0,
      upperBand: 0,
      p50: currentPrice,
      raw: { p5: currentPrice, p25: currentPrice, p50: currentPrice, p75: currentPrice, p95: currentPrice },
    },
  ]
  for (const p of percentiles) {
    rows.push({
      day: p.day,
      base: p.p5,
      lowerBand: p.p25 - p.p5,
      midBand: p.p75 - p.p25,
      upperBand: p.p95 - p.p75,
      p50: p.p50,
      raw: p,
    })
  }
  return rows
}

function fanDomain(percentiles, currentPrice, stopLoss, takeProfit) {
  const values = percentiles.flatMap((p) => [p.p5, p.p95])
  values.push(currentPrice)
  if (stopLoss) values.push(stopLoss)
  if (takeProfit) values.push(takeProfit)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const padding = (max - min) * 0.08
  return [min - padding, max + padding]
}

function FanTooltip({ active, payload, currency }) {
  if (!active || !payload?.length) return null
  const raw = payload[0]?.payload?.raw
  if (!raw) return null
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__date">Percentil 95: {formatCurrency(raw.p95, currency)}</p>
      <p className="chart-tooltip__row">
        <span className="chart-tooltip__label">Percentil 75</span>
        <span className="chart-tooltip__value">{formatCurrency(raw.p75, currency)}</span>
      </p>
      <p className="chart-tooltip__row">
        <span className="chart-tooltip__label">Mediana</span>
        <span className="chart-tooltip__value">{formatCurrency(raw.p50, currency)}</span>
      </p>
      <p className="chart-tooltip__row">
        <span className="chart-tooltip__label">Percentil 25</span>
        <span className="chart-tooltip__value">{formatCurrency(raw.p25, currency)}</span>
      </p>
      <p className="chart-tooltip__row">
        <span className="chart-tooltip__label">Percentil 5</span>
        <span className="chart-tooltip__value">{formatCurrency(raw.p5, currency)}</span>
      </p>
    </div>
  )
}

function MonteCarloChart({ monteCarlo, currentPrice, currency, stopLoss, takeProfit }) {
  const data = useMemo(
    () => (monteCarlo ? buildFanData(monteCarlo.percentiles, currentPrice) : []),
    [monteCarlo, currentPrice]
  )
  const domain = useMemo(
    () => (monteCarlo ? fanDomain(monteCarlo.percentiles, currentPrice, stopLoss, takeProfit) : ['auto', 'auto']),
    [monteCarlo, currentPrice, stopLoss, takeProfit]
  )

  if (!monteCarlo) {
    return (
      <p className="empty-state">
        No hay suficiente histórico (se necesita más de un año de datos) para simular trayectorias de precio.
      </p>
    )
  }

  const hasBarriers = monteCarlo.probability_stop_before_target !== null

  return (
    <div>
      <div className="viz-root">
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
            <XAxis
              dataKey="day"
              tickFormatter={(d) => `${d}d`}
              stroke="var(--chart-axis)"
              tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--chart-axis)' }}
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
            <Tooltip content={<FanTooltip currency={currency} />} />
            {stopLoss && (
              <ReferenceLine
                y={stopLoss}
                stroke="var(--series-critical)"
                strokeDasharray="4 3"
                label={{ value: 'Stop', fill: 'var(--series-critical)', fontSize: 11, position: 'insideBottomLeft' }}
              />
            )}
            {takeProfit && (
              <ReferenceLine
                y={takeProfit}
                stroke="var(--series-good)"
                strokeDasharray="4 3"
                label={{ value: 'Objetivo', fill: 'var(--series-good)', fontSize: 11, position: 'insideTopLeft' }}
              />
            )}
            <Area type="monotone" dataKey="base" stackId="fan" stroke="none" fill="transparent" isAnimationActive={false} />
            <Area
              type="monotone"
              dataKey="lowerBand"
              stackId="fan"
              stroke="none"
              fill="var(--series-1)"
              fillOpacity={0.15}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="midBand"
              stackId="fan"
              stroke="none"
              fill="var(--series-1)"
              fillOpacity={0.35}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="upperBand"
              stackId="fan"
              stroke="none"
              fill="var(--series-1)"
              fillOpacity={0.15}
              isAnimationActive={false}
            />
            <Line type="monotone" dataKey="p50" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="monte-carlo__legend-hint">
        Banda oscura: rango del 50% de los escenarios simulados. Banda clara: rango del 90%. Línea: mediana. Método:{' '}
        {monteCarlo.method === 'garch_filtered' ? 'bootstrap filtrado por GARCH' : 'bootstrap por bloques'},{' '}
        {monteCarlo.n_simulations.toLocaleString('es')} simulaciones.
      </p>

      <div className="stat-grid" style={{ marginTop: 14 }}>
        <StatTile
          label="Probabilidad de pérdida al horizonte"
          value={formatPercent(monteCarlo.probability_of_loss)}
          tone={monteCarlo.probability_of_loss > 0.5 ? 'down' : 'neutral'}
        />
        {hasBarriers && (
          <>
            <StatTile
              label="Toca el stop-loss primero"
              value={formatPercent(monteCarlo.probability_stop_before_target)}
              tone="down"
            />
            <StatTile
              label="Toca el objetivo primero"
              value={formatPercent(monteCarlo.probability_target_before_stop)}
              tone="up"
            />
            <StatTile
              label="No toca ninguno en el horizonte"
              value={formatPercent(monteCarlo.probability_neither_hit)}
            />
          </>
        )}
      </div>
    </div>
  )
}

export default MonteCarloChart
