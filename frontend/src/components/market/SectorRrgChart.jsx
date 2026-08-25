import { useMemo } from 'react'
import {
  ComposedChart,
  Scatter,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

// Cuarta auditoría, Bloque E: Relative Rotation Graph - fuerza relativa
// (RS-Ratio, eje X) cruzada con su propio ritmo de cambio (RS-Momentum, eje
// Y), en los cuatro cuadrantes clásicos. Colores por CUADRANTE, no por
// sector - con hasta 11 sectores, distinguir por color de sector obligaría a
// más matices de los que un ojo puede separar de un vistazo, y lo que
// importa aquí es justo el cuadrante (el nombre de cada sector ya se ve
// directamente etiquetado sobre su punto). Reutiliza los tonos categóricos
// que la app ya define, no una paleta nueva.
const QUADRANT_META = {
  leading: { label: 'Liderando', color: 'var(--series-good)' },
  weakening: { label: 'Debilitando', color: 'var(--cat-4)' },
  lagging: { label: 'Rezagado', color: 'var(--series-critical)' },
  improving: { label: 'Mejorando', color: 'var(--cat-1)' },
}
const QUADRANT_ORDER = ['leading', 'weakening', 'lagging', 'improving']

function pad(min, max) {
  const span = Math.max(max - min, 4)
  const margin = span * 0.25
  return [min - margin, max + margin]
}

function RrgTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  const meta = QUADRANT_META[point.quadrant]
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__date">
        {point.sector} ({point.etf})
      </p>
      <p className="chart-tooltip__row">
        <span className="chart-tooltip__swatch" style={{ background: meta.color }} />
        <span className="chart-tooltip__label">{meta.label}</span>
      </p>
      <p className="chart-tooltip__row">
        <span className="chart-tooltip__label">RS-Ratio</span>
        <span className="chart-tooltip__value">{point.rs_ratio.toFixed(1)}</span>
      </p>
      <p className="chart-tooltip__row">
        <span className="chart-tooltip__label">RS-Momentum</span>
        <span className="chart-tooltip__value">{point.rs_momentum.toFixed(1)}</span>
      </p>
      {point.as_of && (
        <p className="chart-tooltip__row">
          <span className="chart-tooltip__label">Fecha</span>
          <span className="chart-tooltip__value">{point.as_of}</span>
        </p>
      )}
    </div>
  )
}

// The tail is faint and thin (it's context, not the headline); only the
// current point (the last one in each sector's own tail) is a solid,
// labeled dot - the thing a reader actually scans the chart for.
function makeDotShape(color) {
  return function DotShape(props) {
    const { cx, cy, payload } = props
    if (!payload.isLatest) {
      return <circle cx={cx} cy={cy} r={2.5} fill={color} fillOpacity={0.35} stroke="none" />
    }
    return (
      <g>
        <circle cx={cx} cy={cy} r={5} fill={color} stroke="var(--surface)" strokeWidth={1.5} />
        <text x={cx + 8} y={cy + 4} fontSize={11} fill="var(--text-h)">
          {payload.sector}
        </text>
      </g>
    )
  }
}

function SectorRrgChart({ readings }) {
  const bySector = useMemo(() => {
    if (!readings?.length) return []
    return readings.map((reading) => ({
      reading,
      points: reading.tail.map((point, i) => ({
        x: point.rs_ratio,
        y: point.rs_momentum,
        as_of: point.as_of,
        sector: reading.sector,
        etf: reading.etf,
        quadrant: reading.quadrant,
        rs_ratio: point.rs_ratio,
        rs_momentum: point.rs_momentum,
        isLatest: i === reading.tail.length - 1,
      })),
    }))
  }, [readings])

  const domain = useMemo(() => {
    const allX = bySector.flatMap((s) => s.points.map((p) => p.x))
    const allY = bySector.flatMap((s) => s.points.map((p) => p.y))
    if (allX.length === 0) return { x: [90, 110], y: [90, 110] }
    return { x: pad(Math.min(...allX), Math.max(...allX)), y: pad(Math.min(...allY), Math.max(...allY)) }
  }, [bySector])

  if (!readings || readings.length === 0) {
    return (
      <p className="empty-state">
        Sin suficiente historial compartido con el benchmark para calcular el RRG todavía.
      </p>
    )
  }

  return (
    <div>
      <div className="viz-root">
        <ResponsiveContainer width="100%" height={440}>
          <ComposedChart margin={{ top: 16, right: 24, left: 0, bottom: 8 }}>
            <XAxis
              type="number"
              dataKey="x"
              domain={domain.x}
              tickFormatter={(v) => v.toFixed(0)}
              stroke="var(--chart-axis)"
              tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--chart-axis)' }}
              label={{ value: 'RS-Ratio (fuerza relativa)', position: 'insideBottom', offset: -4, fill: 'var(--chart-muted)', fontSize: 12 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={domain.y}
              tickFormatter={(v) => v.toFixed(0)}
              stroke="var(--chart-axis)"
              tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              width={40}
              label={{ value: 'RS-Momentum', angle: -90, position: 'insideLeft', fill: 'var(--chart-muted)', fontSize: 12 }}
            />
            <ReferenceLine x={100} stroke="var(--chart-axis)" strokeDasharray="4 3" />
            <ReferenceLine y={100} stroke="var(--chart-axis)" strokeDasharray="4 3" />
            <Tooltip content={<RrgTooltip />} cursor={{ strokeDasharray: '2 2' }} />
            {bySector.map(({ reading, points }) => (
              <Scatter
                key={reading.sector}
                data={points}
                line={{ stroke: QUADRANT_META[reading.quadrant].color, strokeWidth: 1, strokeOpacity: 0.5 }}
                shape={makeDotShape(QUADRANT_META[reading.quadrant].color)}
                isAnimationActive={false}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="rrg-legend">
        {QUADRANT_ORDER.map((quadrant) => (
          <span key={quadrant} className="rrg-legend__item">
            <span className="rrg-legend__swatch" style={{ background: QUADRANT_META[quadrant].color }} />
            {QUADRANT_META[quadrant].label}
          </span>
        ))}
      </div>
      <p className="chart-hint">
        La cola de cada punto muestra su trayectoria reciente - la rotación clásica avanza en sentido horario
        (liderando → debilitando → rezagado → mejorando → liderando de nuevo).
      </p>
    </div>
  )
}

export default SectorRrgChart
