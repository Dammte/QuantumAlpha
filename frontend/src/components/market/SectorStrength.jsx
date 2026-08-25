import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../../api'
import { formatPercent } from '../../format'
import ChartTooltip from '../ChartTooltip'

const PERIODS = [
  { key: 'change_1d', label: '1D' },
  { key: 'change_1w', label: '1S' },
  { key: 'change_1m', label: '1M' },
  { key: 'change_3m', label: '3M' },
  { key: 'change_6m', label: '6M' },
  { key: 'change_1y', label: '1A' },
]

function SectorStrength({ region }) {
  const [sectors, setSectors] = useState([])
  const [period, setPeriod] = useState('change_1m')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setSectors(await api.getSectorPerformance({ region }))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  const data = useMemo(
    () =>
      [...sectors]
        .filter((s) => s[period] !== null && s[period] !== undefined)
        .sort((a, b) => {
          // Tercera auditoría, Bloque H: "fuerza relativa" debe ordenar por
          // fuerza relativa real (rs_rank - el mismo percentil de periodos
          // combinados que decide qué sector lidera en el resto de la app),
          // no por el retorno absoluto del periodo elegido en los botones -
          // los dos pueden divergir (un sector +5% en un mercado +8% es un
          // rezagado en RS aunque su barra absoluta sea positiva). El valor
          // de cada barra sigue siendo el retorno del periodo elegido; solo
          // el ORDEN cambia. Sin rs_rank en ninguno de los dos (menos de 252
          // sesiones de historia), se cae de vuelta al retorno absoluto en
          // vez de un orden arbitrario.
          const aHasRank = a.rs_rank !== null && a.rs_rank !== undefined
          const bHasRank = b.rs_rank !== null && b.rs_rank !== undefined
          if (aHasRank !== bHasRank) return aHasRank ? -1 : 1
          if (aHasRank && bHasRank) return b.rs_rank - a.rs_rank
          return b[period] - a[period]
        })
        .map((s) => ({ sector: s.sector, etf: s.etf, value: s[period] * 100, rsRank: s.rs_rank })),
    [sectors, period],
  )

  if (loading) return <p className="empty-state">Cargando sectores…</p>
  if (error) return <div className="banner banner--error">{error}</div>

  return (
    <div>
      <p className="empty-state" style={{ marginBottom: 12 }}>
        Ordenado por fuerza relativa (RS, percentil combinando varios periodos - visible en el tooltip de cada
        barra), no por el retorno absoluto del periodo que elijas abajo: pueden divergir, y RS es lo que el resto
        de la app usa para decidir qué sector lidera.
      </p>
      <div className="filters-row">
        {PERIODS.map((p) => (
          <button
            key={p.key}
            type="button"
            className={`timeframe-tab ${period === p.key ? 'timeframe-tab--active' : ''}`}
            onClick={() => setPeriod(p.key)}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="viz-root">
        <ResponsiveContainer width="100%" height={Math.max(280, data.length * 34)}>
          <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
            <XAxis
              type="number"
              tickFormatter={(v) => `${v.toFixed(0)}%`}
              stroke="var(--chart-axis)"
              tick={{ fill: 'var(--chart-muted)', fontSize: 12 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="category"
              dataKey="sector"
              width={150}
              stroke="var(--chart-axis)"
              tick={{ fill: 'var(--text-h)', fontSize: 13 }}
              tickLine={false}
              axisLine={false}
            />
            <ReferenceLine x={0} stroke="var(--chart-axis)" />
            <Tooltip
              cursor={{ fill: 'var(--chart-grid)' }}
              content={
                <ChartTooltip
                  rows={(payload) =>
                    payload.map((p) => ({
                      label: `${p.payload.sector} (${p.payload.etf})`,
                      color: p.value >= 0 ? 'var(--series-1)' : 'var(--series-critical)',
                      value: `${formatPercent(p.value / 100, { signed: true })}${
                        p.payload.rsRank !== null && p.payload.rsRank !== undefined ? ` · RS ${p.payload.rsRank}` : ''
                      }`,
                    }))
                  }
                />
              }
            />
            <Bar dataKey="value" radius={[4, 4, 4, 4]}>
              {data.map((row) => (
                <Cell key={row.sector} fill={row.value >= 0 ? 'var(--series-1)' : 'var(--series-critical)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default SectorStrength
