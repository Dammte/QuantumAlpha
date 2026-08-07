import { useEffect, useState } from 'react'
import { api } from '../../api'
import StatTile from '../StatTile'
import { formatPercent } from '../../format'
import TickerSnapshotTable from './TickerSnapshotTable'

const DETAIL_GROUPS = [
  { key: 'uptrend', title: 'En tendencia alcista', columns: ['price', 'change_1m', 'rsi14', 'rs_rating', 'trend'] },
  { key: 'downtrend', title: 'En tendencia bajista', columns: ['price', 'change_1m', 'rsi14', 'rs_rating', 'trend'] },
  { key: 'stage2', title: 'Fase 2 de Weinstein (avance)', columns: ['price', 'change_1m', 'rs_rating', 'trend'] },
  { key: 'stage4', title: 'Fase 4 de Weinstein (declive)', columns: ['price', 'change_1m', 'rs_rating', 'trend'] },
  { key: 'golden_cross', title: 'Golden cross reciente (MA50/MA200)', columns: ['price', 'change_1m', 'rs_rating'] },
  { key: 'death_cross', title: 'Death cross reciente (MA50/MA200)', columns: ['price', 'change_1m', 'rs_rating'] },
  { key: 'overbought', title: 'Sobrecompra (RSI ≥ 70)', columns: ['price', 'rsi14', 'change_1m'] },
  { key: 'oversold', title: 'Sobreventa (RSI ≤ 30)', columns: ['price', 'rsi14', 'change_1m'] },
  {
    key: 'minervini_pass',
    title: 'Cumplen el Trend Template de Minervini (8/8)',
    columns: ['price', 'change_1m', 'rs_rating'],
  },
  { key: 'strong_trend', title: 'Tendencia fuerte confirmada (ADX ≥ 25)', columns: ['price', 'change_1m', 'atr_multiple'] },
]

function TrendBreadthPanel({ region }) {
  const [breadth, setBreadth] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const [breadthData, detailData] = await Promise.all([
          api.getMarketTrend({ region }),
          api.getMarketTrendDetail({ region }),
        ])
        setBreadth(breadthData)
        setDetail(detailData)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  if (loading) return <p className="empty-state">Calculando amplitud de mercado…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (!breadth || !detail) return null

  return (
    <div>
      <div className="stat-grid" style={{ marginBottom: 28 }}>
        <StatTile label="Universo analizado" value={breadth.total} hint="tickers con datos suficientes" />
        <StatTile
          label="% sobre MA50"
          value={formatPercent(breadth.pct_above_sma50)}
          tone={breadth.pct_above_sma50 >= 0.5 ? 'up' : 'down'}
        />
        <StatTile
          label="% sobre MA200"
          value={formatPercent(breadth.pct_above_sma200)}
          tone={breadth.pct_above_sma200 >= 0.5 ? 'up' : 'down'}
        />
        <StatTile label="En Fase 2 (Weinstein)" value={breadth.count_stage2} tone="up" />
        <StatTile label="Cumplen Minervini (8/8)" value={breadth.count_minervini_pass} tone="up" />
        <StatTile label="Golden crosses" value={breadth.golden_crosses} tone="up" />
        <StatTile label="Death crosses" value={breadth.death_crosses} tone="down" />
        <StatTile label="Sobrecompra / Sobreventa" value={`${breadth.count_overbought} / ${breadth.count_oversold}`} />
      </div>

      <div className="movers-grid">
        {DETAIL_GROUPS.map((group) => (
          <div key={group.key} className="movers-card">
            <p className="movers-card__title">{group.title}</p>
            <TickerSnapshotTable
              rows={detail[group.key] ?? []}
              columns={group.columns}
              showSector={false}
              emptyMessage="Ningún ticker cumple esta condición ahora mismo."
            />
          </div>
        ))}
      </div>
    </div>
  )
}

export default TrendBreadthPanel
