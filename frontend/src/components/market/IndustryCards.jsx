import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api'
import { formatPercent } from '../../format'

const PERIODS = [
  { key: 'change_1d', label: '1D' },
  { key: 'change_1w', label: '1S' },
  { key: 'change_1m', label: '1M' },
  { key: 'change_3m', label: '3M' },
  { key: 'change_6m', label: '6M' },
  { key: 'change_1y', label: '1A' },
]

const CAP_TIER_LABELS = { mega: 'Mega', large: 'Large', mid: 'Mid', small: 'Small' }

function IndustryCards({ region }) {
  const [industries, setIndustries] = useState([])
  const [period, setPeriod] = useState('change_1m')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setIndustries(await api.getIndustryPerformance({ region }))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  const sorted = useMemo(
    () =>
      [...industries]
        .filter((i) => i[period] !== null && i[period] !== undefined)
        .sort((a, b) => b[period] - a[period]),
    [industries, period],
  )

  if (loading) return <p className="empty-state">Cargando industrias…</p>
  if (error) return <div className="banner banner--error">{error}</div>

  return (
    <div>
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

      <div className="industry-cards">
        {sorted.map((industry, index) => {
          const isUp = industry[period] >= 0
          return (
            <div key={industry.industry} className="industry-card">
              <div className="industry-card__header">
                <span className="industry-card__rank">#{index + 1}</span>
                <div>
                  <p className="industry-card__name">{industry.industry}</p>
                  <p className="industry-card__sector">
                    {industry.sector}
                    {industry.etf && <span className="industry-card__etf"> · {industry.etf}</span>}
                  </p>
                </div>
                <span className={`industry-card__change ${isUp ? 'delta-up' : 'delta-down'}`}>
                  {formatPercent(industry[period], { signed: true })}
                </span>
              </div>

              {industry.avg_rs_rating !== null && (
                <p className="industry-card__avg-rs">RS medio del subsector: {industry.avg_rs_rating.toFixed(0)}</p>
              )}

              {industry.leaders.length > 0 && (
                <ul className="industry-card__leaders">
                  {industry.leaders.map((leader) => (
                    <li key={leader.ticker}>
                      <span className="positions-table__ticker">{leader.ticker}</span>
                      <span className={`cap-tier-badge cap-tier-badge--${leader.cap_tier}`}>
                        {CAP_TIER_LABELS[leader.cap_tier] ?? leader.cap_tier}
                      </span>
                      <span className="industry-card__leader-rs">RS {leader.rs_rating ?? '—'}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default IndustryCards
