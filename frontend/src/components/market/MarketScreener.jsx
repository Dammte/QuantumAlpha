import { useEffect, useState } from 'react'
import { api } from '../../api'
import TickerSnapshotTable from './TickerSnapshotTable'

const SORT_OPTIONS = [
  { value: 'change_1d', label: '1D %' },
  { value: 'change_1w', label: '1S %' },
  { value: 'change_1m', label: '1M %' },
  { value: 'change_3m', label: '3M %' },
  { value: 'change_6m', label: '6M %' },
  { value: 'change_1y', label: '1A %' },
  { value: 'rsi14', label: 'RSI' },
  { value: 'relative_volume', label: 'Vol. relativo' },
  { value: 'rs_rating', label: 'RS Rating' },
  { value: 'adx14', label: 'ADX' },
  { value: 'price', label: 'Precio' },
]

const CAP_TIERS = [
  { value: '', label: 'Todas' },
  { value: 'mega', label: 'Mega' },
  { value: 'large', label: 'Large' },
  { value: 'mid', label: 'Mid' },
  { value: 'small', label: 'Small' },
]

const STAGES = [
  { value: '', label: 'Todas' },
  { value: 'stage1', label: 'Fase 1 (base)' },
  { value: 'stage2', label: 'Fase 2 (avance)' },
  { value: 'stage3', label: 'Fase 3 (techo)' },
  { value: 'stage4', label: 'Fase 4 (declive)' },
]

const EMPTY_FORM = {
  sector: '',
  industry: '',
  min_price: '',
  max_price: '',
  min_rsi: '',
  max_rsi: '',
  min_relative_volume: '',
  min_rs_rating: '',
  above_sma50: '',
  above_sma200: '',
  trend: '',
  stage: '',
  cap_tier: '',
  minervini_pass: '',
}

function MarketScreener({ region }) {
  const [universe, setUniverse] = useState({ sectors: {}, industries: [] })
  const [form, setForm] = useState(EMPTY_FORM)
  const [sortBy, setSortBy] = useState('change_1d')
  const [sortDir, setSortDir] = useState('desc')
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.getMarketUniverse({ region }).then((data) => {
      setUniverse(data)
      // Industry names differ per region (sector names don't) - drop a stale pick.
      setForm((f) => ({ ...f, industry: '' }))
    })
  }, [region])

  const runScreener = async (filters) => {
    setLoading(true)
    setError(null)
    try {
      const boolFilter = (v) => (v === '' ? undefined : v === 'true')
      const data = await api.getMarketScreener({
        ...filters,
        region,
        above_sma50: boolFilter(filters.above_sma50),
        above_sma200: boolFilter(filters.above_sma200),
        minervini_pass: boolFilter(filters.minervini_pass),
        sort_by: sortBy,
        sort_dir: sortDir,
      })
      setRows(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    async function load() {
      await runScreener(form)
    }
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    region,
    sortBy,
    sortDir,
    form.sector,
    form.industry,
    form.trend,
    form.stage,
    form.cap_tier,
    form.minervini_pass,
    form.above_sma50,
    form.above_sma200,
  ])

  const update = (field) => (e) => setForm({ ...form, [field]: e.target.value })

  const industryOptions = form.sector
    ? universe.industries.filter((i) => i.sector === form.sector)
    : universe.industries

  return (
    <div>
      <div className="screener-filters">
        <label className="filter">
          <span>Sector</span>
          <select value={form.sector} onChange={(e) => setForm({ ...form, sector: e.target.value, industry: '' })}>
            <option value="">Todos</option>
            {Object.keys(universe.sectors).map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className="filter">
          <span>Industria</span>
          <select value={form.industry} onChange={update('industry')}>
            <option value="">Todas</option>
            {industryOptions.map((i) => (
              <option key={i.name} value={i.name}>
                {i.name}
              </option>
            ))}
          </select>
        </label>
        <label className="filter">
          <span>Capitalización</span>
          <select value={form.cap_tier} onChange={update('cap_tier')}>
            {CAP_TIERS.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label className="filter">
          <span>Tendencia</span>
          <select value={form.trend} onChange={update('trend')}>
            <option value="">Todas</option>
            <option value="uptrend">Alcista</option>
            <option value="downtrend">Bajista</option>
            <option value="sideways">Lateral</option>
          </select>
        </label>
        <label className="filter">
          <span>Fase (Weinstein)</span>
          <select value={form.stage} onChange={update('stage')}>
            {STAGES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label className="filter">
          <span>Minervini (8/8)</span>
          <select value={form.minervini_pass} onChange={update('minervini_pass')}>
            <option value="">-</option>
            <option value="true">Sí</option>
          </select>
        </label>
        <label className="filter">
          <span>Precio mín.</span>
          <input type="number" value={form.min_price} onChange={update('min_price')} placeholder="0" />
        </label>
        <label className="filter">
          <span>Precio máx.</span>
          <input type="number" value={form.max_price} onChange={update('max_price')} placeholder="1000" />
        </label>
        <label className="filter">
          <span>RSI mín.</span>
          <input type="number" value={form.min_rsi} onChange={update('min_rsi')} placeholder="0" />
        </label>
        <label className="filter">
          <span>RSI máx.</span>
          <input type="number" value={form.max_rsi} onChange={update('max_rsi')} placeholder="100" />
        </label>
        <label className="filter">
          <span>RS Rating mín.</span>
          <input type="number" value={form.min_rs_rating} onChange={update('min_rs_rating')} placeholder="0" />
        </label>
        <label className="filter">
          <span>Vol. rel. mín.</span>
          <input
            type="number"
            step="0.1"
            value={form.min_relative_volume}
            onChange={update('min_relative_volume')}
            placeholder="1.0"
          />
        </label>
        <label className="filter">
          <span>Sobre MA50</span>
          <select value={form.above_sma50} onChange={update('above_sma50')}>
            <option value="">-</option>
            <option value="true">Sí</option>
            <option value="false">No</option>
          </select>
        </label>
        <label className="filter">
          <span>Sobre MA200</span>
          <select value={form.above_sma200} onChange={update('above_sma200')}>
            <option value="">-</option>
            <option value="true">Sí</option>
            <option value="false">No</option>
          </select>
        </label>
        <label className="filter">
          <span>Ordenar por</span>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <label className="filter">
          <span>Dirección</span>
          <select value={sortDir} onChange={(e) => setSortDir(e.target.value)}>
            <option value="desc">Descendente</option>
            <option value="asc">Ascendente</option>
          </select>
        </label>
        <button type="button" className="button-primary" onClick={() => runScreener(form)} disabled={loading}>
          {loading ? 'Buscando…' : 'Aplicar filtros'}
        </button>
      </div>

      {error && <div className="banner banner--error">{error}</div>}
      <p className="screener-count">{loading ? 'Cargando…' : `${rows.length} resultados`}</p>
      <TickerSnapshotTable
        rows={rows}
        columns={['price', 'change_1d', 'change_1w', 'change_1m', 'rsi14', 'rs_rating', 'adx14', 'stage', 'trend']}
      />
    </div>
  )
}

export default MarketScreener
