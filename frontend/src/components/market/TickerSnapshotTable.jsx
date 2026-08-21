import { formatCurrency, formatPercent, formatRatio } from '../../format'
import { stageLabel } from '../../marketFormat'
import ImminentCrossBadge from './ImminentCrossBadge'
import TrendBadge from './TrendBadge'

const DEFAULT_COLUMNS = ['price', 'change_1d', 'change_1w', 'rsi14', 'relative_volume', 'trend']

const COLUMN_LABELS = {
  price: 'Precio',
  change_1d: '1D',
  change_1w: '1S',
  change_1m: '1M',
  change_3m: '3M',
  change_6m: '6M',
  change_1y: '1A',
  rsi14: 'RSI(14)',
  relative_volume: 'Vol. rel.',
  dist_52w_high: 'Vs. máx 52s',
  dist_52w_low: 'Vs. mín 52s',
  atr_multiple: 'ATR mult.',
  adx14: 'ADX(14)',
  rs_rating: 'RS Rating',
  mansfield_rs: 'Mansfield RS',
  trend: 'Tendencia',
  stage: 'Fase',
  imminent_cross_short_term: 'Próximo cruce (corto plazo)',
}

function renderCell(row, column) {
  switch (column) {
    case 'price':
      return formatCurrency(row.price, row.currency)
    case 'change_1d':
    case 'change_1w':
    case 'change_1m':
    case 'change_3m':
    case 'change_6m':
    case 'change_1y':
    case 'dist_52w_high':
    case 'dist_52w_low':
      return (
        <span className={row[column] >= 0 ? 'delta-up' : 'delta-down'}>
          {formatPercent(row[column], { signed: true })}
        </span>
      )
    case 'rsi14':
      return row.rsi14 === null ? '—' : row.rsi14.toFixed(1)
    case 'relative_volume':
      return row.relative_volume === null ? '—' : `${row.relative_volume.toFixed(2)}x`
    case 'atr_multiple':
      return formatRatio(row.atr_multiple)
    case 'adx14':
      return row.adx14 === null ? '—' : row.adx14.toFixed(1)
    case 'rs_rating':
      return row.rs_rating === null ? '—' : row.rs_rating
    case 'mansfield_rs':
      return row.mansfield_rs === null ? '—' : row.mansfield_rs.toFixed(2)
    case 'trend':
      return <TrendBadge trend={row.trend} />
    case 'stage':
      return stageLabel(row.stage)
    case 'imminent_cross_short_term':
      return row.imminent_cross_short_term ? (
        <ImminentCrossBadge imminentCross={row.imminent_cross_short_term} shortTerm />
      ) : (
        '—'
      )
    default:
      return row[column]
  }
}

function TickerSnapshotTable({ rows, columns = DEFAULT_COLUMNS, showSector = true, emptyMessage = 'Sin resultados.' }) {
  if (rows.length === 0) {
    return <p className="empty-state">{emptyMessage}</p>
  }

  return (
    <div className="table-scroll table-scroll--capped">
      <table className="positions-table">
        <thead>
          <tr>
            <th>Ticker</th>
            {showSector && <th>Sector</th>}
            {columns.map((col) => (
              <th key={col} className="num">
                {COLUMN_LABELS[col]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.ticker}>
              <td className="positions-table__ticker">{row.ticker}</td>
              {showSector && <td>{row.sector}</td>}
              {columns.map((col) => (
                <td key={col} className="num">
                  {renderCell(row, col)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default TickerSnapshotTable
