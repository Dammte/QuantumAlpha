import { formatRelativeTime } from '../format'

// Shown wherever a result now comes from the durable/long-TTL cache
// (portfolio risk, watchlist, premium watchlist - see durable_cache.py on the
// backend) instead of being recomputed on every visit: makes that explicit
// ("actualizado hace X") and gives an explicit way to force a real recompute,
// rather than the page just silently showing possibly hours-old data with no
// indication of it.
function RefreshBar({ computedAt, onRefresh, refreshing }) {
  const relative = formatRelativeTime(computedAt)
  return (
    <div className="refresh-bar">
      {relative && <span className="refresh-bar__label">Actualizado {relative}</span>}
      <button type="button" className="refresh-bar__button" onClick={onRefresh} disabled={refreshing}>
        {refreshing ? 'Actualizando…' : 'Actualizar ahora'}
      </button>
    </div>
  )
}

export default RefreshBar
