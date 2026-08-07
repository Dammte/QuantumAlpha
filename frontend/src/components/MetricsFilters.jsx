function MetricsFilters({ filters, onChange }) {
  return (
    <div className="filters-row">
      <label className="filter">
        <span>Desde</span>
        <input type="date" value={filters.start} onChange={(e) => onChange({ ...filters, start: e.target.value })} />
      </label>
      <label className="filter">
        <span>Hasta</span>
        <input type="date" value={filters.end} onChange={(e) => onChange({ ...filters, end: e.target.value })} />
      </label>
      <label className="filter">
        <span>Benchmark</span>
        <input
          type="text"
          placeholder="^GSPC"
          value={filters.benchmarkTicker}
          onChange={(e) => onChange({ ...filters, benchmarkTicker: e.target.value.toUpperCase() })}
        />
      </label>
    </div>
  )
}

export default MetricsFilters
