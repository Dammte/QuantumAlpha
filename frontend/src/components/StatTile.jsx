function StatTile({ label, value, tone = 'neutral', hint }) {
  return (
    <div className="stat-tile">
      <p className="stat-tile__label">{label}</p>
      <p className={`stat-tile__value stat-tile__value--${tone}`}>{value}</p>
      {hint && <p className="stat-tile__hint">{hint}</p>}
    </div>
  )
}

export default StatTile
