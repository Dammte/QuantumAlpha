import { MARKET_SECTIONS } from '../marketSections'
import { REGIONS } from '../regions'
import MarketClock from './MarketClock'

function Sidebar({ view, marketSection, region, onSelectPortfolio, onSelectMarketSection, onRegionChange, onSelectPerformance }) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__logo">Q</span>
        <div>
          <p className="sidebar__title">QuantumAlpha</p>
          <p className="sidebar__subtitle">Panel de control</p>
        </div>
      </div>

      <MarketClock />

      <nav className="sidebar__nav" aria-label="Navegación principal">
        <button
          type="button"
          className={`sidebar__item ${view === 'portfolio' ? 'sidebar__item--active' : ''}`}
          onClick={onSelectPortfolio}
        >
          Mi Cartera
        </button>

        <button
          type="button"
          className={`sidebar__item ${view === 'performance' ? 'sidebar__item--active' : ''}`}
          onClick={onSelectPerformance}
        >
          Rendimiento del sistema
        </button>

        <p className="sidebar__group-label">Mercado</p>

        <div className="sidebar__region-switch" role="tablist" aria-label="Región del mercado">
          {REGIONS.map((r) => (
            <button
              key={r.key}
              type="button"
              role="tab"
              aria-selected={region === r.key}
              className={`sidebar__region-switch__item ${region === r.key ? 'sidebar__region-switch__item--active' : ''}`}
              onClick={() => onRegionChange(r.key)}
              title={r.hint}
            >
              {r.flag} {r.shortLabel}
            </button>
          ))}
        </div>

        {MARKET_SECTIONS.map((s) => (
          <button
            key={s.key}
            type="button"
            className={`sidebar__item sidebar__item--nested ${
              view === 'market' && marketSection === s.key ? 'sidebar__item--active' : ''
            }`}
            onClick={() => onSelectMarketSection(s.key)}
          >
            {s.label}
          </button>
        ))}
      </nav>
    </aside>
  )
}

export default Sidebar
