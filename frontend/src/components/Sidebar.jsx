import { RADAR_SECTIONS, ACTIVO_SECTIONS } from '../sectionNav'
import { REGIONS } from '../regions'
import MarketClock from './MarketClock'

// Reconstruction (2026-09), Fase 6: 4 top-level views matching the
// reconstruction brief's own 4 questions (Parte 0) - qué hacer hoy con lo
// que ya tengo (Hoy), qué está a punto de disparar (Radar), es esta entrada
// concreta buena (Activo), está funcionando el sistema (Sistema) - replacing
// the old "Mi Cartera / Rendimiento del sistema / Mercado (8 secciones)"
// layout. Radar and Activo each keep their own nested sub-sections (same
// pattern the old "Mercado" group used) and their own region switcher, since
// only those two views read from a specific market/region.
function RegionSwitch({ region, onRegionChange }) {
  return (
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
  )
}

function Sidebar({
  view,
  radarSection,
  activoSection,
  region,
  onSelectHoy,
  onSelectSistema,
  onSelectRadarSection,
  onSelectActivoSection,
  onRegionChange,
}) {
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
          className={`sidebar__item ${view === 'hoy' ? 'sidebar__item--active' : ''}`}
          onClick={onSelectHoy}
        >
          Hoy
        </button>

        <p className="sidebar__group-label">Radar</p>
        <RegionSwitch region={region} onRegionChange={onRegionChange} />
        {RADAR_SECTIONS.map((s) => (
          <button
            key={s.key}
            type="button"
            className={`sidebar__item sidebar__item--nested ${
              view === 'radar' && radarSection === s.key ? 'sidebar__item--active' : ''
            }`}
            onClick={() => onSelectRadarSection(s.key)}
          >
            {s.label}
          </button>
        ))}

        <p className="sidebar__group-label">Activo</p>
        {ACTIVO_SECTIONS.map((s) => (
          <button
            key={s.key}
            type="button"
            className={`sidebar__item sidebar__item--nested ${
              view === 'activo' && activoSection === s.key ? 'sidebar__item--active' : ''
            }`}
            onClick={() => onSelectActivoSection(s.key)}
          >
            {s.label}
          </button>
        ))}

        <button
          type="button"
          className={`sidebar__item ${view === 'sistema' ? 'sidebar__item--active' : ''}`}
          onClick={onSelectSistema}
        >
          Sistema
        </button>
      </nav>
    </aside>
  )
}

export default Sidebar
