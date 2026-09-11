import { useState } from 'react'
import SectorStrength from './SectorStrength'
import IndustryCards from './IndustryCards'

// 2026-09: sector rotation, RRG y forecast (Markov) se retiraron - ver
// docs/quant_methodology.md. Lo que queda de "Sectores" (fuerza relativa e
// industrias) también se retira más adelante, sustituido por un único campo
// sector_rs_percentile en ticker_daily_state (Fase 2/3).
function SectorsView({ region }) {
  const [view, setView] = useState('industries')

  return (
    <div>
      <div className="sub-toggle">
        <button
          type="button"
          className={`sub-toggle__item ${view === 'sectors' ? 'sub-toggle__item--active' : ''}`}
          onClick={() => setView('sectors')}
        >
          Sectores (visión general)
        </button>
        <button
          type="button"
          className={`sub-toggle__item ${view === 'industries' ? 'sub-toggle__item--active' : ''}`}
          onClick={() => setView('industries')}
        >
          Industrias (detalle)
        </button>
      </div>
      {view === 'sectors' ? <SectorStrength region={region} /> : <IndustryCards region={region} />}
    </div>
  )
}

export default SectorsView
