import { useState } from 'react'
import SectorStrength from './SectorStrength'
import IndustryCards from './IndustryCards'
import SectorRotationCard from './SectorRotationCard'
import SectorRrgCard from './SectorRrgCard'
import SectorForecastCard from './SectorForecastCard'

function SectorsView({ region, onNavigateToTicker }) {
  const [view, setView] = useState('industries')

  return (
    <div>
      <SectorForecastCard region={region} onNavigateToTicker={onNavigateToTicker} />
      <SectorRotationCard region={region} />
      <section className="panel panel--nested">
        <h3>Mapa de rotación sectorial (RRG)</h3>
        <SectorRrgCard region={region} />
      </section>
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
