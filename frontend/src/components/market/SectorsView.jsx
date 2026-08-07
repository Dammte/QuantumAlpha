import { useState } from 'react'
import SectorStrength from './SectorStrength'
import IndustryCards from './IndustryCards'
import SectorRotationCard from './SectorRotationCard'

function SectorsView({ region }) {
  const [view, setView] = useState('industries')

  return (
    <div>
      <SectorRotationCard region={region} />
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
