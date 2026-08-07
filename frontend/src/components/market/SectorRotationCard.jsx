import { useEffect, useState } from 'react'
import { api } from '../../api'

function SectorRotationCard({ region }) {
  const [rotation, setRotation] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setRotation(await api.getSectorRotation({ region }))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  if (loading) return <p className="empty-state">Cargando rotación sectorial…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (!rotation) return null

  return (
    <div className={`rotation-card ${rotation.defensive_leadership ? 'rotation-card--warn' : ''}`}>
      <div className="rotation-card__columns">
        <div>
          <p className="rotation-card__label">Sectores líderes ahora mismo</p>
          <ol className="rotation-card__list rotation-card__list--up">
            {rotation.leaders.map((sector) => (
              <li key={sector}>{sector}</li>
            ))}
          </ol>
        </div>
        <div>
          <p className="rotation-card__label">Sectores rezagados ahora mismo</p>
          <ol className="rotation-card__list rotation-card__list--down">
            {rotation.laggards.map((sector) => (
              <li key={sector}>{sector}</li>
            ))}
          </ol>
        </div>
      </div>

      {rotation.cycle_description && (
        <p className="rotation-card__cycle">
          <strong>Lectura de ciclo: {rotation.cycle_phase}.</strong> {rotation.cycle_description}
        </p>
      )}

      {rotation.warning && <p className="rotation-card__warning">{rotation.warning}</p>}

      <p className="rotation-card__disclaimer">
        Basado únicamente en fuerza relativa de precios entre los 11 sectores (mismo método que el RS Rating
        por acción, aplicado a nivel sectorial) - no interpreta noticias ni causas, solo qué está liderando
        objetivamente ahora mismo y qué ha significado históricamente ese patrón de liderazgo.
        {region === 'europe' && (
          <> El modelo de fases del ciclo se construyó sobre el ciclo económico de EE.UU.; aplicado aquí a
          datos sectoriales europeos es una aproximación razonable (los ciclos están muy correlacionados),
          no un modelo europeo investigado por separado.</>
        )}
      </p>
    </div>
  )
}

export default SectorRotationCard
