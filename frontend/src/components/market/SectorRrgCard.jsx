import { useEffect, useState } from 'react'
import { api } from '../../api'
import SectorRrgChart from './SectorRrgChart'

// Cuarta auditoría, Bloque E: Relative Rotation Graph - complementa (no
// sustituye) a SectorRotationCard. Esa tarjeta responde "¿qué sector lidera
// ahora y qué ha significado eso históricamente en el ciclo económico?" -
// esta responde "¿está ese liderazgo acelerando o ya perdiendo fuelle?", el
// eje de momentum que SectorRotationCard nunca tuvo.
function SectorRrgCard({ region }) {
  const [readings, setReadings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setReadings(await api.getSectorRrg({ region }))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  if (loading) return <p className="empty-state">Calculando el mapa de rotación (RRG)…</p>
  if (error) return <div className="banner banner--error">{error}</div>

  return (
    <div>
      <p className="panel__hint">
        Cada sector, situado por su fuerza relativa frente al índice general (eje horizontal) y por el
        ritmo al que esa fuerza está cambiando (eje vertical) - un sector puede liderar en fuerza y aun así
        estar perdiendo impulso, o ir por detrás pero acelerando. La cola muestra su trayectoria reciente.
      </p>
      <SectorRrgChart readings={readings} />
    </div>
  )
}

export default SectorRrgCard
