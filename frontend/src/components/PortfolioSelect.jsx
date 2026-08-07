import { useState } from 'react'

function PortfolioSelect({ portfolios, selectedId, onSelect, onCreate }) {
  const [name, setName] = useState('')
  const [creating, setCreating] = useState(false)

  const handleCreate = async (event) => {
    event.preventDefault()
    if (!name.trim()) return
    setCreating(true)
    try {
      await onCreate(name.trim())
      setName('')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="portfolio-select">
      <label className="filter">
        <span>Cartera</span>
        <select value={selectedId ?? ''} onChange={(e) => onSelect(Number(e.target.value))}>
          {portfolios.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
      <form className="portfolio-select__new" onSubmit={handleCreate}>
        <input placeholder="Nueva cartera…" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="submit" className="button-secondary" disabled={creating}>
          {creating ? 'Creando…' : '+ Crear'}
        </button>
      </form>
    </div>
  )
}

export default PortfolioSelect
