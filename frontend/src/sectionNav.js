// Reconstruction (2026-09), Fase 6: the old single "Mercado" nav group (8
// sections under one umbrella) splits into the two top-level views the
// reconstruction brief actually asks for - "qué está a punto de disparar"
// (Radar, universe-wide) and "es esta entrada concreta buena" (Activo, one
// ticker at a time). Screener/Movers/Sectores/Tendencia/Soportes-Resistencias/
// A revisar are all universe-wide screening tools, same family as Radar
// itself - they move there, not into Activo, which stays a single-ticker
// deep dive. See docs/quant_methodology.md §25 and App.jsx/Sidebar.jsx.

export const RADAR_SECTIONS = [
  { key: 'radar', label: 'Radar' },
  { key: 'watchlist', label: 'A revisar' },
  { key: 'screener', label: 'Screener' },
  { key: 'movers', label: 'Movers' },
  { key: 'sectors', label: 'Sectores' },
  { key: 'trend', label: 'Tendencia' },
  { key: 'levels', label: 'Soportes/Resistencias' },
]

export const ACTIVO_SECTIONS = [
  { key: 'analysis', label: 'Analizar activo' },
  { key: 'context', label: 'Contexto' },
]
