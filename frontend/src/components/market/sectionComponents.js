import MarketScreener from './MarketScreener'
import MarketMovers from './MarketMovers'
import TrendBreadthPanel from './TrendBreadthPanel'
import SupportResistancePanel from './SupportResistancePanel'
import RadarView from './RadarView'
import MarketContextPanel from './MarketContextPanel'
import TickerAnalysisPanel from './analysis/TickerAnalysisPanel'

// Reconstruction (2026-09): "Sectores" se retira aquí (Parte 2.5 del
// documento real - sustituido por un único campo `sector_rs_percentile` que
// se construye en el job diario, Fase 2/3, no por una vista). Movers/
// Tendencia/Soportes-Resistencias/Contexto siguen vivos por ahora - su
// consolidación en las 4 vistas finales (Parte 14) es trabajo de Fase 6, no
// de esta fase de borrado. Kept in a plain (non-component) module, not
// SectionedView.jsx itself, so Vite's fast refresh can still treat that file
// as component-only.
export const RADAR_COMPONENT_BY_SECTION = {
  radar: RadarView,
  screener: MarketScreener,
  movers: MarketMovers,
  trend: TrendBreadthPanel,
  levels: SupportResistancePanel,
}

export const ACTIVO_COMPONENT_BY_SECTION = {
  analysis: TickerAnalysisPanel,
  context: MarketContextPanel,
}
