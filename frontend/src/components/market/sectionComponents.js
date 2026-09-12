import MarketScreener from './MarketScreener'
import MarketMovers from './MarketMovers'
import SectorsView from './SectorsView'
import TrendBreadthPanel from './TrendBreadthPanel'
import SupportResistancePanel from './SupportResistancePanel'
import RadarView from './RadarView'
import MarketContextPanel from './MarketContextPanel'
import TickerAnalysisPanel from './analysis/TickerAnalysisPanel'

// Reconstruction (2026-09), Fase 6: the old single "Mercado" nav (8 sections
// under one umbrella) splits into the two top-level views the reconstruction
// brief actually asks for - see sectionNav.js's own comment for the full
// reasoning. Radar is every universe-wide screening tool (this is where
// Screener/Movers/Sectores/Tendencia/Soportes-Resistencias move to, not a
// demotion - they're the same family as Radar itself); Activo stays a
// single-ticker deep dive, nothing else. Kept in a plain (non-component)
// module, not SectionedView.jsx itself, so Vite's fast refresh can still
// treat that file as component-only.
export const RADAR_COMPONENT_BY_SECTION = {
  radar: RadarView,
  screener: MarketScreener,
  movers: MarketMovers,
  sectors: SectorsView,
  trend: TrendBreadthPanel,
  levels: SupportResistancePanel,
}

export const ACTIVO_COMPONENT_BY_SECTION = {
  analysis: TickerAnalysisPanel,
  context: MarketContextPanel,
}
