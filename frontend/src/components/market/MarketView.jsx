import MarketScreener from './MarketScreener'
import MarketMovers from './MarketMovers'
import SectorsView from './SectorsView'
import TrendBreadthPanel from './TrendBreadthPanel'
import SupportResistancePanel from './SupportResistancePanel'
import Watchlist from './Watchlist'
import MarketContextPanel from './MarketContextPanel'
import TickerAnalysisPanel from './analysis/TickerAnalysisPanel'

const COMPONENT_BY_SECTION = {
  analysis: TickerAnalysisPanel,
  context: MarketContextPanel,
  screener: MarketScreener,
  movers: MarketMovers,
  sectors: SectorsView,
  trend: TrendBreadthPanel,
  levels: SupportResistancePanel,
  watchlist: Watchlist,
}

function MarketView({ section, presetTicker, onNavigateToTicker, region }) {
  const ActiveComponent = COMPONENT_BY_SECTION[section] ?? TickerAnalysisPanel

  return (
    <div className="dashboard">
      <section className="panel">
        <ActiveComponent presetTicker={presetTicker} onNavigateToTicker={onNavigateToTicker} region={region} />
      </section>
    </div>
  )
}

export default MarketView
