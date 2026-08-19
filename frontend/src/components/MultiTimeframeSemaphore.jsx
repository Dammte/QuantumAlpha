import { TIMEFRAME_ALIGNMENT_LABELS, TIMEFRAME_ALIGNMENT_TONE } from '../format'
import { TREND_TONE, stageLabel, trendLabel } from '../marketFormat'

const CROSS_DIRECTION_LABEL = { golden: 'alcista', death: 'bajista' }

// `read` is multi_timeframe.TimeframeRead (weekly or daily) - the dot's own
// color is just its trend, the tooltip carries the fuller read (stage, MA
// crosses, price-vs-SMA) so a single glance still tells apart "downtrend
// because of a stage 4 top" from "downtrend, but back above SMA50 already".
function timeframeTooltip(label, read) {
  if (!read) return `${label}: historial insuficiente todavía`
  const parts = [trendLabel(read.trend)]
  if (read.stage) parts.push(stageLabel(read.stage))
  if (read.ma_cross_50_200) {
    parts.push(`cruce SMA50/200 ${CROSS_DIRECTION_LABEL[read.ma_cross_50_200] ?? read.ma_cross_50_200} confirmado`)
  }
  if (read.price_vs_sma50) parts.push(`precio ${read.price_vs_sma50 === 'above' ? 'sobre' : 'bajo'} SMA50`)
  return `${label}: ${parts.join(' · ')}`
}

function TimeframeDot({ label, read }) {
  const tone = read ? (TREND_TONE[read.trend] ?? 'neutral') : 'empty'
  return (
    <span className={`mtf-dot mtf-dot--${tone}`} title={timeframeTooltip(label, read)}>
      <span className="mtf-dot__mark" aria-hidden="true">●</span>
      <span className="mtf-dot__label">{label}</span>
    </span>
  )
}

// `multiTimeframe` is MultiTimeframeResponse (see multi_timeframe.py's
// docstring - the D1 fix: before this existed the system only ever saw daily
// bars). `compact` renders just the two dots (for a table row - hover for
// detail); the full form adds the alignment badge and, when they exist, the
// specific conflicts between what weekly and daily are each saying.
function MultiTimeframeSemaphore({ multiTimeframe, compact = false }) {
  if (!multiTimeframe) return null
  const tone = TIMEFRAME_ALIGNMENT_TONE[multiTimeframe.alignment] ?? 'neutral'
  const label = TIMEFRAME_ALIGNMENT_LABELS[multiTimeframe.alignment] ?? multiTimeframe.alignment

  return (
    <div className={`mtf-semaphore ${compact ? 'mtf-semaphore--compact' : ''}`}>
      <div className="mtf-semaphore__dots" title={compact ? label : undefined}>
        <TimeframeDot label="Semanal" read={multiTimeframe.weekly} />
        <TimeframeDot label="Diario" read={multiTimeframe.daily} />
      </div>
      {!compact && (
        <>
          <span className={`sector-tier-badge sector-tier-badge--${tone}`}>{label}</span>
          {multiTimeframe.conflicts.length > 0 && (
            <ul className="mtf-semaphore__conflicts">
              {multiTimeframe.conflicts.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  )
}

export default MultiTimeframeSemaphore
