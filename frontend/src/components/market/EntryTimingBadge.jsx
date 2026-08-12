import { entryTimingTone } from '../../format'

// `entryTiming` is CoreSignalsResponse.entry_timing (see entry_timing.py on the
// backend) - a read on how much of a setup's move looks still-ahead vs
// already-behind, shown next to (never instead of) the buy/wait/avoid verdict.
function EntryTimingBadge({ entryTiming, showDescription = false }) {
  if (!entryTiming) return null
  const tone = entryTimingTone(entryTiming.status)
  return (
    <p className="entry-timing">
      <span className={`sector-tier-badge sector-tier-badge--${tone}`}>{entryTiming.label}</span>
      {showDescription && <span className="entry-timing__description"> {entryTiming.description}</span>}
    </p>
  )
}

export default EntryTimingBadge
