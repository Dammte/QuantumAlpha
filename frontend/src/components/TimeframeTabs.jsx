import { TIMEFRAMES } from '../timeframes'

function TimeframeTabs({ value, onChange }) {
  return (
    <div className="timeframe-tabs" role="tablist" aria-label="Periodo">
      {TIMEFRAMES.map((tf) => (
        <button
          key={tf.key}
          role="tab"
          type="button"
          aria-selected={value === tf.key}
          className={`timeframe-tab ${value === tf.key ? 'timeframe-tab--active' : ''}`}
          onClick={() => onChange(tf.key)}
        >
          {tf.label}
        </button>
      ))}
    </div>
  )
}

export default TimeframeTabs
