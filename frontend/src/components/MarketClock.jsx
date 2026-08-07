import { useEffect, useState } from 'react'

// Regular trading hours only (no pre-market/after-hours, no holiday calendar -
// a full exchange holiday calendar is a lot of upkeep for a personal tool, and
// closed-for-holiday is a rare enough edge case that "closed" simply won't
// show up as "open" on those few days without it; everything else is exact).
const MARKETS = [
  { key: 'us', label: 'EE.UU.', flag: '🇺🇸', timeZone: 'America/New_York', open: [9, 30], close: [16, 0] },
  { key: 'europe', label: 'Europa', flag: '🇪🇺', timeZone: 'Europe/Paris', open: [9, 0], close: [17, 30] },
]

const WEEKEND_DAYS = new Set(['Sat', 'Sun'])

function statusFor(market, now) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: market.timeZone,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    weekday: 'short',
  }).formatToParts(now)
  const map = Object.fromEntries(parts.map((p) => [p.type, p.value]))
  const hour = parseInt(map.hour, 10) % 24
  const minute = parseInt(map.minute, 10)
  const minutesNow = hour * 60 + minute
  const [openH, openM] = market.open
  const [closeH, closeM] = market.close
  const isOpen = !WEEKEND_DAYS.has(map.weekday) && minutesNow >= openH * 60 + openM && minutesNow < closeH * 60 + closeM
  const localTime = `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`
  return { isOpen, localTime }
}

function MarketClock() {
  const [now, setNow] = useState(() => new Date())

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30_000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="market-clock" aria-label="Estado de los mercados">
      {MARKETS.map((market) => {
        const { isOpen, localTime } = statusFor(market, now)
        return (
          <div
            key={market.key}
            className={`market-clock__item ${isOpen ? 'market-clock__item--open' : 'market-clock__item--closed'}`}
            title={`${market.label}: ${isOpen ? 'abierto' : 'cerrado'} · hora local ${localTime} (no incluye festivos)`}
          >
            <span className="market-clock__dot" />
            <span className="market-clock__flag">{market.flag}</span>
            <span className="market-clock__label">{market.label}</span>
            <span className="market-clock__state">{isOpen ? 'Abierto' : 'Cerrado'}</span>
          </div>
        )
      })}
    </div>
  )
}

export default MarketClock
