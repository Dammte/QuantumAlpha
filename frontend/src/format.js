// London Stock Exchange quotes most stocks in pence (GBp/GBX), not pounds -
// yfinance passes that through as-is. "GBp" isn't a real ISO 4217 code
// Intl.NumberFormat recognizes, so normalize to actual pounds before formatting.
const PENCE_CURRENCIES = new Set(['GBp', 'GBX'])

export function formatCurrency(value, currency = 'USD') {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const isPence = PENCE_CURRENCIES.has(currency)
  const amount = isPence ? value / 100 : value
  const isoCurrency = isPence ? 'GBP' : currency
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: isoCurrency, maximumFractionDigits: 2 }).format(amount)
}

export function formatPercent(value, { signed = false } = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
    signDisplay: signed ? 'exceptZero' : 'auto',
  }).format(value)
  return formatted
}

export function formatRatio(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(2)
}

export function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(value)
}

export function formatRelativeTime(isoString) {
  if (!isoString) return null
  const then = new Date(isoString).getTime()
  if (Number.isNaN(then)) return null
  const diffMinutes = Math.round((Date.now() - then) / 60000)
  if (diffMinutes < 1) return 'justo ahora'
  if (diffMinutes < 60) return `hace ${diffMinutes} min`
  const diffHours = Math.round(diffMinutes / 60)
  if (diffHours < 24) return `hace ${diffHours} h`
  const diffDays = Math.round(diffHours / 24)
  return `hace ${diffDays} d`
}

// entry_timing.status -> the same tone vocabulary sector-tier-badge/signal-badge
// already use (up/neutral/warn/down), so an "óptimo" reads as unambiguously good
// and "muy extendido" as unambiguously risky wherever this shows up.
export const ENTRY_TIMING_TONE = {
  optimal: 'up',
  valid: 'neutral',
  late: 'warn',
  extended: 'down',
}

export function entryTimingTone(status) {
  return ENTRY_TIMING_TONE[status] ?? 'neutral'
}

// Mirrors recommendation_engine.py's BUY_THRESHOLD (5): a score sits on an
// open-ended additive scale, not a fixed 0-10 range, so "perfect" is defined
// relative to that floor - roughly double it - rather than an arbitrary max.
// See docs/quant_methodology.md and recommendation_engine.py for the full
// factor breakdown behind any given score.
export const EXCEPTIONAL_SCORE_THRESHOLD = 10

export function isExceptionalScore(score) {
  return score >= EXCEPTIONAL_SCORE_THRESHOLD
}

export const GARCH_REGIME_LABELS = {
  baja: 'volatilidad baja',
  normal: 'volatilidad normal',
  elevada: 'volatilidad elevada',
  alta: 'volatilidad alta',
}

export function garchRegimeLabel(regime) {
  return GARCH_REGIME_LABELS[regime] ?? regime
}
