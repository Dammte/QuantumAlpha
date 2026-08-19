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

// signal (see portfolio_risk_service.py) -> the same short label wherever a
// held position's trend signal is shown (positions table, swap suggestions).
export const SIGNAL_LABELS = {
  exit_warning: 'VENDER',
  add_candidate: 'Aumentar',
  watch: 'Vigilar',
  hold: 'Mantener',
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

// exit_urgency (see exit_engine.ExitUrgency) - the position-level urgency
// read that can override `signal` above (see portfolio_risk_service.py) but
// is shown as its own, more granular badge wherever a specific action is
// asked for: "acciones requeridas hoy" and the position detail card. `hold`
// intentionally shares no visual urgency with the other four - it's the
// "nothing to do" state, not a fifth severity tier.
export const EXIT_URGENCY_LABELS = {
  exit_now: 'Vender ya',
  reduce: 'Reducir posición',
  tighten_stop: 'Subir el stop',
  watch: 'Vigilar de cerca',
  hold: 'Tesis intacta',
}

export const EXIT_URGENCY_TONE = {
  exit_now: 'down',
  reduce: 'down',
  tighten_stop: 'warn',
  watch: 'warn',
  hold: 'neutral',
}

export const EXIT_URGENCY_ICON = {
  exit_now: '🔴',
  reduce: '🟠',
  tighten_stop: '🟡',
  watch: '🔵',
  hold: '',
}

// Mirrors exit_engine.py's own `_URGENCY_SEVERITY` (index = severity, most
// urgent first) - "hold" deliberately excluded, it never needs an action.
export const EXIT_URGENCY_SEVERITY = ['exit_now', 'reduce', 'tighten_stop', 'watch']

// The tiers the "acciones requeridas hoy" panel surfaces - "watch" is a
// passive monitoring state (already visible as the WATCH badge in the
// positions table), not something to act on today, so it's deliberately
// narrower than EXIT_URGENCY_SEVERITY above.
export const ACTION_REQUIRED_URGENCY_TIERS = ['exit_now', 'reduce', 'tighten_stop']

export function exitUrgencyRank(urgency) {
  const idx = EXIT_URGENCY_SEVERITY.indexOf(urgency)
  return idx === -1 ? EXIT_URGENCY_SEVERITY.length : idx
}

export function formatRMultiple(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}R`
}

// multi_timeframe.MultiTimeframeRead.alignment - the semáforo's headline read.
export const TIMEFRAME_ALIGNMENT_LABELS = {
  bullish_aligned: 'Alineación alcista (semanal + diaria)',
  bearish_aligned: 'Alineación bajista (semanal + diaria)',
  conflicted: 'Temporalidades en conflicto',
  transitioning: 'En transición',
}

export const TIMEFRAME_ALIGNMENT_TONE = {
  bullish_aligned: 'up',
  bearish_aligned: 'down',
  conflicted: 'warn',
  transitioning: 'neutral',
}

// trade_manager.ScaledExitPlan.action - a suggested (never automatic)
// partial exit at the +1R/+2R milestones.
export const SCALED_EXIT_ACTION_LABELS = {
  none: 'Sin recorte sugerido todavía',
  sell_at_1r: 'Recoger parte en +1R',
  sell_at_2r: 'Recoger parte en +2R',
}
