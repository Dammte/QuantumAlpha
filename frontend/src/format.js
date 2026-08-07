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

export const GARCH_REGIME_LABELS = {
  baja: 'volatilidad baja',
  normal: 'volatilidad normal',
  elevada: 'volatilidad elevada',
  alta: 'volatilidad alta',
}

export function garchRegimeLabel(regime) {
  return GARCH_REGIME_LABELS[regime] ?? regime
}
