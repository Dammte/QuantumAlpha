const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch {
      // response had no JSON body
    }
    throw new Error(detail)
  }

  if (response.status === 204) return null
  return response.json()
}

function toQueryString(params) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') search.set(key, value)
  }
  const query = search.toString()
  return query ? `?${query}` : ''
}

export const api = {
  listPortfolios: () => request('/api/v1/portfolios'),
  createPortfolio: (name, baseCurrency = 'USD') =>
    request('/api/v1/portfolios', {
      method: 'POST',
      body: JSON.stringify({ name, base_currency: baseCurrency }),
    }),
  deletePortfolio: (id) => request(`/api/v1/portfolios/${id}`, { method: 'DELETE' }),

  getSummary: (id) => request(`/api/v1/portfolios/${id}/summary`),

  listTransactions: (id) => request(`/api/v1/portfolios/${id}/transactions`),
  addTransaction: (id, transaction) =>
    request(`/api/v1/portfolios/${id}/transactions`, {
      method: 'POST',
      body: JSON.stringify(transaction),
    }),

  getMetrics: (id, { start, end, benchmarkTicker } = {}) => {
    const params = new URLSearchParams()
    if (start) params.set('start', start)
    if (end) params.set('end', end)
    if (benchmarkTicker) params.set('benchmark_ticker', benchmarkTicker)
    const query = params.toString()
    return request(`/api/v1/portfolios/${id}/metrics${query ? `?${query}` : ''}`)
  },

  getHistory: (id, { start, end, benchmarkTicker } = {}) => {
    const params = new URLSearchParams()
    if (start) params.set('start', start)
    if (end) params.set('end', end)
    if (benchmarkTicker) params.set('benchmark_ticker', benchmarkTicker)
    const query = params.toString()
    return request(`/api/v1/portfolios/${id}/history${query ? `?${query}` : ''}`)
  },

  deleteTransaction: (portfolioId, transactionId) =>
    request(`/api/v1/portfolios/${portfolioId}/transactions/${transactionId}`, { method: 'DELETE' }),

  getMarketUniverse: ({ region } = {}) => request(`/api/v1/market/universe${toQueryString({ region })}`),

  getMarketScreener: (filters = {}) =>
    request(`/api/v1/market/screener${toQueryString(filters)}`),

  getMarketMovers: ({ region, refresh } = {}) =>
    request(`/api/v1/market/movers${toQueryString({ region, refresh })}`),

  getSectorPerformance: ({ region, refresh } = {}) =>
    request(`/api/v1/market/sectors${toQueryString({ region, refresh })}`),

  getSectorForecast: ({ region, refresh } = {}) =>
    request(`/api/v1/market/sectors/forecast${toQueryString({ region, refresh })}`),

  getSectorRotation: ({ region, refresh } = {}) =>
    request(`/api/v1/market/sectors/rotation${toQueryString({ region, refresh })}`),

  getMarketTrend: ({ region, refresh } = {}) =>
    request(`/api/v1/market/trend${toQueryString({ region, refresh })}`),

  getSupportResistance: (ticker, { start, end } = {}) =>
    request(`/api/v1/market/tickers/${ticker}/levels${toQueryString({ start, end })}`),

  getIndustryPerformance: ({ region, refresh } = {}) =>
    request(`/api/v1/market/industries${toQueryString({ region, refresh })}`),

  getMarketTrendDetail: ({ region, refresh } = {}) =>
    request(`/api/v1/market/trend/detail${toQueryString({ region, refresh })}`),

  getWatchlist: ({ region, horizon, refresh } = {}) =>
    request(`/api/v1/market/watchlist${toQueryString({ region, horizon, refresh })}`),

  getPremiumWatchlist: ({ region, tier, refresh } = {}) =>
    request(`/api/v1/market/watchlist/premium${toQueryString({ region, tier, refresh })}`),

  getLevelsProximity: ({ region, threshold, refresh } = {}) =>
    request(`/api/v1/market/levels/proximity${toQueryString({ region, threshold, refresh })}`),

  getMarketContext: () => request('/api/v1/market/context'),

  getPortfolioRisk: (portfolioId, { refresh } = {}) =>
    request(`/api/v1/portfolios/${portfolioId}/risk${toQueryString({ refresh })}`),

  getPortfolioConstruction: (portfolioId) => request(`/api/v1/portfolios/${portfolioId}/construction`),

  getTickerAnalysis: (ticker, { horizon } = {}) =>
    request(`/api/v1/market/tickers/${ticker}/analysis${toQueryString({ horizon })}`),

  getSignalPerformance: () => request('/api/v1/system/signal-performance'),

  getFactorAblation: ({ horizonDays } = {}) =>
    request(`/api/v1/system/factor-ablation${toQueryString({ horizon_days: horizonDays })}`),
}
