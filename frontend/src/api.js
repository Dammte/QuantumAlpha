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

  getMarketTrend: ({ region, refresh } = {}) =>
    request(`/api/v1/market/trend${toQueryString({ region, refresh })}`),

  getSupportResistance: (ticker, { start, end } = {}) =>
    request(`/api/v1/market/tickers/${ticker}/levels${toQueryString({ start, end })}`),

  getMarketTrendDetail: ({ region, refresh } = {}) =>
    request(`/api/v1/market/trend/detail${toQueryString({ region, refresh })}`),

  getLevelsProximity: ({ region, threshold, refresh } = {}) =>
    request(`/api/v1/market/levels/proximity${toQueryString({ region, threshold, refresh })}`),

  getMarketContext: () => request('/api/v1/market/context'),

  getPortfolioRisk: (portfolioId, { refresh } = {}) =>
    request(`/api/v1/portfolios/${portfolioId}/risk${toQueryString({ refresh })}`),

  getPortfolioConstruction: (portfolioId) => request(`/api/v1/portfolios/${portfolioId}/construction`),

  getTickerAnalysis: (ticker, { horizon } = {}) =>
    request(`/api/v1/market/tickers/${ticker}/analysis${toQueryString({ horizon })}`),

  // No `region` default here on purpose - omitting it lets the backend infer
  // the ticker's region itself (`market_universe.region_of`), same as the
  // free-text "Analizar activo" search this is always called from.
  getRelationshipMap: (ticker, { region } = {}) =>
    request(`/api/v1/market/tickers/${ticker}/relationships${toQueryString({ region })}`),

  getSignalPerformance: () => request('/api/v1/system/signal-performance'),

  // Reconstruction (2026-09), Fase 5: pure reads over daily_close.py's own
  // precomputed tables - never a live universe scan/recompute like every
  // request above this one.
  getRadar: ({ region, portfolioId } = {}) =>
    request(`/api/v1/market/radar${toQueryString({ region, portfolio_id: portfolioId })}`),

  getPortfolioToday: (portfolioId) => request(`/api/v1/portfolios/${portfolioId}/today`),
}
