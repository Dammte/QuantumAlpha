// Mirrors the "us"/"europe" keys in backend/app/services/market_universe.py's
// REGIONS registry - kept in sync manually since there are only two today.
export const REGIONS = [
  { key: 'us', label: 'Estados Unidos', shortLabel: 'EE.UU.', flag: '🇺🇸', hint: 'Universo curado de EE.UU. (S&P 500 y similares)' },
  { key: 'europe', label: 'Europa', shortLabel: 'Europa', flag: '🇪🇺', hint: 'Universo curado europeo (Reino Unido, Alemania, Francia, Países Bajos, España, Suiza, Italia)' },
]

export const DEFAULT_REGION = 'us'
