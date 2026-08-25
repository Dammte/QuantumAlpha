import { formatCurrency, formatRatio } from '../../../format'
import CandlestickPatternBadge from '../CandlestickPatternBadge'
import EntryTimingBadge from '../EntryTimingBadge'
import ImminentCrossBadge from '../ImminentCrossBadge'

const VERDICT_META = {
  comprar: { label: 'COMPRAR', tone: 'up' },
  esperar: { label: 'ESPERAR', tone: 'neutral' },
  evitar: { label: 'EVITAR', tone: 'down' },
}

// Mirrors BUY_THRESHOLD / AVOID_THRESHOLD in app/services/recommendation_engine.py.
const BUY_THRESHOLD = 5
const AVOID_THRESHOLD = -3
const GAUGE_MIN = -12
const GAUGE_MAX = 12
const GAUGE_RANGE = GAUGE_MAX - GAUGE_MIN

function zoneWidth(from, to) {
  return ((to - from) / GAUGE_RANGE) * 100
}

function scorePosition(score) {
  const clamped = Math.max(GAUGE_MIN, Math.min(GAUGE_MAX, score))
  return ((clamped - GAUGE_MIN) / GAUGE_RANGE) * 100
}

function ScoreGauge({ score }) {
  return (
    <div className="score-gauge">
      <div className="score-gauge__track">
        <div
          className="score-gauge__zone score-gauge__zone--down"
          style={{ width: `${zoneWidth(GAUGE_MIN, AVOID_THRESHOLD)}%` }}
        />
        <div
          className="score-gauge__zone score-gauge__zone--neutral"
          style={{ width: `${zoneWidth(AVOID_THRESHOLD, BUY_THRESHOLD)}%` }}
        />
        <div
          className="score-gauge__zone score-gauge__zone--up"
          style={{ width: `${zoneWidth(BUY_THRESHOLD, GAUGE_MAX)}%` }}
        />
        <div className="score-gauge__marker" style={{ left: `${scorePosition(score)}%` }} title={`Puntuación: ${score}`} />
      </div>
      <div className="score-gauge__labels">
        <span>Evitar (≤ {AVOID_THRESHOLD})</span>
        <span>Esperar</span>
        <span>Comprar (≥ {BUY_THRESHOLD})</span>
      </div>
    </div>
  )
}

function RecommendationCard({
  recommendation,
  entryTiming,
  imminentCross,
  imminentCrossShortTerm,
  candlestickPattern,
  currency,
  signContradictedFactors,
}) {
  const meta = VERDICT_META[recommendation.verdict] ?? { label: recommendation.verdict, tone: 'neutral' }
  const triggered = recommendation.factors.filter((f) => f.triggered)
  const mismatched = new Set(signContradictedFactors ?? [])
  // Tercera auditoría, Bloque H: la contradicción real es "¿qué fracción de lo
  // que empuja este veredicto tiene el signo medido en contra?", no un simple
  // sí/no - un factor contradicho de 8 y 6 de 8 son avisos muy distintos, y la
  // versión anterior los mostraba con el mismo banner fijo. Umbral de "mayoría"
  // (>=50%) es una decisión de presentación sobre evidencia ya medida, no un
  // peso nuevo de recommendation_engine.py - no requiere estudio de ablación.
  const mismatchedTriggered = triggered.filter((f) => mismatched.has(f.label))
  const mismatchFraction = triggered.length > 0 ? mismatchedTriggered.length / triggered.length : 0
  const mismatchIsMajority = mismatchFraction >= 0.5

  return (
    <div className={`recommendation-card recommendation-card--${meta.tone}`}>
      <div className="recommendation-card__verdict">
        <span className={`recommendation-card__badge recommendation-card__badge--${meta.tone}`}>{meta.label}</span>
        <span className="recommendation-card__score">Puntuación: {recommendation.score}</span>
      </div>

      <ScoreGauge score={recommendation.score} />

      {recommendation.verdict === 'comprar' && (
        <EntryTimingBadge entryTiming={entryTiming} showDescription />
      )}
      <ImminentCrossBadge imminentCross={imminentCross} verdict={recommendation.verdict} />
      <ImminentCrossBadge imminentCross={imminentCrossShortTerm} shortTerm verdict={recommendation.verdict} />
      <CandlestickPatternBadge pattern={candlestickPattern} />

      {recommendation.verdict === 'comprar' && (
        <div className="recommendation-card__levels">
          <div>
            <p className="recommendation-card__level-label">Stop-loss sugerido</p>
            <p className="recommendation-card__level-value delta-down">{formatCurrency(recommendation.stop_loss, currency)}</p>
          </div>
          <div>
            <p className="recommendation-card__level-label">Objetivo estimado</p>
            <p className="recommendation-card__level-value delta-up">{formatCurrency(recommendation.take_profit, currency)}</p>
            <p className="recommendation-card__level-hint">{recommendation.take_profit_method}</p>
          </div>
          <div>
            <p className="recommendation-card__level-label">Ratio riesgo/beneficio</p>
            <p className="recommendation-card__level-value">{formatRatio(recommendation.risk_reward)} : 1</p>
          </div>
        </div>
      )}

      {mismatchedTriggered.length > 0 && (
        <div
          className={`recommendation-card__sign-warning ${
            mismatchIsMajority ? 'recommendation-card__sign-warning--majority' : 'recommendation-card__sign-warning--minor'
          }`}
        >
          <strong>
            ⚠️ {mismatchedTriggered.length} de {triggered.length} factores activos en este veredicto{' '}
            {mismatchIsMajority ? '(la mayoría)' : '(una minoría)'} tienen el signo contrario al medido por el
            estudio de ablación
          </strong>{' '}
          - no se ha corregido el peso, solo se muestra la contradicción. Marcados con ⚠️ abajo. Ver Metodología
          §16.
        </div>
      )}

      <ul className="recommendation-card__factors">
        {triggered.map((f) => (
          <li key={f.label} className={f.points >= 0 ? 'delta-up' : 'delta-down'}>
            <span>{f.points >= 0 ? '▲' : '▼'}</span> {f.label} ({f.points >= 0 ? '+' : ''}
            {f.points}){mismatched.has(f.label) && <span className="system-performance__mismatch-tag"> ⚠️ signo contrario medido</span>}
          </li>
        ))}
        {triggered.length === 0 && <li className="empty-state">Sin factores técnicos destacables ahora mismo.</li>}
      </ul>
      <p className="recommendation-card__disclaimer">
        Puntuación transparente basada en reglas técnicas propias (tendencia, fases de Weinstein, Minervini, RS
        Rating, ADX, RSI, soportes/resistencias, divergencia de volumen OBV, cadena de Markov, volatilidad GARCH,
        crecimiento y rentabilidad fundamental). Los pesos se calibran contra un estudio de ablación estadístico
        sobre el universo completo de tickers, no a ojo - ver metodología. No es asesoramiento financiero.
      </p>
    </div>
  )
}

export default RecommendationCard
