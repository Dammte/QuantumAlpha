import { useEffect, useState } from 'react'
import { api } from '../api'
import { SIGNAL_LABELS, formatNumber, formatPercent, formatRelativeTime } from '../format'

const VERDICT_LABELS = { comprar: 'Comprar', esperar: 'Esperar', evitar: 'Evitar' }

const MAX_FALSE_NEGATIVES_SHOWN = 25

function outcomeLabel(kind, label) {
  return kind === 'verdict' ? (VERDICT_LABELS[label] ?? label) : (SIGNAL_LABELS[label] ?? label)
}

function OutcomeTable({ title, hint, kind, outcomes }) {
  const sorted = [...outcomes].sort((a, b) => a.label.localeCompare(b.label) || a.horizon_days - b.horizon_days)
  return (
    <section className="panel panel--nested">
      <h3>{title}</h3>
      <p className="system-performance__hint">{hint}</p>
      {sorted.length === 0 ? (
        <p className="empty-state">Todavía no hay historial suficiente para calcular esto.</p>
      ) : (
        <div className="table-scroll">
          <table className="system-performance__table">
            <thead>
              <tr>
                <th>Señal</th>
                <th className="num">Horizonte</th>
                <th className="num">N</th>
                <th className="num">Hit rate</th>
                <th className="num">Retorno medio</th>
                <th className="num">Retorno mediano</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((o) => (
                <tr key={`${o.label}-${o.horizon_days}`}>
                  <td>{outcomeLabel(kind, o.label)}</td>
                  <td className="num">{o.horizon_days} sesiones</td>
                  <td className="num">{o.n}</td>
                  <td className="num">{o.hit_rate !== null ? formatPercent(o.hit_rate) : '—'}</td>
                  <td className={`num ${(o.mean_return ?? 0) >= 0 ? 'delta-up' : 'delta-down'}`}>
                    {o.mean_return !== null ? formatPercent(o.mean_return, { signed: true }) : '—'}
                  </td>
                  <td className={`num ${(o.median_return ?? 0) >= 0 ? 'delta-up' : 'delta-down'}`}>
                    {o.median_return !== null ? formatPercent(o.median_return, { signed: true }) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

// Fase 0 (docs/quant_methodology.md): did the system's past verdicts and
// position signals actually work out, measured honestly against what
// actually happened afterward - not a self-report. See
// signal_performance_service.py; this view is its only consumer.
function SystemPerformanceView() {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setReport(await api.getSignalPerformance())
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <p className="empty-state">Calculando rendimiento histórico…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (!report) return null

  const falseNegatives = [...report.false_negatives]
    .sort((a, b) => new Date(b.snapshot_at) - new Date(a.snapshot_at))
    .slice(0, MAX_FALSE_NEGATIVES_SHOWN)

  return (
    <div>
      <p className="system-performance__updated">Calculado {formatRelativeTime(report.as_of) ?? 'ahora'}</p>

      <OutcomeTable
        title="Por veredicto (comprar / esperar / evitar)"
        hint="Retorno realizado a N sesiones desde cada recomendación guardada - historial disponible desde que se añadió la trazabilidad de señales."
        kind="verdict"
        outcomes={report.verdict_outcomes}
      />

      <OutcomeTable
        title="Por señal de posición (vender / aumentar / vigilar / mantener)"
        hint="Solo cubre lo ocurrido desde que este motor de salida empezó a registrar cada evaluación fresca - no se puede reconstruir hacia atrás, ese dato nunca se guardó antes."
        kind="signal"
        outcomes={report.signal_outcomes}
      />

      <section className="panel panel--nested">
        <h3>Falsos negativos ("mantener" seguido de una caída real)</h3>
        <p className="system-performance__hint">
          Posiciones marcadas "mantener" que en los 10 días siguientes cayeron más de un 5% - el error concreto que
          motivó este rediseño del motor de salida.
        </p>
        {falseNegatives.length === 0 ? (
          <p className="empty-state">Ninguno registrado todavía.</p>
        ) : (
          <div className="table-scroll">
            <table className="system-performance__table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Fecha de la señal</th>
                  <th className="num" title="En la divisa propia del ticker - no se guarda la divisa junto al snapshot, así que se muestra sin símbolo">
                    Precio en la señal
                  </th>
                  <th className="num">Precio después</th>
                  <th className="num">Retorno</th>
                  <th className="num">Horizonte</th>
                </tr>
              </thead>
              <tbody>
                {falseNegatives.map((fn) => (
                  <tr key={`${fn.portfolio_id}-${fn.ticker}-${fn.snapshot_at}`}>
                    <td>{fn.ticker}</td>
                    <td>{new Date(fn.snapshot_at).toLocaleDateString('es-ES')}</td>
                    <td className="num">{formatNumber(fn.price_at_signal)}</td>
                    <td className="num">{formatNumber(fn.price_after)}</td>
                    <td className="num delta-down">{formatPercent(fn.return_pct, { signed: true })}</td>
                    <td className="num">{fn.horizon_days} sesiones</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {report.false_negatives.length > MAX_FALSE_NEGATIVES_SHOWN && (
          <p className="system-performance__hint">
            Mostrando los {MAX_FALSE_NEGATIVES_SHOWN} más recientes de {report.false_negatives.length}.
          </p>
        )}
      </section>
    </div>
  )
}

export default SystemPerformanceView
