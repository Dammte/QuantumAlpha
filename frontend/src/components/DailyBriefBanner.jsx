import { formatRelativeTime } from '../format'

// Reconstruction (2026-09), Fase 5: the one genuinely new thing GET
// /portfolios/{id}/today adds over the existing live `/risk` read that
// TodayActionsPanel already covers - `daily_brief_repo`'s own
// `new_entry_triggers`/`new_gate_passes` are universe-wide counts computed
// once by daily_close.py, not derivable from this portfolio's own held
// positions. Position-level urgency stays TodayActionsPanel's job (richer
// per-ticker reasons, computed live) - this banner never repeats that,
// just the headline and what changed across the universe since the last
// look. Renders nothing while daily_close.py hasn't run yet for this
// portfolio (`brief` is `null`), same as every other precompute-backed
// panel in this app.
function DailyBriefBanner({ brief }) {
  if (!brief) return null

  const relative = formatRelativeTime(brief.computed_at)

  return (
    <section className="panel daily-brief-banner">
      <p className="daily-brief-banner__headline">{brief.headline}</p>
      <p className="daily-brief-banner__stats">
        {brief.new_entry_triggers} nuevo{brief.new_entry_triggers === 1 ? '' : 's'} disparador
        {brief.new_entry_triggers === 1 ? '' : 'es'} de entrada · {brief.new_gate_passes} nuevo
        {brief.new_gate_passes === 1 ? '' : 's'} gate{brief.new_gate_passes === 1 ? '' : 's'} aprobado
        {brief.new_gate_passes === 1 ? '' : 's'} en el universo hoy
        {relative && <span className="daily-brief-banner__updated"> · {relative}</span>}
      </p>
    </section>
  )
}

export default DailyBriefBanner
