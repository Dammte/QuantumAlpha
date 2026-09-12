// Reconstruction (2026-09), Fase 7: renders TickerAnalysisResponse.llm_narrative
// - a Gemini-generated, plain-language explanation of the gate result above
// it, never a second opinion or a decision of its own (see
// LLMNarrator's own docstring on the backend). `null` whenever
// GEMINI_API_KEY isn't configured (the default) or the call failed for any
// reason - renders nothing at all in that case, same as every other
// optional field in this app.
function GateNarrative({ narrative }) {
  if (!narrative) return null

  return (
    <div className="gate-narrative">
      <p className="gate-narrative__label">Explicación en lenguaje natural (IA)</p>
      <p className="gate-narrative__text">{narrative}</p>
      <p className="gate-narrative__disclaimer">
        Generada a partir del gate ya calculado arriba - no es una segunda opinión ni cambia el resultado.
      </p>
    </div>
  )
}

export default GateNarrative
