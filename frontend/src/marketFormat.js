export const TREND_LABELS = {
  uptrend: 'Alcista',
  downtrend: 'Bajista',
  sideways: 'Lateral',
}

export const TREND_TONE = {
  uptrend: 'up',
  downtrend: 'down',
  sideways: 'neutral',
}

export function trendLabel(trend) {
  return TREND_LABELS[trend] ?? trend
}

// technical_analysis.Stage - Weinstein's 4-stage cycle.
export const STAGE_LABELS = {
  stage1: 'Fase 1 (base)',
  stage2: 'Fase 2 (avance)',
  stage3: 'Fase 3 (techo)',
  stage4: 'Fase 4 (declive)',
}

export function stageLabel(stage) {
  return stage ? (STAGE_LABELS[stage] ?? stage) : '—'
}

// setups.types.SetupStage - Parte 2 en adelante de la biblioteca de setups
// del Radar (docs/quant_methodology.md §28). Nunca "probabilidad", solo en
// qué punto de su formación está el setup.
export const SETUP_STAGE_LABELS = {
  forming: 'Formándose',
  ready: 'Listo',
  triggered: 'Disparado',
  failed: 'Fallido',
}

export function setupStageLabel(stage) {
  return SETUP_STAGE_LABELS[stage] ?? stage
}

// setups.types.SetupFamily
export const SETUP_FAMILY_LABELS = {
  stage_transition: 'Transición de etapa',
  vcp: 'VCP',
  breakout: 'Ruptura',
  pullback: 'Retroceso',
  ma_cross: 'Cruce rápido',
  channel: 'Canal',
  classic_pattern: 'Patrón clásico',
}

export function setupFamilyLabel(family) {
  return SETUP_FAMILY_LABELS[family] ?? family
}

// levels_engine.Grade - geometría, no probabilidad (Parte 5.3). "C" usa el
// mismo tono ámbar que sector-tier-badge--warn (no rojo: "pasa el mínimo de
// viabilidad pero con concesiones" no es lo mismo que "vender").
export const GRADE_TONE = { A: 'buy', B: 'neutral', C: 'warn' }

export function gradeTone(grade) {
  return GRADE_TONE[grade] ?? 'neutral'
}
