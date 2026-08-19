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
