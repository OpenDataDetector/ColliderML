// Shared reactive workflow state for the ColliderML pipeline builder.
//
// Both <WorkflowBuilder> (the visual cards) and <Copilot> (the LLM agent)
// import this singleton store, so the agent's tool calls mutate exactly the
// same pipeline the user sees and edits. This is what lets the copilot
// "drive the builder."

import { reactive } from 'vue'

export type StageType =
  | 'geometry'
  | 'generation'
  | 'simulation'
  | 'digitization'
  | 'reconstruction'

export interface Stage {
  id: string
  type: StageType
  params: Record<string, any>
}

export const STAGE_ORDER: StageType[] = [
  'geometry',
  'generation',
  'simulation',
  'digitization',
  'reconstruction',
]

export const CHANNELS = [
  'higgs_portal', 'ttbar', 'zmumu', 'zee', 'diphoton',
  'jets', 'susy_gmsb', 'hidden_valley', 'zprime', 'single_muon',
]

// Hard-process channels are MadGraph→Pythia; the rest are Pythia-only.
export const MADGRAPH_CHANNELS = ['ttbar', 'susy_gmsb', 'zprime', 'hidden_valley', 'jets']

export const STAGE_META: Record<StageType, { label: string; blurb: string }> = {
  geometry: { label: 'Geometry', blurb: 'Detector description (DD4hep)' },
  generation: { label: 'Generation', blurb: 'Event generation (MadGraph / Pythia)' },
  simulation: { label: 'Simulation', blurb: 'Geant4 detector simulation (ddsim)' },
  digitization: { label: 'Digitization', blurb: 'ACTS digitization' },
  reconstruction: { label: 'Reconstruction', blurb: 'ACTS seeding + CKF track finding' },
}

function defaultParams(type: StageType): Record<string, any> {
  switch (type) {
    case 'geometry': return { detector: 'ODD' }
    case 'generation': return { generator: 'madgraph', channel: 'ttbar', events: 100, seed: 42 }
    case 'simulation': return { pileup: 0 }
    default: return {}
  }
}

let _counter = 0
function newId(type: StageType): string {
  _counter += 1
  return `${type}-${_counter}`
}

export interface Issue { level: 'error' | 'warn'; message: string }

export const store = reactive({
  stages: [] as Stage[],
  // HF token for authenticated /v1/simulate submits (kept only in the browser).
  hfToken: '',
  lastSubmission: null as null | Record<string, any>,
})

// --- Mutations (also used by the copilot's tools) -------------------------

export function addStage(type: StageType, params?: Record<string, any>, position?: number): Stage {
  const stage: Stage = { id: newId(type), type, params: { ...defaultParams(type), ...(params || {}) } }
  if (position == null || position < 0 || position >= store.stages.length) {
    store.stages.push(stage)
  } else {
    store.stages.splice(position, 0, stage)
  }
  return stage
}

export function removeStage(id: string): boolean {
  const i = store.stages.findIndex(s => s.id === id)
  if (i === -1) return false
  store.stages.splice(i, 1)
  return true
}

export function updateStage(id: string, params: Record<string, any>): boolean {
  const s = store.stages.find(s => s.id === id)
  if (!s) return false
  s.params = { ...s.params, ...params }
  return true
}

export function reorder(order: string[]): boolean {
  const byId = new Map(store.stages.map(s => [s.id, s]))
  const next: Stage[] = []
  for (const id of order) {
    const s = byId.get(id)
    if (s) { next.push(s); byId.delete(id) }
  }
  // Keep any stages the order list forgot, at the end.
  for (const s of byId.values()) next.push(s)
  store.stages = next
  return true
}

export function moveStage(id: string, dir: -1 | 1): void {
  const i = store.stages.findIndex(s => s.id === id)
  const j = i + dir
  if (i === -1 || j < 0 || j >= store.stages.length) return
  const [s] = store.stages.splice(i, 1)
  store.stages.splice(j, 0, s)
}

export function resetWorkflow(): void {
  store.stages = []
}

export function presetWorkflow(channel = 'ttbar', pileup = 200): void {
  resetWorkflow()
  const generator = MADGRAPH_CHANNELS.includes(channel) ? 'madgraph' : 'pythia'
  addStage('geometry')
  addStage('generation', { generator, channel, events: 100, seed: 42 })
  addStage('simulation', { pileup })
  addStage('digitization')
  addStage('reconstruction')
}

// --- Validation -----------------------------------------------------------

export function validate(): Issue[] {
  const issues: Issue[] = []
  const types = store.stages.map(s => s.type)

  // Ordering
  let lastRank = -1
  for (const s of store.stages) {
    const rank = STAGE_ORDER.indexOf(s.type)
    if (rank < lastRank) {
      issues.push({ level: 'error', message: `Stage "${STAGE_META[s.type].label}" is out of order — expected ${STAGE_ORDER.join(' → ')}.` })
      break
    }
    lastRank = Math.max(lastRank, rank)
  }

  // Dependencies
  const has = (t: StageType) => types.includes(t)
  if (has('simulation') && !has('generation'))
    issues.push({ level: 'error', message: 'Simulation requires a preceding Generation stage.' })
  if (has('digitization') && !has('simulation'))
    issues.push({ level: 'error', message: 'Digitization requires a Simulation stage.' })
  if (has('reconstruction') && !has('digitization'))
    issues.push({ level: 'error', message: 'Reconstruction requires a Digitization stage.' })

  // Per-stage param checks
  for (const s of store.stages) {
    if (s.type === 'generation') {
      const { channel, generator, events } = s.params
      if (!CHANNELS.includes(channel))
        issues.push({ level: 'error', message: `Unknown channel "${channel}".` })
      const wanted = MADGRAPH_CHANNELS.includes(channel) ? 'madgraph' : 'pythia'
      if (generator && generator !== wanted)
        issues.push({ level: 'warn', message: `Channel "${channel}" is normally ${wanted}; you selected ${generator}.` })
      if (events == null || events < 1 || events > 100000)
        issues.push({ level: 'error', message: `Events must be between 1 and 100000 (got ${events}).` })
    }
    if (s.type === 'simulation') {
      const pu = s.params.pileup
      if (pu == null || pu < 0 || pu > 200)
        issues.push({ level: 'error', message: `Pileup must be between 0 and 200 (got ${pu}).` })
      else if (pu > 100)
        issues.push({ level: 'warn', message: `Pileup ${pu} is very expensive to simulate.` })
    }
  }

  // Completeness
  if (!has('generation') || !has('simulation'))
    issues.push({ level: 'warn', message: 'A runnable workflow needs at least Generation + Simulation.' })

  return issues
}

// Build the /v1/simulate payload from the assembled pipeline.
export function toSimulatePayload(): { channel: string; events: number; pileup: number; seed: number } | null {
  const gen = store.stages.find(s => s.type === 'generation')
  const sim = store.stages.find(s => s.type === 'simulation')
  if (!gen) return null
  return {
    channel: gen.params.channel,
    events: Number(gen.params.events ?? 10),
    pileup: Number(sim?.params.pileup ?? 0),
    seed: Number(gen.params.seed ?? 42),
  }
}

export function snapshot(): Record<string, any> {
  return { stages: store.stages.map(s => ({ id: s.id, type: s.type, params: s.params })) }
}
