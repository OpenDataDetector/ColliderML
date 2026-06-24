<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  store, STAGE_ORDER, STAGE_META, CHANNELS, MADGRAPH_CHANNELS,
  addStage, removeStage, moveStage, validate, presetWorkflow, resetWorkflow,
  toSimulatePayload, type StageType,
} from '../composables/workflow'

const addType = ref<StageType>('geometry')
const issues = computed(() => validate())
const errors = computed(() => issues.value.filter(i => i.level === 'error'))

const submitting = ref(false)
const submitMsg = ref('')

function onAdd() { addStage(addType.value) }

async function submit() {
  const payload = toSimulatePayload()
  if (!payload) { submitMsg.value = 'Add a Generation stage first.'; return }
  if (errors.value.length) { submitMsg.value = 'Fix validation errors before submitting.'; return }
  submitting.value = true
  submitMsg.value = 'Submitting…'
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (store.hfToken) headers['Authorization'] = `Bearer ${store.hfToken}`
    const r = await fetch('/v1/simulate', { method: 'POST', headers, body: JSON.stringify(payload) })
    const data = await r.json()
    if (!r.ok) {
      submitMsg.value = `❌ ${r.status}: ${data.detail || JSON.stringify(data)}`
    } else {
      store.lastSubmission = data
      submitMsg.value = `✅ Submitted ${payload.channel} ×${payload.events} (pu${payload.pileup}) — request ${data.request_id}, state ${data.state}, ${data.credits_charged} credits.`
    }
  } catch (e: any) {
    submitMsg.value = `❌ ${e.message} (is the backend running on :8000?)`
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="wb">
    <div class="wb-toolbar">
      <strong>Pipeline</strong>
      <span class="wb-spacer" />
      <button class="wb-btn" @click="presetWorkflow('ttbar', 200)">ttbar preset</button>
      <button class="wb-btn" @click="resetWorkflow()">Clear</button>
    </div>

    <div v-if="!store.stages.length" class="wb-empty">
      No stages yet. Add one below, or ask the copilot to build a pipeline for you.
    </div>

    <div class="wb-flow">
      <template v-for="(stage, i) in store.stages" :key="stage.id">
        <div class="wb-card" :class="'t-' + stage.type">
          <div class="wb-card-head">
            <span class="wb-badge">{{ i + 1 }}</span>
            <span class="wb-card-title">{{ STAGE_META[stage.type].label }}</span>
            <span class="wb-spacer" />
            <button class="wb-icon" title="Move up" :disabled="i === 0" @click="moveStage(stage.id, -1)">↑</button>
            <button class="wb-icon" title="Move down" :disabled="i === store.stages.length - 1" @click="moveStage(stage.id, 1)">↓</button>
            <button class="wb-icon wb-x" title="Remove" @click="removeStage(stage.id)">✕</button>
          </div>
          <div class="wb-card-body">
            <p class="wb-blurb">{{ STAGE_META[stage.type].blurb }}</p>

            <template v-if="stage.type === 'geometry'">
              <label>Detector
                <select v-model="stage.params.detector">
                  <option value="ODD">ODD (Open Data Detector)</option>
                </select>
              </label>
            </template>

            <template v-else-if="stage.type === 'generation'">
              <label>Channel
                <select v-model="stage.params.channel">
                  <option v-for="c in CHANNELS" :key="c" :value="c">{{ c }}</option>
                </select>
              </label>
              <label>Generator
                <select v-model="stage.params.generator">
                  <option value="madgraph">madgraph</option>
                  <option value="pythia">pythia</option>
                </select>
              </label>
              <label>Events
                <input type="number" min="1" max="100000" v-model.number="stage.params.events" />
              </label>
              <label>Seed
                <input type="number" v-model.number="stage.params.seed" />
              </label>
              <p v-if="stage.params.channel && stage.params.generator &&
                       stage.params.generator !== (MADGRAPH_CHANNELS.includes(stage.params.channel) ? 'madgraph' : 'pythia')"
                 class="wb-hint">
                Tip: {{ stage.params.channel }} is normally
                {{ MADGRAPH_CHANNELS.includes(stage.params.channel) ? 'madgraph' : 'pythia' }}.
              </p>
            </template>

            <template v-else-if="stage.type === 'simulation'">
              <label>Pileup ⟨μ⟩
                <input type="number" min="0" max="200" v-model.number="stage.params.pileup" />
              </label>
            </template>

            <template v-else>
              <p class="wb-blurb wb-muted">No parameters — uses ODD defaults.</p>
            </template>
          </div>
        </div>
        <div v-if="i < store.stages.length - 1" class="wb-arrow">→</div>
      </template>
    </div>

    <div class="wb-add">
      <select v-model="addType">
        <option v-for="t in STAGE_ORDER" :key="t" :value="t">{{ STAGE_META[t].label }}</option>
      </select>
      <button class="wb-btn wb-add-btn" @click="onAdd">+ Add stage</button>
    </div>

    <div class="wb-validate" v-if="store.stages.length">
      <strong>Validation</strong>
      <p v-if="!issues.length" class="wb-ok">✓ Workflow looks sound.</p>
      <ul v-else>
        <li v-for="(issue, k) in issues" :key="k" :class="issue.level === 'error' ? 'wb-err' : 'wb-warn'">
          {{ issue.level === 'error' ? '✗' : '⚠' }} {{ issue.message }}
        </li>
      </ul>
    </div>

    <div class="wb-submit">
      <details class="wb-token">
        <summary>HuggingFace token (for live submit)</summary>
        <input type="password" placeholder="hf_… (kept in your browser only)" v-model="store.hfToken" />
        <p class="wb-blurb wb-muted">Verified by the backend against huggingface.co; needed to authorize <code>/v1/simulate</code>.</p>
      </details>
      <button class="wb-btn wb-primary" :disabled="submitting || !!errors.length" @click="submit">
        Submit to SaaS backend →
      </button>
      <p v-if="submitMsg" class="wb-submitmsg">{{ submitMsg }}</p>
    </div>
  </div>
</template>

<style scoped>
.wb { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 16px; margin: 16px 0; background: var(--vp-c-bg-soft); }
.wb-toolbar { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.wb-spacer { flex: 1; }
.wb-empty { color: var(--vp-c-text-2); font-size: 14px; padding: 12px 0; }
.wb-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; }
.wb-card { flex: 1 1 200px; min-width: 200px; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg); overflow: hidden; }
.wb-card.t-geometry { border-top: 3px solid #8b5cf6; }
.wb-card.t-generation { border-top: 3px solid #3b82f6; }
.wb-card.t-simulation { border-top: 3px solid #10b981; }
.wb-card.t-digitization { border-top: 3px solid #f59e0b; }
.wb-card.t-reconstruction { border-top: 3px solid #ef4444; }
.wb-card-head { display: flex; align-items: center; gap: 6px; padding: 8px 10px; border-bottom: 1px solid var(--vp-c-divider); }
.wb-badge { background: var(--vp-c-brand-1); color: #fff; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; }
.wb-card-title { font-weight: 600; font-size: 14px; }
.wb-card-body { padding: 10px; }
.wb-blurb { font-size: 12px; color: var(--vp-c-text-2); margin: 0 0 8px; }
.wb-muted { font-style: italic; }
.wb-hint { font-size: 12px; color: #f59e0b; margin: 6px 0 0; }
.wb-card-body label { display: block; font-size: 12px; color: var(--vp-c-text-2); margin-bottom: 6px; }
.wb-card-body select, .wb-card-body input { width: 100%; margin-top: 2px; padding: 4px 6px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); font-size: 13px; }
.wb-arrow { display: flex; align-items: center; color: var(--vp-c-text-3); font-size: 20px; }
.wb-icon { background: none; border: none; cursor: pointer; color: var(--vp-c-text-2); font-size: 14px; padding: 2px 4px; }
.wb-icon:disabled { opacity: 0.3; cursor: default; }
.wb-x:hover { color: #ef4444; }
.wb-add { display: flex; gap: 8px; margin-top: 14px; }
.wb-add select { padding: 6px 8px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); }
.wb-btn { padding: 6px 12px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); cursor: pointer; font-size: 13px; }
.wb-btn:hover { border-color: var(--vp-c-brand-1); }
.wb-primary { background: var(--vp-c-brand-1); color: #fff; border-color: var(--vp-c-brand-1); }
.wb-primary:disabled { opacity: 0.5; cursor: default; }
.wb-validate { margin-top: 14px; font-size: 13px; }
.wb-validate ul { margin: 6px 0 0; padding-left: 18px; }
.wb-ok { color: #10b981; }
.wb-err { color: #ef4444; }
.wb-warn { color: #f59e0b; }
.wb-submit { margin-top: 16px; border-top: 1px solid var(--vp-c-divider); padding-top: 14px; }
.wb-token { margin-bottom: 10px; font-size: 13px; }
.wb-token input { width: 100%; margin-top: 6px; padding: 6px 8px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); }
.wb-submitmsg { margin-top: 10px; font-size: 13px; font-family: var(--vp-font-family-mono); }
</style>
