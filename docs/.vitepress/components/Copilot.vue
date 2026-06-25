<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vitepress'
import {
  store, snapshot, addStage, removeStage, updateStage, reorder, validate,
  toSimulatePayload,
} from '../composables/workflow'
import { apiUrl } from '../composables/api'

const route = useRoute()
const isBuilder = computed(() => route.path.includes('builder'))
const mode = computed(() => (isBuilder.value ? 'builder' : 'docs'))

const open = ref(false)
const busy = ref(false)
const input = ref('')
const scroller = ref<HTMLElement | null>(null)
const configured = ref<boolean | null>(null)
const providerLabel = ref('')
const waking = ref(false)
const dotClass = computed(() => (waking.value ? 'waking' : configured.value ? 'on' : 'off'))

// Full API transcript (Anthropic block format) + a render-friendly view.
type Msg = { role: 'user' | 'assistant'; content: any }
const apiMessages = ref<Msg[]>([])

interface Bubble { who: 'you' | 'copilot' | 'tool'; text: string }
const bubbles = computed<Bubble[]>(() => {
  const out: Bubble[] = []
  for (const m of apiMessages.value) {
    if (typeof m.content === 'string') {
      out.push({ who: m.role === 'user' ? 'you' : 'copilot', text: m.content })
    } else if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b.type === 'text' && b.text) out.push({ who: 'copilot', text: b.text })
        else if (b.type === 'tool_use') out.push({ who: 'tool', text: `🔧 ${b.name}(${JSON.stringify(b.input)})` })
        // tool_result blocks (role user) are internal; not shown.
      }
    }
  }
  return out
})

async function pingHealth(timeoutMs: number): Promise<any | null> {
  try {
    const ctrl = new AbortController()
    const t = setTimeout(() => ctrl.abort(), timeoutMs)
    const r = await fetch(apiUrl('/v1/chat/health'), { signal: ctrl.signal })
    clearTimeout(t)
    return r.ok ? await r.json() : null
  } catch {
    return null
  }
}

async function checkHealth() {
  // The backend is on Render's free tier and spins down when idle, so the first
  // request after a while can take ~30-60s to cold-start. Show a "waking" state
  // and keep polling rather than flashing "unreachable".
  let d = await pingHealth(8000)
  if (!d) {
    waking.value = true
    providerLabel.value = 'waking the backend…'
    const deadline = Date.now() + 60000
    while (!d && Date.now() < deadline) {
      await new Promise((res) => setTimeout(res, 5000))
      d = await pingHealth(8000)
    }
    waking.value = false
  }
  if (d) {
    configured.value = d.configured
    providerLabel.value = d.configured ? `${d.provider}/${d.model}` : 'not configured'
  } else {
    configured.value = false
    providerLabel.value = 'backend unreachable'
  }
}
onMounted(checkHealth)

watch(bubbles, async () => {
  await nextTick()
  if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight
})

async function execTool(name: string, args: any): Promise<any> {
  switch (name) {
    case 'get_workflow': return snapshot()
    case 'add_stage': { const s = addStage(args.type, args.params, args.position); return { ok: true, added: s.id, workflow: snapshot() } }
    case 'update_stage': return { ok: updateStage(args.id, args.params || {}), workflow: snapshot() }
    case 'remove_stage': return { ok: removeStage(args.id), workflow: snapshot() }
    case 'reorder_pipeline': return { ok: reorder(args.order || []), workflow: snapshot() }
    case 'validate_workflow': return { issues: validate() }
    case 'submit_workflow': return await doSubmit()
    default: return { error: `unknown tool ${name}` }
  }
}

async function doSubmit(): Promise<any> {
  const payload = toSimulatePayload()
  if (!payload) return { error: 'no Generation stage; nothing to submit' }
  const issues = validate().filter(i => i.level === 'error')
  if (issues.length) return { error: 'validation errors', issues }
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (store.hfToken) headers['Authorization'] = `Bearer ${store.hfToken}`
  const r = await fetch(apiUrl('/v1/simulate'), { method: 'POST', headers, body: JSON.stringify(payload) })
  const data = await r.json()
  if (r.ok) store.lastSubmission = data
  return { status: r.status, ...data }
}

async function send() {
  const text = input.value.trim()
  if (!text || busy.value) return
  input.value = ''
  apiMessages.value.push({ role: 'user', content: text })
  busy.value = true
  try {
    await runLoop()
  } catch (e: any) {
    apiMessages.value.push({ role: 'assistant', content: [{ type: 'text', text: `⚠ ${e.message}` }] })
  } finally {
    busy.value = false
  }
}

async function runLoop() {
  for (let i = 0; i < 6; i++) {
    const body = {
      mode: mode.value,
      workflow: snapshot(),
      messages: apiMessages.value,
    }
    const r = await fetch(apiUrl('/v1/chat'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    })
    const data = await r.json()
    if (!r.ok) {
      // e.g. 429 rate-limit / daily cap — surface the backend's message.
      const msg = data?.detail || `request failed (${r.status})`
      apiMessages.value.push({ role: 'assistant', content: [{ type: 'text', text: `⚠ ${msg}` }] })
      return
    }
    // Keep only the fields valid for each block type. The backend returns blocks
    // with all fields populated (text blocks carry id/name/input = null); echoing
    // those back into the next turn makes Anthropic reject "extra inputs".
    const blocks = (data.content || []).map((b: any) => {
      if (b.type === 'tool_use') return { type: 'tool_use', id: b.id, name: b.name, input: b.input ?? {} }
      if (b.type === 'text') return { type: 'text', text: b.text ?? '' }
      return b
    })
    apiMessages.value.push({ role: 'assistant', content: blocks })

    const toolUses = blocks.filter((b: any) => b.type === 'tool_use')
    if (!toolUses.length) return

    const results: any[] = []
    for (const tu of toolUses) {
      const result = await execTool(tu.name, tu.input || {})
      results.push({ type: 'tool_result', tool_use_id: tu.id, content: JSON.stringify(result) })
    }
    apiMessages.value.push({ role: 'user', content: results })
    // loop again so the model can react to tool results
  }
}

const placeholder = computed(() =>
  isBuilder.value
    ? 'e.g. "build a ttbar pu200 workflow and validate it"'
    : 'Ask about the library, CLI, simulation, tasks…')
</script>

<template>
  <div class="cp">
    <button v-if="!open" class="cp-fab" @click="open = true" title="ColliderML copilot">
      <span class="cp-fab-dot" :class="dotClass" />
      💬 Copilot
    </button>

    <div v-else class="cp-panel">
      <div class="cp-head">
        <strong>ColliderML Copilot</strong>
        <span class="cp-mode">{{ isBuilder ? 'builder' : 'docs' }}</span>
        <span class="cp-spacer" />
        <button class="cp-x" @click="open = false">✕</button>
      </div>

      <div class="cp-sub">
        <span class="cp-fab-dot" :class="dotClass" />
        <span>{{ providerLabel }}</span>
      </div>

      <div ref="scroller" class="cp-body">
        <p v-if="!bubbles.length" class="cp-hello">
          {{ isBuilder
            ? "I can read and edit the pipeline on this page. Try: \"add geometry, generation (ttbar) and simulation at pu200, then validate\"."
            : "Ask me anything about ColliderML — loading data, local/remote simulation, the benchmark tasks, the CLI." }}
        </p>
        <div v-for="(b, k) in bubbles" :key="k" class="cp-msg" :class="'who-' + b.who">
          <div class="cp-bubble">{{ b.text }}</div>
        </div>
        <div v-if="busy" class="cp-msg who-copilot"><div class="cp-bubble cp-typing">…</div></div>
      </div>

      <div class="cp-input">
        <textarea v-model="input" :placeholder="placeholder" rows="2"
          @keydown.enter.exact.prevent="send" />
        <button class="cp-send" :disabled="busy || !input.trim()" @click="send">↑</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cp { position: fixed; right: 20px; bottom: 20px; z-index: 100; }
.cp-fab { display: flex; align-items: center; gap: 8px; padding: 10px 16px; border-radius: 24px; border: 1px solid var(--vp-c-divider); background: var(--vp-c-bg); color: var(--vp-c-text-1); cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,0.15); font-size: 14px; }
.cp-fab:hover { border-color: var(--vp-c-brand-1); }
.cp-fab-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.cp-fab-dot.on { background: #10b981; }
.cp-fab-dot.off { background: #ef4444; }
.cp-fab-dot.waking { background: #f59e0b; animation: cp-pulse 1s ease-in-out infinite; }
@keyframes cp-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.cp-panel { width: 380px; max-width: calc(100vw - 40px); height: 540px; max-height: calc(100vh - 40px); display: flex; flex-direction: column; border: 1px solid var(--vp-c-divider); border-radius: 14px; background: var(--vp-c-bg); box-shadow: 0 8px 30px rgba(0,0,0,0.25); overflow: hidden; }
.cp-head { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-bottom: 1px solid var(--vp-c-divider); }
.cp-mode { font-size: 11px; padding: 1px 8px; border-radius: 10px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-2); }
.cp-spacer { flex: 1; }
.cp-x { background: none; border: none; cursor: pointer; color: var(--vp-c-text-2); font-size: 14px; }
.cp-sub { display: flex; align-items: center; gap: 6px; padding: 4px 14px; font-size: 11px; color: var(--vp-c-text-3); border-bottom: 1px solid var(--vp-c-divider); }
.cp-body { flex: 1; overflow-y: auto; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
.cp-hello { font-size: 13px; color: var(--vp-c-text-2); }
.cp-msg { display: flex; }
.cp-msg.who-you { justify-content: flex-end; }
.cp-bubble { max-width: 85%; padding: 8px 11px; border-radius: 12px; font-size: 13px; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
.who-you .cp-bubble { background: var(--vp-c-brand-1); color: #fff; border-bottom-right-radius: 4px; }
.who-copilot .cp-bubble { background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); border-bottom-left-radius: 4px; }
.who-tool .cp-bubble { background: transparent; color: var(--vp-c-text-3); font-family: var(--vp-font-family-mono); font-size: 11px; padding: 2px 6px; border: 1px dashed var(--vp-c-divider); }
.cp-typing { letter-spacing: 2px; }
.cp-input { display: flex; gap: 6px; padding: 10px; border-top: 1px solid var(--vp-c-divider); }
.cp-input textarea { flex: 1; resize: none; border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 6px 8px; background: var(--vp-c-bg); color: var(--vp-c-text-1); font-size: 13px; font-family: inherit; }
.cp-send { width: 36px; border: none; border-radius: 8px; background: var(--vp-c-brand-1); color: #fff; cursor: pointer; font-size: 16px; }
.cp-send:disabled { opacity: 0.5; cursor: default; }
</style>
