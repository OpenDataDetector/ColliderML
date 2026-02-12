<script setup>
import { ref, computed, onMounted } from 'vue'

const selections = ref({
  channels: ['ttbar'],
  pileup: 'pu0',
  objects: ['particles', 'tracker_hits', 'calo_hits', 'tracks'],
  eventCount: 1000
})

const expanded = ref({
  channels: false,
  objects: false
})

// Available data from HuggingFace
const channelTypes = ref([])
const pileupTypes = ref([])
const loading = ref(true)

// Fixed object types for all ColliderML datasets (sizes loaded dynamically)
const objectTypes = [
  { id: 'particles', label: 'Particles', description: 'Truth-level particle data' },
  { id: 'tracker_hits', label: 'Tracker Hits', description: 'Silicon detector hits' },
  { id: 'calo_hits', label: 'Calo Hits', description: 'Calorimeter energy deposits' },
  { id: 'tracks', label: 'Tracks', description: 'Reconstructed particle trajectories' }
]

// Size estimates loaded from JSON (GB per 1000 events)
const sizeEstimates = ref({})

async function fetchSizeEstimates() {
  try {
    console.log('[ColliderML] Loading size estimates...')
    const response = await fetch('/ColliderML/size-estimates.json')

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    sizeEstimates.value = await response.json()
    console.log('[ColliderML] Size estimates loaded:', sizeEstimates.value)
  } catch (error) {
    console.error('[ColliderML] Failed to load size estimates:', error)
    // Fallback to default estimates
    sizeEstimates.value = {
      pu0: {
        particles: 0.25,
        tracker_hits: 0.2,
        calo_hits: 0.60,
        tracks: 0.003
      },
      pu200: {
        particles: 8.0,
        tracker_hits: 10.0,
        calo_hits: 14.0,
        tracks: 0.07
      }
    }
  }
}

async function fetchAvailableDatasets() {
  try {
    console.log('[ColliderML] Fetching available configs from HuggingFace...')
    const response = await fetch(
      'https://huggingface.co/api/datasets/CERN/Colliderml-release-1'
    )

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const datasetInfo = await response.json()

    // Parse available configs from the single consolidated dataset
    // Config format: {process}_{pileup}_{object_type}
    const configs = datasetInfo.siblings
      ?.filter(s => s.rfilename.startsWith('data/'))
      ?.map(s => {
        // Extract config from path like: data/ttbar_pu0_particles/train-00000-of-00001.parquet
        const match = s.rfilename.match(/data\/([^/]+)\//)
        return match ? match[1] : null
      })
      .filter((c, i, arr) => c && arr.indexOf(c) === i) // unique configs

    console.log('[ColliderML] Found configs:', configs)

    if (!configs || configs.length === 0) {
      throw new Error('No configs found in dataset')
    }

    // Parse configs to extract processes and pileup levels
    const processes = new Set()
    const pileups = new Set()

    configs.forEach(config => {
      const parts = config.split('_')
      if (parts.length >= 3) {
        // Handle multi-word processes like "dihiggs"
        const objectTypes = ['particles', 'tracker', 'calo', 'tracks']
        let pileupIdx = parts.findIndex(p => p.startsWith('pu'))

        if (pileupIdx > 0) {
          const process = parts.slice(0, pileupIdx).join('_')
          const pileup = parts[pileupIdx]
          processes.add(process)
          pileups.add(pileup)
        }
      }
    })

    console.log('[ColliderML] Found processes:', Array.from(processes))
    console.log('[ColliderML] Found pileups:', Array.from(pileups))

    channelTypes.value = Array.from(processes).map(ch => ({
      id: ch,
      label: ch,
      available: true
    }))

    pileupTypes.value = Array.from(pileups).sort().map(pu => ({
      id: pu,
      label: pu === 'pu0' ? 'No Pileup (pu0)' : `Pileup ${pu.replace('pu', '')}`,
      available: true
    }))

    loading.value = false

  } catch (error) {
    console.error('[ColliderML] Failed to fetch dataset info:', error)
    // Fallback to known configs
    channelTypes.value = [
      { id: 'ttbar', label: 'ttbar', available: true },
      { id: 'ggf', label: 'ggf', available: true },
      { id: 'dihiggs', label: 'dihiggs', available: true }
    ]
    pileupTypes.value = [
      { id: 'pu0', label: 'No Pileup (pu0)', available: true },
      { id: 'pu200', label: 'Pileup 200', available: true }
    ]
    loading.value = false
  }
}

onMounted(async () => {
  await fetchSizeEstimates()
  await fetchAvailableDatasets()
})

// Check if current selection is available
const isDatasetAvailable = computed(() => {
  const channel = selections.value.channels[0]
  const pileup = selections.value.pileup

  const hasChannel = channelTypes.value.some(ch => ch.id === channel)
  const hasPileup = pileupTypes.value.some(pu => pu.id === pileup)

  return hasChannel && hasPileup
})

// Size estimation (MB per 1000 events per object)
const estimatedSizeMB = computed(() => {
  const channelCount = selections.value.channels.length
  if (channelCount === 0 || selections.value.objects.length === 0) return 0

  // Get size estimates for current pileup level
  const currentPileup = selections.value.pileup
  const estimates = sizeEstimates.value[currentPileup] || {}

  const sizePerObject = selections.value.objects.reduce((total, objId) => {
    const size = estimates[objId] || 0
    return total + size
  }, 0)

  // Size in GB per 1000 events
  const sizeGB = sizePerObject * channelCount * (selections.value.eventCount / 1000)
  return sizeGB * 1024 // Convert to MB
})

const estimatedSize = computed(() => {
  const mb = estimatedSizeMB.value
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)}GB`
  } else if (mb < 1) {
    return `${Math.round(mb * 1000)}KB`
  } else {
    return `${Math.round(mb)}MB`
  }
})

// Generate CLI download command
const command = computed(() => {
  const channel = selections.value.channels[0] || 'ttbar'
  const pileup = selections.value.pileup || 'pu0'
  const objects = selections.value.objects
  const events = selections.value.eventCount

  if (objects.length === 0) {
    return '# Select at least one object type'
  }

  const objectsStr = objects.join(',')
  return `colliderml download --channels ${channel} --pileup ${pileup} --objects ${objectsStr} --max-events ${events}`
})

const libraryDocsLink = '/ColliderML/library/overview'
const notebookLink = 'https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/colliderml_loader_exploration.ipynb'

const toggleItem = (category, id) => {
  if (category === 'channels') {
    selections.value.channels = [id]
  } else if (category === 'objects') {
    const items = selections.value[category]
    const index = items.indexOf(id)
    if (index === -1) {
      items.push(id)
    } else {
      items.splice(index, 1)
    }
  }
}

const selectAll = (category) => {
  if (category === 'objects') {
    selections.value.objects = objectTypes.map(obj => obj.id)
  }
}

const deselectAll = (category) => {
  if (category === 'objects') {
    selections.value.objects = []
  }
}

const isSelected = (category, id) => {
  if (category === 'channels') {
    return selections.value.channels.includes(id)
  }
  return selections.value[category] && selections.value[category].includes(id)
}

const toggleExpand = (category) => {
  expanded.value[category] = !expanded.value[category]
}

const isCopied = ref(false)

const copyCommand = async () => {
  try {
    await navigator.clipboard.writeText(command.value)
    isCopied.value = true
    setTimeout(() => {
      isCopied.value = false
    }, 2000)
  } catch (err) {
    console.warn('Failed to copy:', err)
    // Fallback for older browsers
    const el = document.createElement('textarea')
    el.value = command.value
    document.body.appendChild(el)
    el.select()
    document.execCommand('copy')
    document.body.removeChild(el)
    isCopied.value = true
    setTimeout(() => {
      isCopied.value = false
    }, 2000)
  }
}

// Event count slider
const eventCountValues = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

const sliderValue = computed({
  get: () => {
    const index = eventCountValues.findIndex(v => v >= selections.value.eventCount)
    return index === -1 ? eventCountValues.length - 1 : index
  },
  set: (value) => {
    selections.value.eventCount = eventCountValues[value]
  }
})

const formatEventCount = (count) => {
  if (count >= 1000000) return `${count/1000000}M`
  if (count >= 1000) return `${count/1000}k`
  return count.toString()
}
</script>

<template>
  <div class="config-modal">
    <div class="config-panel">
      <h3>Dataset Configuration</h3>

      <div v-if="loading" class="loading-state">
        <div class="spinner"></div>
        <p>Loading available datasets from HuggingFace...</p>
      </div>

      <template v-else>
        <!-- Pileup Level Card -->
        <div class="config-card">
          <div class="card-header">
            <h4>Pileup Level</h4>
          </div>
          <div class="button-grid">
            <button
              v-for="pu in pileupTypes"
              :key="pu.id"
              class="select-button"
              :class="{ selected: selections.pileup === pu.id }"
              @click="selections.pileup = pu.id"
            >
              {{ pu.label }}
            </button>
          </div>
        </div>

        <!-- Channels Card -->
        <div class="config-card">
          <div class="card-header">
            <h4>Physics Process</h4>
            <button
              v-if="channelTypes.length > 3"
              class="expand-button"
              :class="{ expanded: expanded.channels }"
              @click="toggleExpand('channels')"
            >
              {{ expanded.channels ? '↑ Show Less' : '↓ Show More' }}
            </button>
          </div>
          <div class="button-grid">
            <button
              v-for="channel in expanded.channels ? channelTypes : channelTypes.slice(0, 3)"
              :key="channel.id"
              class="select-button"
              :class="{ selected: isSelected('channels', channel.id) }"
              @click="toggleItem('channels', channel.id)"
            >
              {{ channel.label }}
            </button>
          </div>
        </div>

        <!-- Objects Card -->
        <div class="config-card">
          <div class="card-header">
            <h4>Data Objects</h4>
            <button class="expand-button" :class="{ expanded: expanded.objects }" @click="toggleExpand('objects')">
              {{ expanded.objects ? '↑ Show Less' : '↓ Show More' }}
            </button>
          </div>
          <div class="object-grid">
            <button
              v-for="obj in objectTypes"
              :key="obj.id"
              class="object-button"
              :class="{ selected: isSelected('objects', obj.id) }"
              @click="toggleItem('objects', obj.id)"
            >
              <div class="object-name">{{ obj.label }}</div>
              <div v-if="expanded.objects" class="object-desc">{{ obj.description }}</div>
            </button>
          </div>
          <div class="select-all">
            <div class="button-group">
              <button class="select-all-button" @click="deselectAll('objects')">
                Deselect All
              </button>
              <button class="select-all-button" @click="selectAll('objects')">
                Select All
              </button>
            </div>
          </div>
        </div>

        <!-- Event Count Slider Card -->
        <div class="config-card">
          <div class="card-header">
            <h4>Number of Events: {{ selections.eventCount.toLocaleString() }}</h4>
          </div>
          <div class="slider-container">
            <div class="slider-with-labels">
              <span class="slider-label">{{ formatEventCount(eventCountValues[0]) }}</span>
              <div class="slider-track-container">
                <input
                  type="range"
                  v-model="sliderValue"
                  :min="0"
                  :max="eventCountValues.length - 1"
                  step="1"
                  class="range-slider"
                >
                <div class="tick-marks">
                  <div
                    v-for="(value, index) in eventCountValues"
                    :key="index"
                    class="tick"
                    :class="{ active: index <= sliderValue }"
                  >
                    <span class="tick-label">{{ formatEventCount(value) }}</span>
                  </div>
                </div>
              </div>
              <span class="slider-label">{{ formatEventCount(eventCountValues[eventCountValues.length - 1]) }}</span>
            </div>
          </div>
        </div>

        <!-- Estimation Card -->
        <div class="config-card">
          <div class="estimation-section">
            <h4>Estimated Download Size</h4>
            <div class="estimates">
              <div class="estimate-item">
                <span class="label">Approximate Size</span>
                <div class="value-box">{{ estimatedSize }}</div>
              </div>
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- Command Display -->
    <div class="command-section">
      <h4 class="command-label">Download command</h4>
      <div class="command">
        <pre><code>{{ command }}</code></pre>
        <button
          @click="copyCommand"
          class="copy-button"
          :class="{ 'copied': isCopied }"
          :title="isCopied ? 'Copied!' : 'Copy to clipboard'"
        >
          <svg v-if="!isCopied" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
          <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </button>
      </div>
      <div class="library-note">
        Then load the data in Python with <code>load_tables</code>; see the <a :href="libraryDocsLink">library docs</a> or the <a :href="notebookLink" target="_blank" rel="noopener">exploration notebook</a>.
      </div>
    </div>
  </div>
</template>

<style scoped>
.config-modal {
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  box-shadow: var(--modal-shadow);
}

.config-panel {
  background: var(--vp-c-bg);
  border-radius: 8px;
  padding: 20px;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px;
  gap: 16px;
}

.spinner {
  border: 3px solid var(--vp-c-divider);
  border-top: 3px solid var(--vp-c-brand);
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

h3 {
  margin: 0 0 20px 0;
  font-size: 1.2em;
  color: var(--vp-c-text-1);
}

h4 {
  margin: 0 0 12px 0;
  font-size: 1em;
  color: var(--vp-c-text-2);
}

.config-card {
  background: var(--vp-c-bg);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.card-header h4 {
  margin: 0;
}

.expand-button {
  background: none;
  border: none;
  color: var(--vp-c-text-2);
  cursor: pointer;
  font-size: 0.9em;
  padding: 4px 8px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.expand-button:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.button-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px;
  margin-bottom: 8px;
}

.object-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}

.select-button, .object-button {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 0.9em;
  cursor: pointer;
  transition: all 0.2s ease;
  width: 100%;
  text-align: center;
}

.object-button {
  text-align: left;
  padding: 12px;
}

.object-name {
  font-weight: 500;
  margin-bottom: 4px;
}

.object-desc {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  line-height: 1.3;
}

.select-button.selected, .object-button.selected {
  background: var(--vp-c-brand);
  color: white;
  border-color: var(--vp-c-brand);
}

.select-button.selected .object-desc {
  color: rgba(255, 255, 255, 0.8);
}

.select-button:hover:not(.selected), .object-button:hover:not(.selected) {
  border-color: var(--vp-c-brand);
}

.select-all {
  display: flex;
  justify-content: flex-end;
  margin-top: 8px;
}

.button-group {
  display: flex;
  gap: 8px;
}

.select-all-button {
  background: none;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  padding: 4px 12px;
  font-size: 0.9em;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s ease;
}

.select-all-button:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-brand);
}

.estimation-section {
  min-width: 0;
}

.estimates {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.estimate-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.estimate-item .label {
  color: var(--vp-c-text-2);
  white-space: nowrap;
}

.value-box {
  background: var(--vp-c-bg-soft);
  padding: 6px 16px;
  border-radius: 6px;
  font-weight: 500;
  font-size: 1.1em;
  min-width: 80px;
  text-align: center;
  white-space: nowrap;
}

.command-section {
  background: var(--vp-c-bg);
  border-radius: 8px;
  padding: 16px;
  margin-top: 16px;
}

.command-label {
  margin: 0 0 8px 0;
  font-size: 1em;
  color: var(--vp-c-text-2);
}

.command {
  display: flex;
  align-items: stretch;
  background: var(--vp-c-bg-soft);
  padding: 0;
  border-radius: 6px;
  overflow: hidden;
}

.command pre {
  margin: 0;
  padding: 12px;
  flex-grow: 1;
  overflow-x: auto;
}

.command code {
  font-family: monospace;
  font-size: 0.9em;
}

.library-note {
  margin-top: 10px;
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}

.library-note a {
  color: var(--vp-c-brand);
  text-decoration: none;
}

.library-note a:hover {
  text-decoration: underline;
}

.library-note code {
  font-size: 0.9em;
}

.copy-button {
  background: transparent;
  color: var(--vp-c-text-2);
  border: none;
  border-left: 1px solid var(--vp-c-divider);
  padding: 0 12px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 42px;
}

.copy-button:hover {
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-1);
}

.copy-button.copied {
  color: var(--vp-c-green);
  background: var(--vp-c-bg-mute);
}

/* Slider styles */
.slider-container {
  padding: 1rem;
}

.slider-with-labels {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.slider-track-container {
  flex-grow: 1;
  position: relative;
  padding: 1rem 0;
}

.tick-marks {
  position: absolute;
  left: 0;
  right: 0;
  bottom: -8px;
  display: flex;
  justify-content: space-between;
  pointer-events: none;
}

.tick {
  position: relative;
  width: 2px;
  height: 8px;
  background: var(--vp-c-divider);
  transition: background-color 0.2s;
}

.tick.active {
  background: var(--vp-c-brand);
}

.tick-label {
  position: absolute;
  top: 12px;
  left: 50%;
  transform: translateX(-50%) rotate(-45deg);
  font-size: 0.75em;
  color: var(--vp-c-text-2);
  white-space: nowrap;
}

.range-slider {
  -webkit-appearance: none;
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-bg-soft);
  outline: none;
  margin: 0;
}

.range-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand);
  cursor: pointer;
  border: 2px solid var(--vp-c-bg);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: background-color 0.2s, transform 0.1s;
}

.range-slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand);
  cursor: pointer;
  border: 2px solid var(--vp-c-bg);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: background-color 0.2s, transform 0.1s;
}

.range-slider::-webkit-slider-thumb:hover {
  transform: scale(1.1);
}

.range-slider::-moz-range-thumb:hover {
  transform: scale(1.1);
}

.slider-label {
  color: var(--vp-c-text-2);
  font-size: 0.9em;
  white-space: nowrap;
}
</style>
