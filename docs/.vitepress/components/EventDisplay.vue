<script setup lang="ts">
import { withBase } from 'vitepress'
// Self-contained hep-viz Three.js scene served from /public/event-display/.
// withBase() keeps the src correct under any deploy base (/ColliderML/, staging, …).
const src = withBase('/event-display/index.html')
</script>

<template>
  <div class="evd">
    <iframe :src="src" title="ColliderML interactive event display" loading="lazy"
      allow="fullscreen" referrerpolicy="no-referrer"></iframe>
    <a class="evd-pop" :href="src" target="_blank" rel="noopener">Open full-screen ↗</a>
  </div>
</template>

<style scoped>
.evd { position: relative; border: 1px solid var(--vp-c-divider); border-radius: 12px; overflow: hidden; margin: 18px 0; }
.evd iframe { width: 100%; height: 640px; border: 0; display: block; background: #0b0e14; }
.evd-pop { position: absolute; top: 10px; right: 12px; font-size: 12px; padding: 4px 10px; border-radius: 6px;
  background: rgba(0,0,0,0.55); color: #fff; text-decoration: none; backdrop-filter: blur(4px); }
.evd-pop:hover { background: rgba(0,0,0,0.75); }
</style>
