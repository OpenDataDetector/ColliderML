<script setup>
import { ref, onMounted } from 'vue'

const isExpanded = ref(true)

onMounted(() => {
  if (typeof window === 'undefined') return

  const lastVisit = localStorage.getItem('lastVisit')
  const now = new Date().getTime()
  
  // If no last visit or it was more than a month ago, expand the section
  if (!lastVisit || (now - parseInt(lastVisit)) > 30 * 24 * 60 * 60 * 1000) {
    isExpanded.value = true
  } else {
    isExpanded.value = false
  }
  
  // Update last visit time
  localStorage.setItem('lastVisit', now.toString())
})
</script>

<template>
  <div class="about-section">
    <div 
      class="section-header" 
      @click="isExpanded = !isExpanded"
      :class="{ 'expanded': isExpanded }"
    >
      <h2>About the Data</h2>
        <div class="toggle-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>
      </div>
    
    <div class="section-content" :class="{ 'expanded': isExpanded }">
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
.about-section {
  margin: 2rem 0;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  transition: background-color 0.2s;
}

.section-header:hover {
  background: var(--vp-c-bg-soft);
}

.section-header h2 {
  margin: 0;
  font-size: 1.5rem;
}

.toggle-icon {
  transition: transform 0.3s ease;
}

.section-header.expanded .toggle-icon {
  transform: rotate(180deg);
}

.section-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.5s ease-in-out;
  opacity: 0;
}

.section-content.expanded {
  max-height: 2000px; /* Adjust based on content */
  opacity: 1;
}
</style> 