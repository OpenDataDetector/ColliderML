import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import DataConfig from '../components/DataConfig.vue'
import AboutData from '../components/AboutData.vue'
import WorkflowBuilder from '../components/WorkflowBuilder.vue'
import Copilot from '../components/Copilot.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  // Inject the copilot on every page via the global layout-bottom slot.
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(Copilot),
    })
  },
  enhanceApp({ app }) {
    app.component('DataConfig', DataConfig)
    app.component('AboutData', AboutData)
    app.component('WorkflowBuilder', WorkflowBuilder)
    app.component('Copilot', Copilot)
  },
} as Theme
