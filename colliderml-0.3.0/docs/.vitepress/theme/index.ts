import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import DataConfig from '../components/DataConfig.vue'
import AboutData from '../components/AboutData.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('DataConfig', DataConfig)
    app.component('AboutData', AboutData)
  }
} as Theme 