import { defineConfig } from 'vitepress'

// Base path is branch-aware: `main` serves the canonical site at
// `/ColliderML/`, `staging` (and PR previews) serve at `/ColliderML/staging/`.
// The deploy workflow sets VITEPRESS_BASE accordingly; leave the default
// for local `vitepress dev`.
const base = process.env.VITEPRESS_BASE ?? '/ColliderML/'

const config = defineConfig({
  title: 'ColliderML',
  description: 'A modern machine learning library for high-energy physics data analysis',
  lang: 'en-US',
  lastUpdated: true,
  base,

  // Ignore dead links for now (pages don't exist yet)
  ignoreDeadLinks: true,

  themeConfig: {
    // logo: { src: '/logo.png', height: 32 },
    siteTitle: 'ColliderML',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/introduction' },
      { text: 'Library', link: '/library/overview' },
      { text: 'Spaces', link: '/spaces/overview' },
      { text: 'Appendices', link: '/appendices' }
    ],
    sidebar: {
      '/library/': [
        {
          text: 'Library',
          items: [
            { text: 'Overview', link: '/library/overview' },
            { text: 'CLI', link: '/library/cli' },
            { text: 'Loading', link: '/library/loading' },
            { text: 'Exploding', link: '/library/exploding' },
            { text: 'Physics utilities', link: '/library/physics' },
            { text: 'Simulate', link: '/library/simulate' },
            { text: 'Remote (SaaS)', link: '/library/remote' },
            { text: 'Tasks (benchmarks)', link: '/library/tasks' },
            { text: 'Benchmarks (timing)', link: '/library/benchmarks' },
          ],
        },
      ],
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Introduction', link: '/guide/introduction' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quickstart', link: '/guide/quickstart' },
            { text: 'Local simulation', link: '/guide/simulation' },
            { text: 'Remote simulation', link: '/guide/remote-simulation' },
            { text: 'Benchmark tasks', link: '/guide/tasks' },
          ],
        },
      ],
      '/spaces/': [
        {
          text: 'HuggingFace Spaces',
          items: [
            { text: 'Overview', link: '/spaces/overview' },
            { text: 'Event Display', link: '/spaces/event-display' },
            { text: 'Simulation Form', link: '/spaces/simulation-form' },
            { text: 'Leaderboard', link: '/spaces/leaderboard' },
            { text: 'Model Zoo', link: '/spaces/model-zoo' },
          ],
        },
      ],
    },
    search: {
      provider: 'local',
      options: {
        _render(src: string, env: any) {
          // Exclude /internal/ pages from search index
          if (env.relativePath?.startsWith('internal/')) return ''
          return src
        }
      }
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present ColliderML Contributors'
    }
  },
  
  vite: {
    server: {
      proxy: {
        '/nersc': {
          target: 'https://portal.nersc.gov/cfs/m4958/ColliderML',
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/nersc/, '')
        }
      }
    },
    resolve: {
      alias: {
        '@components': './components'
      }
    },
    plugins: []
  },
  
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag: string) => tag.includes('-')
      }
    }
  }
})

if (config.vite) {
  config.vite.optimizeDeps = {
    include: ['vue'],
    exclude: ['vitepress']
  }
}

export default config 