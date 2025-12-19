import { defineConfig } from 'vitepress'

const config = defineConfig({
  title: 'ColliderML',
  description: 'A modern machine learning library for high-energy physics data analysis',
  lang: 'en-US',
  lastUpdated: true,
  // Set base for GitHub Pages project site if repo is 'colliderml'.
  // If using a custom domain or org/user site, adjust to '/'
  base: '/ColliderML/',
  
  // Ignore dead links for now (pages don't exist yet)
  ignoreDeadLinks: true,
  
  themeConfig: {
    // logo: { src: '/logo.png', height: 32 },
    siteTitle: 'ColliderML',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Library', link: '/library/overview' },
      { text: 'Appendices', link: '/appendices' }
    ],
    sidebar: {
      '/library/': [
        {
          text: 'Library',
          items: [
            { text: 'Overview', link: '/library/overview' },
            { text: 'CLI (download)', link: '/library/cli' },
            { text: 'Loading', link: '/library/loading' },
            { text: 'Exploding', link: '/library/exploding' },
            { text: 'Physics utilities', link: '/library/physics' },
            { text: 'Benchmarks', link: '/library/benchmarks' },
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
          ],
        },
      ],
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