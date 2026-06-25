// Single source of truth for the ColliderML backend origin.
//
// Dev (`vitepress dev`): VITE_COLLIDERML_BACKEND is unset → apiBase = '' → calls
//   stay relative ('/v1/...') and go through the Vite dev-server proxy (config.ts).
// Prod (static GitHub Pages build): the deploy workflow injects
//   VITE_COLLIDERML_BACKEND at build time → the bundle ships absolute
//   https://colliderml-backend.onrender.com/v1/... URLs. This is required because
//   the static site has no proxy. Mirrors DataConfig.vue, which already calls an
//   absolute (huggingface.co) URL from the browser; backend CORS allows the
//   Pages origin.
export const apiBase: string = import.meta.env.VITE_COLLIDERML_BACKEND ?? ''

export function apiUrl(path: string): string {
  return apiBase.replace(/\/$/, '') + path
}
