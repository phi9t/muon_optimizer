export const REPO_HOME = 'https://github.com/phi9t/muon_optimizer'
export const PAPER_URL = 'https://kellerjordan.github.io/posts/muon/'

export function dataUrl(path: string): string {
  const clean = path.replace(/^\//, '')
  return `${import.meta.env.BASE_URL}${clean}`
}

export function logoMarkUrl(): string {
  return dataUrl('logo-mark.svg')
}
