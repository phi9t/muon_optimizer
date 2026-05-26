export interface TrajectoryPoint {
  step: number
  x: number
  y: number
  loss: number
}

export interface LandscapeGrid {
  x: number[]
  y: number[]
  values: number[][]
  z_min: number
  z_max: number
}

export interface LandscapePayload {
  id: string
  name: string
  description: string
  minimum: { x: number | null; y: number | null; loss: number | null }
  initial_point: { x: number; y: number }
  learning_rates: Record<string, number>
  steps: number
  bounds: { x: [number, number]; y: [number, number] }
  grid: LandscapeGrid
  trajectories: Record<string, TrajectoryPoint[]>
}

export interface LandscapeIndexEntry {
  id: string
  name: string
  description: string
  steps: number
}

export interface LandscapeIndex {
  profile: string
  problems: LandscapeIndexEntry[]
}
