export interface OptimizerMetrics {
  final_test_acc: number
  best_test_acc: number
  final_train_loss: number
  convergence_epoch: number
  avg_epoch_time: number
  total_time: number
}

export interface OptimizerSeries {
  name: string
  train_losses: number[]
  train_accuracies: number[]
  test_accuracies: number[]
  epoch_times: number[]
  metrics: OptimizerMetrics
}

export interface MnistPayload {
  profile: string
  epochs: number
  batch_size: number
  dataset: string
  optimizers: OptimizerSeries[]
}

export interface DefinitionsPayload {
  metrics: Record<string, { plain: string }>
  primer: Record<string, string>
  optimizers: Record<string, { plain: string }>
}
