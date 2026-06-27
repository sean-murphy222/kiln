/**
 * Mappers from Foundry API client types to Foundry store types.
 *
 * The store's TrainingRun flattens `config.base_model` into a top-level
 * `base_model` field; this helper performs that translation. Evaluation,
 * diagnostic, version and merge shapes match the store directly.
 */

import type { TrainingRun as ApiTrainingRun } from "@/api/foundry";
import type { TrainingRun } from "@/store/useFoundryStore";

export function mapTrainingRun(run: ApiTrainingRun): TrainingRun {
  return {
    id: run.id,
    name: run.name,
    status: run.status,
    progress: run.progress,
    base_model: run.config?.base_model ?? "",
    metrics: run.metrics ?? {},
    created_at: run.created_at,
    completed_at: run.completed_at,
    error: run.error,
  };
}
