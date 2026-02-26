/**
 * Foundry API Client
 *
 * Communicates with the Foundry backend (training & evaluation) via REST API.
 */

const API_BASE = "http://127.0.0.1:8420/api/foundry";

// ============================================================
// Type definitions
// ============================================================

export interface TrainingConfig {
  base_model: string;
  curriculum_path: string;
  adapter_name: string;
  lora_rank?: number;
  lora_alpha?: number;
  learning_rate?: number;
  epochs?: number;
  batch_size?: number;
  auto_configure?: boolean;
}

export interface TrainingRun {
  id: string;
  name: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  config: TrainingConfig;
  metrics: Record<string, number>;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface CompetencyScore {
  competency_name: string;
  correct: number;
  total: number;
  score: number;
  rating: "strong" | "adequate" | "weak" | "untested";
}

export interface EvaluationResult {
  id: string;
  training_run_id: string;
  model_name: string;
  overall_score: number;
  overall_correct: number;
  overall_total: number;
  competency_scores: CompetencyScore[];
  created_at: string;
}

export interface EvaluationComparison {
  baseline: EvaluationResult;
  candidate: EvaluationResult;
  regressions: Array<{
    competency: string;
    baseline_score: number;
    candidate_score: number;
    delta: number;
  }>;
  improvements: Array<{
    competency: string;
    baseline_score: number;
    candidate_score: number;
    delta: number;
  }>;
  verdict: "pass" | "fail" | "mixed";
}

export interface DiagnosticIssue {
  type: string;
  severity: "high" | "medium" | "low";
  description: string;
  recommendation: string;
}

export interface DiagnosticReport {
  run_id: string;
  issues: DiagnosticIssue[];
  convergence_status: "converging" | "diverging" | "plateau" | "unknown";
  overfit_risk: "high" | "medium" | "low" | "none";
  analyzed_at: string;
}

export interface ModelVersion {
  id: string;
  name: string;
  adapter_path: string;
  training_run_id: string;
  evaluation_score: number | null;
  created_at: string;
}

export interface MergeRequest {
  adapters: string[];
  method: "linear" | "ties";
  output_name: string;
  weights?: number[];
}

export interface MergeResult {
  id: string;
  output_path: string;
  method: string;
  adapters_merged: string[];
  created_at: string;
}

// ============================================================
// API Error
// ============================================================

export class FoundryAPIError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

// ============================================================
// Generic fetch wrapper
// ============================================================

async function foundryFetch<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Unknown error" }));
    throw new FoundryAPIError(
      response.status,
      error.detail || response.statusText,
    );
  }

  return response.json();
}

// ============================================================
// Endpoint groups
// ============================================================

export const foundryHealthAPI = {
  check: () => foundryFetch<{ status: string }>("/health"),
};

export const trainingAPI = {
  configure: (config: TrainingConfig) =>
    foundryFetch<{ config: TrainingConfig; estimated_time: string }>(
      "/training/configure",
      {
        method: "POST",
        body: JSON.stringify(config),
      },
    ),

  start: (runId: string) =>
    foundryFetch<TrainingRun>(`/training/start`, {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    }),

  getStatus: (runId: string) =>
    foundryFetch<TrainingRun>(`/training/${runId}/status`),

  cancel: (runId: string) =>
    foundryFetch<TrainingRun>(`/training/${runId}/cancel`, { method: "POST" }),

  listRuns: () => foundryFetch<TrainingRun[]>("/training/runs"),
};

export const evaluationAPI = {
  run: (data: { training_run_id: string; test_set_path?: string }) =>
    foundryFetch<EvaluationResult>("/evaluation/run", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  get: (evalId: string) =>
    foundryFetch<EvaluationResult>(`/evaluation/${evalId}`),

  compare: (baselineId: string, candidateId: string) =>
    foundryFetch<EvaluationComparison>(
      `/evaluation/compare?baseline=${baselineId}&candidate=${candidateId}`,
    ),

  list: () => foundryFetch<EvaluationResult[]>("/evaluation/history"),
};

export const diagnosticsAPI = {
  analyze: (runId: string) =>
    foundryFetch<DiagnosticReport>(`/diagnostics/analyze/${runId}`, {
      method: "POST",
    }),

  get: (runId: string) =>
    foundryFetch<DiagnosticReport>(`/diagnostics/${runId}`),
};

export const regressionAPI = {
  check: (data: { baseline_version: string; candidate_version: string }) =>
    foundryFetch<EvaluationComparison>("/regression/check", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  listVersions: () => foundryFetch<ModelVersion[]>("/regression/versions"),

  register: (data: {
    name: string;
    adapter_path: string;
    training_run_id: string;
  }) =>
    foundryFetch<ModelVersion>("/regression/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),
};

export const mergingAPI = {
  merge: (request: MergeRequest) =>
    foundryFetch<MergeResult>("/merging/merge", {
      method: "POST",
      body: JSON.stringify(request),
    }),

  listRegistry: () => foundryFetch<MergeResult[]>("/merging/registry"),

  getMethods: () => foundryFetch<{ methods: string[] }>("/merging/methods"),
};
