/**
 * Foundry API Client
 *
 * Communicates with the Foundry backend (training & evaluation) via REST API.
 *
 * The Foundry FastAPI router (foundry/src/server.py) wraps list results
 * ({ "runs": [...] }, { "versions": [...] }, { "merges": [...] }) and uses
 * domain `to_dict()` shapes that differ from the UI's normalized shapes:
 *   - POST /training/configure requires `base_model_family` + `output_dir`
 *   - POST /training/start takes `run_id` as a QUERY parameter (no body)
 *   - The training registry list carries only run_id/adapter_path/discipline_id/
 *     created_at (no status/name/config), so completed-run metadata is synthesized
 *   - Evaluation reports key competency_scores by id (a dict) and use
 *     overall_accuracy/total_cases; there is NO /evaluation/history endpoint
 *
 * Each method translates the real backend payload into the interfaces declared
 * in this file, which are the contract the UI components rely on.
 */

const API_BASE = "http://127.0.0.1:8420/api/foundry";

// ============================================================
// Type definitions (UI-normalized shapes)
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
  config: { base_model: string } & Partial<TrainingConfig>;
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
// Raw backend payload shapes (foundry/src/*.py to_dict)
// ============================================================

interface RawTrainingRun {
  run_id: string;
  config_path: string;
  result_path: string;
  adapter_path: string | null;
  discipline_id: string | null;
  created_at: string;
}

interface RawCompetencyScore {
  competency_id: string;
  competency_name: string;
  total_cases: number;
  correct: number;
  partially_correct: number;
  incorrect: number;
  no_response: number;
  rating: string;
  summary: string;
}

interface RawEvaluationReport {
  run_id: string;
  model_name: string;
  discipline_id: string;
  status: string;
  competency_scores: Record<string, RawCompetencyScore>;
  total_cases: number;
  overall_correct: number;
  overall_accuracy: number;
  overall_rating: string;
}

interface RawDiagnosticIssue {
  category: string;
  severity: string;
  title: string;
  description: string;
  suggestion: string;
  evidence: string[];
  detected_at_epoch: number | null;
}

interface RawDiagnosticReport {
  run_id?: string;
  issues: RawDiagnosticIssue[];
  trends: Record<string, { is_decreasing: boolean; is_plateaued: boolean }>;
  overall_health: string;
}

interface RawVersionEntry {
  version_id: string;
  model_name: string;
  discipline_id: string;
  training_run_id: string | null;
  evaluation_run_id: string;
  adapter_path: string | null;
  change_type: string;
  change_description: string;
  created_at: string;
  is_active: boolean;
}

interface RawMergeResult {
  merge_id: string;
  method: string;
  status: string;
  adapters: Array<{ adapter_path: string; discipline_name?: string }>;
  weights_used: number[];
  merged_adapter_path: string | null;
  started_at: string;
  completed_at: string | null;
}

// ============================================================
// Transforms
// ============================================================

const RATING_MAP: Record<string, CompetencyScore["rating"]> = {
  strong: "strong",
  adequate: "adequate",
  needs_improvement: "weak",
  weak: "weak",
  untested: "untested",
};

const SEVERITY_MAP: Record<string, DiagnosticIssue["severity"]> = {
  high: "high",
  critical: "high",
  medium: "medium",
  warning: "medium",
  low: "low",
  info: "low",
};

/** Derive the LoRA base-model family enum value the backend expects. */
function baseModelFamily(baseModel: string): string {
  const m = baseModel.toLowerCase();
  if (m.includes("mistral")) return "mistral";
  if (m.includes("phi")) return "phi";
  if (m.includes("qwen")) return "qwen";
  return "llama";
}

function adapterName(raw: RawTrainingRun): string {
  if (raw.adapter_path) {
    const parts = raw.adapter_path.replace(/\\/g, "/").split("/");
    const last = parts[parts.length - 1];
    if (last) return last;
  }
  return raw.run_id;
}

function toTrainingRun(raw: RawTrainingRun): TrainingRun {
  // The registry only stores completed runs; status/progress are synthesized.
  return {
    id: raw.run_id,
    name: adapterName(raw),
    status: "completed",
    progress: 100,
    config: { base_model: "", adapter_name: adapterName(raw) },
    metrics: {},
    created_at: raw.created_at,
    started_at: null,
    completed_at: raw.created_at,
    error: null,
  };
}

function toEvaluationResult(raw: RawEvaluationReport): EvaluationResult {
  const competency_scores: CompetencyScore[] = Object.values(
    raw.competency_scores ?? {},
  ).map((cs) => ({
    competency_name: cs.competency_name,
    correct: cs.correct,
    total: cs.total_cases,
    score: cs.total_cases > 0 ? cs.correct / cs.total_cases : 0,
    rating: RATING_MAP[cs.rating] ?? "untested",
  }));
  return {
    id: raw.run_id,
    training_run_id: raw.run_id,
    model_name: raw.model_name,
    overall_score: raw.overall_accuracy,
    overall_correct: raw.overall_correct,
    overall_total: raw.total_cases,
    competency_scores,
    created_at: new Date().toISOString(),
  };
}

function toDiagnosticReport(raw: RawDiagnosticReport): DiagnosticReport {
  const issues: DiagnosticIssue[] = (raw.issues ?? []).map((i) => ({
    type: i.title || i.category,
    severity: SEVERITY_MAP[i.severity] ?? "low",
    description: i.description,
    recommendation: i.suggestion,
  }));
  // Derive convergence/overfit indicators from trends + issue categories.
  const lossTrend = raw.trends?.train_loss ?? raw.trends?.val_loss;
  let convergence_status: DiagnosticReport["convergence_status"] = "unknown";
  if (lossTrend) {
    if (lossTrend.is_plateaued) convergence_status = "plateau";
    else if (lossTrend.is_decreasing) convergence_status = "converging";
    else convergence_status = "diverging";
  }
  const hasOverfit = (raw.issues ?? []).some((i) =>
    i.category.toLowerCase().includes("overfit"),
  );
  return {
    run_id: raw.run_id ?? "",
    issues,
    convergence_status,
    overfit_risk: hasOverfit ? "high" : "none",
    analyzed_at: new Date().toISOString(),
  };
}

function toModelVersion(raw: RawVersionEntry): ModelVersion {
  return {
    id: raw.version_id,
    name: raw.model_name,
    adapter_path: raw.adapter_path ?? "",
    training_run_id: raw.training_run_id ?? "",
    evaluation_score: null,
    created_at: raw.created_at,
  };
}

function toMergeResult(raw: RawMergeResult): MergeResult {
  return {
    id: raw.merge_id,
    output_path: raw.merged_adapter_path ?? "",
    method: raw.method,
    adapters_merged: (raw.adapters ?? []).map((a) => a.adapter_path),
    created_at: raw.completed_at ?? raw.started_at,
  };
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
    foundryFetch<{ run_id: string; config: Record<string, unknown> }>(
      "/training/configure",
      {
        method: "POST",
        body: JSON.stringify({
          base_model: config.base_model,
          base_model_family: baseModelFamily(config.base_model),
          curriculum_path: config.curriculum_path,
          output_dir: `outputs/${config.adapter_name}`,
          epochs: config.epochs ?? 3,
          batch_size: config.batch_size ?? 4,
          learning_rate: config.learning_rate ?? 2e-4,
          lora: config.lora_rank
            ? { rank: config.lora_rank, alpha: config.lora_alpha ?? 32 }
            : undefined,
          auto_configure: config.auto_configure ?? false,
        }),
      },
    ),

  // /training/start takes run_id as a query parameter, not a body.
  start: (runId: string) =>
    foundryFetch<{
      run_id: string;
      registry_run_id: string;
      status: string;
      adapter_path: string | null;
    }>(`/training/start?run_id=${encodeURIComponent(runId)}`, {
      method: "POST",
    }),

  getStatus: (runId: string) =>
    foundryFetch<{ run_id: string; status: string; result?: unknown }>(
      `/training/${runId}/status`,
    ),

  cancel: async (runId: string): Promise<TrainingRun> => {
    const raw = await foundryFetch<{ run_id: string; status: string }>(
      `/training/${runId}/cancel`,
      { method: "POST" },
    );
    return {
      id: raw.run_id,
      name: raw.run_id,
      status: "cancelled",
      progress: 0,
      config: { base_model: "" },
      metrics: {},
      created_at: new Date().toISOString(),
      started_at: null,
      completed_at: null,
      error: null,
    };
  },

  listRuns: async (): Promise<TrainingRun[]> => {
    const data = await foundryFetch<{ runs: RawTrainingRun[] }>(
      "/training/runs",
    );
    return (data.runs ?? []).map(toTrainingRun);
  },
};

export const evaluationAPI = {
  /**
   * Backend requires a Forge-exported test set (JSONL) plus a competency-name
   * map, model name and discipline id. The UI only has a training run id today,
   * so this sends a best-effort body. Genuine evaluation needs a test-set
   * picker (see report notes); without one the backend returns 404/400.
   */
  run: async (data: {
    training_run_id: string;
    test_set_path?: string;
    model_name?: string;
    discipline_id?: string;
  }): Promise<EvaluationResult> => {
    const raw = await foundryFetch<RawEvaluationReport>("/evaluation/run", {
      method: "POST",
      body: JSON.stringify({
        test_set_path: data.test_set_path ?? "test_set.jsonl",
        competency_names: {},
        model_name: data.model_name ?? data.training_run_id,
        discipline_id: data.discipline_id ?? "default",
      }),
    });
    return toEvaluationResult(raw);
  },

  get: async (evalId: string): Promise<EvaluationResult> => {
    const raw = await foundryFetch<RawEvaluationReport>(
      `/evaluation/${evalId}`,
    );
    return toEvaluationResult(raw);
  },

  compare: (evalIdA: string, evalIdB: string) =>
    foundryFetch<Record<string, unknown>>(
      `/evaluation/compare?eval_id_a=${encodeURIComponent(
        evalIdA,
      )}&eval_id_b=${encodeURIComponent(evalIdB)}`,
    ),
};

export const diagnosticsAPI = {
  /**
   * POST /diagnostics/analyze/{run_id} requires training metric snapshots which
   * the UI does not currently capture; an empty list is sent and the backend
   * will reject it until a metrics source is wired (see report notes).
   */
  analyze: async (
    runId: string,
    metrics: Array<Record<string, unknown>> = [],
  ): Promise<DiagnosticReport> => {
    const raw = await foundryFetch<RawDiagnosticReport>(
      `/diagnostics/analyze/${runId}`,
      { method: "POST", body: JSON.stringify({ metrics }) },
    );
    return toDiagnosticReport(raw);
  },

  get: async (runId: string): Promise<DiagnosticReport> => {
    const raw = await foundryFetch<RawDiagnosticReport>(
      `/diagnostics/${runId}`,
    );
    return toDiagnosticReport(raw);
  },
};

export const regressionAPI = {
  check: (data: { baseline_eval_id: string; current_eval_id: string }) =>
    foundryFetch<Record<string, unknown>>("/regression/check", {
      method: "POST",
      body: JSON.stringify({ ...data, change_type: "retrain" }),
    }),

  listVersions: async (): Promise<ModelVersion[]> => {
    const data = await foundryFetch<{ versions: RawVersionEntry[] }>(
      "/regression/versions",
    );
    return (data.versions ?? []).map(toModelVersion);
  },

  register: (data: {
    model_name: string;
    discipline_id: string;
    evaluation_run_id: string;
    adapter_path?: string;
  }) =>
    foundryFetch<RawVersionEntry>("/regression/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),
};

export const mergingAPI = {
  /**
   * Backend expects full AdapterInfo dicts; the UI only has version ids/paths,
   * so each selected adapter is sent as a minimal { adapter_path } dict. A real
   * merge needs full adapter metadata + on-disk adapters (see report notes).
   */
  merge: async (request: MergeRequest): Promise<MergeResult> => {
    const raw = await foundryFetch<RawMergeResult>("/merging/merge", {
      method: "POST",
      body: JSON.stringify({
        adapters: request.adapters.map((a) => ({ adapter_path: a })),
        method: request.method,
        weights: request.weights,
        output_dir: request.output_name,
      }),
    });
    return toMergeResult(raw);
  },

  listRegistry: async (): Promise<MergeResult[]> => {
    const data = await foundryFetch<{ merges: RawMergeResult[] }>(
      "/merging/registry",
    );
    return (data.merges ?? []).map(toMergeResult);
  },

  getMethods: () =>
    foundryFetch<{ methods: Array<{ name: string; description: string }> }>(
      "/merging/methods",
    ),
};
