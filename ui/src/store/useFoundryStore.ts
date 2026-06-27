/**
 * Foundry-specific state — training runs, evaluations, diagnostics, versions, merging.
 */

import { create } from 'zustand';

export type FoundryTab = 'training' | 'evaluation' | 'diagnostics' | 'versions' | 'merging';

export interface TrainingRun {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  base_model: string;
  metrics: Record<string, number>;
  created_at: string;
  completed_at: string | null;
  error: string | null;
}

export interface CompetencyScore {
  competency_name: string;
  correct: number;
  total: number;
  score: number;
  rating: 'strong' | 'adequate' | 'weak' | 'untested';
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
  severity: 'high' | 'medium' | 'low';
  description: string;
  recommendation: string;
}

export interface DiagnosticReport {
  run_id: string;
  issues: DiagnosticIssue[];
  convergence_status: 'converging' | 'diverging' | 'plateau' | 'unknown';
  overfit_risk: 'high' | 'medium' | 'low' | 'none';
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

export interface MergeResult {
  id: string;
  output_path: string;
  method: string;
  adapters_merged: string[];
  created_at: string;
}

interface FoundryState {
  activeTab: FoundryTab;
  setActiveTab: (tab: FoundryTab) => void;

  // Training
  trainingRuns: TrainingRun[];
  selectedRunId: string | null;
  setTrainingRuns: (runs: TrainingRun[]) => void;
  addTrainingRun: (run: TrainingRun) => void;
  updateTrainingRun: (id: string, updates: Partial<TrainingRun>) => void;
  selectRun: (id: string | null) => void;

  // Evaluation
  evaluations: EvaluationResult[];
  selectedEvalId: string | null;
  setEvaluations: (evals: EvaluationResult[]) => void;
  addEvaluation: (eval_: EvaluationResult) => void;
  selectEvaluation: (id: string | null) => void;

  // Diagnostics
  diagnosticReport: DiagnosticReport | null;
  setDiagnosticReport: (report: DiagnosticReport | null) => void;

  // Versions
  modelVersions: ModelVersion[];
  setModelVersions: (versions: ModelVersion[]) => void;

  // Merging
  mergeResults: MergeResult[];
  setMergeResults: (results: MergeResult[]) => void;
  addMergeResult: (result: MergeResult) => void;

  // Loading
  isLoading: boolean;
  error: string | null;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useFoundryStore = create<FoundryState>((set) => ({
  activeTab: 'training',
  setActiveTab: (activeTab) => set({ activeTab }),

  trainingRuns: [],
  selectedRunId: null,
  setTrainingRuns: (trainingRuns) => set({ trainingRuns }),
  addTrainingRun: (run) => set((s) => ({ trainingRuns: [run, ...s.trainingRuns] })),
  updateTrainingRun: (id, updates) =>
    set((s) => ({
      trainingRuns: s.trainingRuns.map((r) => (r.id === id ? { ...r, ...updates } : r)),
    })),
  selectRun: (selectedRunId) => set({ selectedRunId }),

  evaluations: [],
  selectedEvalId: null,
  setEvaluations: (evaluations) => set({ evaluations }),
  addEvaluation: (eval_) => set((s) => ({ evaluations: [eval_, ...s.evaluations] })),
  selectEvaluation: (selectedEvalId) => set({ selectedEvalId }),

  diagnosticReport: null,
  setDiagnosticReport: (diagnosticReport) => set({ diagnosticReport }),

  modelVersions: [],
  setModelVersions: (modelVersions) => set({ modelVersions }),

  mergeResults: [],
  setMergeResults: (mergeResults) => set({ mergeResults }),
  addMergeResult: (result) => set((s) => ({ mergeResults: [result, ...s.mergeResults] })),

  isLoading: false,
  error: null,
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
