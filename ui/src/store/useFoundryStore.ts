/**
 * Foundry-specific state — training runs, evaluations, diagnostics.
 */

import { create } from 'zustand';

export interface TrainingRun {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  base_model: string;
  created_at: string;
  completed_at: string | null;
}

export interface EvaluationResult {
  id: string;
  training_run_id: string;
  overall_score: number;
  competency_scores: Record<string, { correct: number; total: number }>;
  created_at: string;
}

interface FoundryState {
  trainingRuns: TrainingRun[];
  selectedRunId: string | null;
  evaluations: EvaluationResult[];
  activeTab: 'training' | 'evaluation' | 'diagnostics' | 'versions' | 'merging';
  isLoading: boolean;
  error: string | null;

  setTrainingRuns: (runs: TrainingRun[]) => void;
  selectRun: (id: string | null) => void;
  setEvaluations: (evals: EvaluationResult[]) => void;
  setActiveTab: (tab: FoundryState['activeTab']) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useFoundryStore = create<FoundryState>((set) => ({
  trainingRuns: [],
  selectedRunId: null,
  evaluations: [],
  activeTab: 'training',
  isLoading: false,
  error: null,

  setTrainingRuns: (trainingRuns) => set({ trainingRuns }),
  selectRun: (selectedRunId) => set({ selectedRunId }),
  setEvaluations: (evaluations) => set({ evaluations }),
  setActiveTab: (activeTab) => set({ activeTab }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
