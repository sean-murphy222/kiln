/**
 * Forge-specific state — disciplines, competencies, examples, discovery.
 */

import { create } from 'zustand';

export interface ForgeDiscipline {
  id: string;
  name: string;
  status: 'draft' | 'active' | 'archived';
  competency_count: number;
  example_count: number;
  coverage_pct: number;
}

export interface ForgeCompetency {
  id: string;
  discipline_id: string;
  name: string;
  description: string;
  example_count: number;
  target_count: number;
}

export interface ForgeExample {
  id: string;
  competency_id: string;
  question: string;
  answer: string;
  status: 'draft' | 'approved' | 'rejected' | 'needs_revision';
  contributor_id: string;
}

interface ForgeState {
  disciplines: ForgeDiscipline[];
  selectedDisciplineId: string | null;
  competencies: ForgeCompetency[];
  examples: ForgeExample[];
  discoverySessionId: string | null;
  isLoading: boolean;
  error: string | null;

  setDisciplines: (disciplines: ForgeDiscipline[]) => void;
  selectDiscipline: (id: string | null) => void;
  setCompetencies: (competencies: ForgeCompetency[]) => void;
  setExamples: (examples: ForgeExample[]) => void;
  setDiscoverySession: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useForgeStore = create<ForgeState>((set) => ({
  disciplines: [],
  selectedDisciplineId: null,
  competencies: [],
  examples: [],
  discoverySessionId: null,
  isLoading: false,
  error: null,

  setDisciplines: (disciplines) => set({ disciplines }),
  selectDiscipline: (selectedDisciplineId) => set({ selectedDisciplineId }),
  setCompetencies: (competencies) => set({ competencies }),
  setExamples: (examples) => set({ examples }),
  setDiscoverySession: (discoverySessionId) => set({ discoverySessionId }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
