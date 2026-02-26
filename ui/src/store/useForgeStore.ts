/**
 * Forge-specific state — disciplines, competencies, examples, discovery.
 */

import { create } from "zustand";

export type ForgeView =
  | "discovery"
  | "competencies"
  | "examples"
  | "consistency";

export interface ForgeDiscipline {
  id: string;
  name: string;
  description: string;
  status: "draft" | "active" | "archived";
  competency_count: number;
  example_count: number;
  coverage_pct: number;
}

export interface ForgeCompetency {
  id: string;
  discipline_id: string;
  name: string;
  description: string;
  level: "foundational" | "intermediate" | "advanced" | "expert";
  parent_id: string | null;
  example_count: number;
  target_count: number;
  children?: ForgeCompetency[];
}

export interface ForgeExample {
  id: string;
  competency_id: string;
  competency_name: string;
  question: string;
  answer: string;
  context: string | null;
  status: "draft" | "approved" | "rejected" | "needs_revision";
  contributor_id: string;
  contributor_name: string;
  created_at: string;
}

export interface DiscoveryQuestion {
  id: string;
  text: string;
  phase: string;
}

export interface DiscoveryAnswer {
  question_id: string;
  answer: string;
}

export interface ConsistencyIssue {
  id: string;
  type: string;
  severity: "high" | "medium" | "low";
  description: string;
  affected_example_ids: string[];
  suggested_fix: string | null;
}

interface ForgeState {
  // View
  activeView: ForgeView;
  setActiveView: (view: ForgeView) => void;

  // Disciplines
  disciplines: ForgeDiscipline[];
  selectedDisciplineId: string | null;
  setDisciplines: (disciplines: ForgeDiscipline[]) => void;
  selectDiscipline: (id: string | null) => void;
  addDiscipline: (discipline: ForgeDiscipline) => void;

  // Competencies
  competencies: ForgeCompetency[];
  setCompetencies: (competencies: ForgeCompetency[]) => void;
  addCompetency: (competency: ForgeCompetency) => void;
  updateCompetency: (id: string, updates: Partial<ForgeCompetency>) => void;
  removeCompetency: (id: string) => void;

  // Examples
  examples: ForgeExample[];
  setExamples: (examples: ForgeExample[]) => void;
  updateExample: (id: string, updates: Partial<ForgeExample>) => void;

  // Discovery
  discoverySessionId: string | null;
  discoveryQuestions: DiscoveryQuestion[];
  discoveryAnswers: DiscoveryAnswer[];
  discoveryCurrentIndex: number;
  setDiscoverySession: (id: string | null) => void;
  setDiscoveryQuestions: (questions: DiscoveryQuestion[]) => void;
  setDiscoveryAnswer: (questionId: string, answer: string) => void;
  setDiscoveryIndex: (index: number) => void;
  resetDiscovery: () => void;

  // Consistency
  consistencyIssues: ConsistencyIssue[];
  consistencyCheckedAt: string | null;
  setConsistencyIssues: (issues: ConsistencyIssue[], checkedAt: string) => void;

  // Loading
  isLoading: boolean;
  error: string | null;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useForgeStore = create<ForgeState>((set) => ({
  // View
  activeView: "discovery",
  setActiveView: (activeView) => set({ activeView }),

  // Disciplines
  disciplines: [],
  selectedDisciplineId: null,
  setDisciplines: (disciplines) => set({ disciplines }),
  selectDiscipline: (selectedDisciplineId) => set({ selectedDisciplineId }),
  addDiscipline: (discipline) =>
    set((state) => ({ disciplines: [...state.disciplines, discipline] })),

  // Competencies
  competencies: [],
  setCompetencies: (competencies) => set({ competencies }),
  addCompetency: (competency) =>
    set((state) => ({ competencies: [...state.competencies, competency] })),
  updateCompetency: (id, updates) =>
    set((state) => ({
      competencies: state.competencies.map((c) =>
        c.id === id ? { ...c, ...updates } : c,
      ),
    })),
  removeCompetency: (id) =>
    set((state) => ({
      competencies: state.competencies.filter((c) => c.id !== id),
    })),

  // Examples
  examples: [],
  setExamples: (examples) => set({ examples }),
  updateExample: (id, updates) =>
    set((state) => ({
      examples: state.examples.map((e) =>
        e.id === id ? { ...e, ...updates } : e,
      ),
    })),

  // Discovery
  discoverySessionId: null,
  discoveryQuestions: [],
  discoveryAnswers: [],
  discoveryCurrentIndex: 0,
  setDiscoverySession: (discoverySessionId) => set({ discoverySessionId }),
  setDiscoveryQuestions: (discoveryQuestions) => set({ discoveryQuestions }),
  setDiscoveryAnswer: (questionId, answer) =>
    set((state) => {
      const existing = state.discoveryAnswers.findIndex(
        (a) => a.question_id === questionId,
      );
      if (existing >= 0) {
        const updated = [...state.discoveryAnswers];
        updated[existing] = { question_id: questionId, answer };
        return { discoveryAnswers: updated };
      }
      return {
        discoveryAnswers: [
          ...state.discoveryAnswers,
          { question_id: questionId, answer },
        ],
      };
    }),
  setDiscoveryIndex: (discoveryCurrentIndex) => set({ discoveryCurrentIndex }),
  resetDiscovery: () =>
    set({
      discoverySessionId: null,
      discoveryQuestions: [],
      discoveryAnswers: [],
      discoveryCurrentIndex: 0,
    }),

  // Consistency
  consistencyIssues: [],
  consistencyCheckedAt: null,
  setConsistencyIssues: (consistencyIssues, consistencyCheckedAt) =>
    set({ consistencyIssues, consistencyCheckedAt }),

  // Loading
  isLoading: false,
  error: null,
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
