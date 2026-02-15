/**
 * Global state management using Zustand.
 */

import { create } from 'zustand';
import type {
  Project,
  Document,
  Chunk,
  SearchResult,
  HierarchyTree,
  StrategyResult,
  ComparisonResult,
  StrategyQueryResult,
} from '../api/chonk';

export type ChunkingStrategy = 'hierarchical' | 'fixed' | 'semantic' | 'custom';

export interface ChunkingParameters {
  max_tokens: number;
  overlap_tokens: number;
  preserve_tables: boolean;
  preserve_code: boolean;
  group_under_headings: boolean;
  heading_weight: number;
}

interface AppState {
  // Project state
  project: Project | null;
  isLoading: boolean;
  error: string | null;

  // UI state
  selectedDocumentId: string | null;
  selectedChunkIds: string[];
  searchQuery: string;
  searchResults: SearchResult[];
  isSearching: boolean;

  // Hierarchy state
  hierarchyTree: HierarchyTree | null;
  isHierarchyLoading: boolean;
  selectedNodeId: string | null;

  // Strategy state
  selectedStrategy: ChunkingStrategy;
  chunkingParameters: ChunkingParameters;

  // Comparison state
  comparisonResults: StrategyResult[];
  isComparing: boolean;
  recommendation: string | null;

  // Query testing state
  testQuery: string;
  queryResults: Record<string, StrategyQueryResult>;
  isTesting: boolean;

  // View state
  sidebarOpen: boolean;
  testPanelOpen: boolean;
  hierarchyPanelOpen: boolean;

  // Actions
  setProject: (project: Project | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  selectDocument: (documentId: string | null) => void;
  selectChunk: (chunkId: string) => void;
  selectChunks: (chunkIds: string[]) => void;
  toggleChunkSelection: (chunkId: string) => void;
  clearChunkSelection: () => void;

  setSearchQuery: (query: string) => void;
  setSearchResults: (results: SearchResult[]) => void;
  setSearching: (searching: boolean) => void;

  toggleSidebar: () => void;
  toggleTestPanel: () => void;
  toggleHierarchyPanel: () => void;

  // Hierarchy actions
  setHierarchyTree: (tree: HierarchyTree | null) => void;
  setHierarchyLoading: (loading: boolean) => void;
  selectNode: (nodeId: string | null) => void;

  // Strategy actions
  setStrategy: (strategy: ChunkingStrategy) => void;
  setParameters: (parameters: Partial<ChunkingParameters>) => void;
  resetParameters: () => void;

  // Comparison actions
  setComparisonResults: (results: StrategyResult[]) => void;
  setComparing: (comparing: boolean) => void;
  setRecommendation: (recommendation: string | null) => void;

  // Query testing actions
  setTestQuery: (query: string) => void;
  setQueryResults: (results: Record<string, StrategyQueryResult>) => void;
  setTesting: (testing: boolean) => void;

  // Computed getters
  getSelectedDocument: () => Document | null;
  getSelectedChunks: () => Chunk[];
}

const DEFAULT_PARAMETERS: ChunkingParameters = {
  max_tokens: 600,
  overlap_tokens: 50,
  preserve_tables: true,
  preserve_code: true,
  group_under_headings: true,
  heading_weight: 1.5,
};

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  project: null,
  isLoading: false,
  error: null,
  selectedDocumentId: null,
  selectedChunkIds: [],
  searchQuery: '',
  searchResults: [],
  isSearching: false,

  // Hierarchy state
  hierarchyTree: null,
  isHierarchyLoading: false,
  selectedNodeId: null,

  // Strategy state
  selectedStrategy: 'hierarchical',
  chunkingParameters: DEFAULT_PARAMETERS,

  // Comparison state
  comparisonResults: [],
  isComparing: false,
  recommendation: null,

  // Query testing state
  testQuery: '',
  queryResults: {},
  isTesting: false,

  // View state
  sidebarOpen: true,
  testPanelOpen: true,
  hierarchyPanelOpen: true,

  // Actions
  setProject: (project) => set({ project }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),

  selectDocument: (documentId) =>
    set({
      selectedDocumentId: documentId,
      selectedChunkIds: [],
    }),

  selectChunk: (chunkId) =>
    set({ selectedChunkIds: [chunkId] }),

  selectChunks: (chunkIds) =>
    set({ selectedChunkIds: chunkIds }),

  toggleChunkSelection: (chunkId) =>
    set((state) => {
      const isSelected = state.selectedChunkIds.includes(chunkId);
      return {
        selectedChunkIds: isSelected
          ? state.selectedChunkIds.filter((id) => id !== chunkId)
          : [...state.selectedChunkIds, chunkId],
      };
    }),

  clearChunkSelection: () =>
    set({ selectedChunkIds: [] }),

  setSearchQuery: (searchQuery) =>
    set({ searchQuery }),

  setSearchResults: (searchResults) =>
    set({ searchResults }),

  setSearching: (isSearching) =>
    set({ isSearching }),

  toggleSidebar: () =>
    set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  toggleTestPanel: () =>
    set((state) => ({ testPanelOpen: !state.testPanelOpen })),

  toggleHierarchyPanel: () =>
    set((state) => ({ hierarchyPanelOpen: !state.hierarchyPanelOpen })),

  // Hierarchy actions
  setHierarchyTree: (hierarchyTree) =>
    set({ hierarchyTree }),

  setHierarchyLoading: (isHierarchyLoading) =>
    set({ isHierarchyLoading }),

  selectNode: (selectedNodeId) =>
    set({ selectedNodeId }),

  // Strategy actions
  setStrategy: (selectedStrategy) =>
    set({ selectedStrategy }),

  setParameters: (parameters) =>
    set((state) => ({
      chunkingParameters: { ...state.chunkingParameters, ...parameters },
    })),

  resetParameters: () =>
    set({ chunkingParameters: DEFAULT_PARAMETERS }),

  // Comparison actions
  setComparisonResults: (comparisonResults) =>
    set({ comparisonResults }),

  setComparing: (isComparing) =>
    set({ isComparing }),

  setRecommendation: (recommendation) =>
    set({ recommendation }),

  // Query testing actions
  setTestQuery: (testQuery) =>
    set({ testQuery }),

  setQueryResults: (queryResults) =>
    set({ queryResults }),

  setTesting: (isTesting) =>
    set({ isTesting }),

  // Computed getters
  getSelectedDocument: () => {
    const { project, selectedDocumentId } = get();
    if (!project || !selectedDocumentId) return null;
    return project.documents.find((d) => d.id === selectedDocumentId) ?? null;
  },

  getSelectedChunks: () => {
    const { project, selectedDocumentId, selectedChunkIds } = get();
    if (!project || !selectedDocumentId) return [];

    const document = project.documents.find((d) => d.id === selectedDocumentId);
    if (!document) return [];

    return document.chunks.filter((c) => selectedChunkIds.includes(c.id));
  },
}));

// Selector hooks for common patterns
export const useProject = () => useStore((state) => state.project);
export const useSelectedDocument = () => useStore((state) => state.getSelectedDocument());
export const useSelectedChunks = () => useStore((state) => state.getSelectedChunks());
export const useIsLoading = () => useStore((state) => state.isLoading);
export const useError = () => useStore((state) => state.error);

// Hierarchy selectors
export const useHierarchyTree = () => useStore((state) => state.hierarchyTree);
export const useSelectedNodeId = () => useStore((state) => state.selectedNodeId);
export const useIsHierarchyLoading = () => useStore((state) => state.isHierarchyLoading);

// Strategy selectors
export const useSelectedStrategy = () => useStore((state) => state.selectedStrategy);
export const useChunkingParameters = () => useStore((state) => state.chunkingParameters);

// Comparison selectors
export const useComparisonResults = () => useStore((state) => state.comparisonResults);
export const useIsComparing = () => useStore((state) => state.isComparing);
export const useRecommendation = () => useStore((state) => state.recommendation);

// Query testing selectors
export const useTestQuery = () => useStore((state) => state.testQuery);
export const useQueryResults = () => useStore((state) => state.queryResults);
export const useIsTesting = () => useStore((state) => state.isTesting);
