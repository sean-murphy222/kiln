/**
 * Quarry-specific state — documents, chunks, search, hierarchy, strategies.
 *
 * Extracted from the original monolithic useStore.ts.
 * The old useStore.ts re-exports everything for backward compatibility.
 */

import { create } from 'zustand';
import type {
  Project,
  Document,
  Chunk,
  SearchResult,
  HierarchyTree,
  StrategyResult,
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

interface QuarryState {
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

  setHierarchyTree: (tree: HierarchyTree | null) => void;
  setHierarchyLoading: (loading: boolean) => void;
  selectNode: (nodeId: string | null) => void;

  setStrategy: (strategy: ChunkingStrategy) => void;
  setParameters: (parameters: Partial<ChunkingParameters>) => void;
  resetParameters: () => void;

  setComparisonResults: (results: StrategyResult[]) => void;
  setComparing: (comparing: boolean) => void;
  setRecommendation: (recommendation: string | null) => void;

  setTestQuery: (query: string) => void;
  setQueryResults: (results: Record<string, StrategyQueryResult>) => void;
  setTesting: (testing: boolean) => void;

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

export const useQuarryStore = create<QuarryState>((set, get) => ({
  project: null,
  isLoading: false,
  error: null,
  selectedDocumentId: null,
  selectedChunkIds: [],
  searchQuery: '',
  searchResults: [],
  isSearching: false,

  hierarchyTree: null,
  isHierarchyLoading: false,
  selectedNodeId: null,

  selectedStrategy: 'hierarchical',
  chunkingParameters: DEFAULT_PARAMETERS,

  comparisonResults: [],
  isComparing: false,
  recommendation: null,

  testQuery: '',
  queryResults: {},
  isTesting: false,

  sidebarOpen: true,
  testPanelOpen: true,
  hierarchyPanelOpen: true,

  setProject: (project) => set({ project }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),

  selectDocument: (documentId) =>
    set({ selectedDocumentId: documentId, selectedChunkIds: [] }),

  selectChunk: (chunkId) => set({ selectedChunkIds: [chunkId] }),
  selectChunks: (chunkIds) => set({ selectedChunkIds: chunkIds }),

  toggleChunkSelection: (chunkId) =>
    set((state) => ({
      selectedChunkIds: state.selectedChunkIds.includes(chunkId)
        ? state.selectedChunkIds.filter((id) => id !== chunkId)
        : [...state.selectedChunkIds, chunkId],
    })),

  clearChunkSelection: () => set({ selectedChunkIds: [] }),

  setSearchQuery: (searchQuery) => set({ searchQuery }),
  setSearchResults: (searchResults) => set({ searchResults }),
  setSearching: (isSearching) => set({ isSearching }),

  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  toggleTestPanel: () => set((s) => ({ testPanelOpen: !s.testPanelOpen })),
  toggleHierarchyPanel: () => set((s) => ({ hierarchyPanelOpen: !s.hierarchyPanelOpen })),

  setHierarchyTree: (hierarchyTree) => set({ hierarchyTree }),
  setHierarchyLoading: (isHierarchyLoading) => set({ isHierarchyLoading }),
  selectNode: (selectedNodeId) => set({ selectedNodeId }),

  setStrategy: (selectedStrategy) => set({ selectedStrategy }),
  setParameters: (parameters) =>
    set((s) => ({ chunkingParameters: { ...s.chunkingParameters, ...parameters } })),
  resetParameters: () => set({ chunkingParameters: DEFAULT_PARAMETERS }),

  setComparisonResults: (comparisonResults) => set({ comparisonResults }),
  setComparing: (isComparing) => set({ isComparing }),
  setRecommendation: (recommendation) => set({ recommendation }),

  setTestQuery: (testQuery) => set({ testQuery }),
  setQueryResults: (queryResults) => set({ queryResults }),
  setTesting: (isTesting) => set({ isTesting }),

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
