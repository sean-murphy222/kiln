/**
 * Backward-compatible facade — re-exports from useQuarryStore.
 *
 * All existing Quarry components import from this file.
 * New code should import from the specific store directly.
 */

export { useQuarryStore as useStore } from './useQuarryStore';
export type { ChunkingStrategy, ChunkingParameters } from './useQuarryStore';

// Selector hooks — re-export from quarry store
import { useQuarryStore } from './useQuarryStore';

export const useProject = () => useQuarryStore((s) => s.project);
export const useSelectedDocument = () => useQuarryStore((s) => s.getSelectedDocument());
export const useSelectedChunks = () => useQuarryStore((s) => s.getSelectedChunks());
export const useIsLoading = () => useQuarryStore((s) => s.isLoading);
export const useError = () => useQuarryStore((s) => s.error);

export const useHierarchyTree = () => useQuarryStore((s) => s.hierarchyTree);
export const useSelectedNodeId = () => useQuarryStore((s) => s.selectedNodeId);
export const useIsHierarchyLoading = () => useQuarryStore((s) => s.isHierarchyLoading);

export const useSelectedStrategy = () => useQuarryStore((s) => s.selectedStrategy);
export const useChunkingParameters = () => useQuarryStore((s) => s.chunkingParameters);

export const useComparisonResults = () => useQuarryStore((s) => s.comparisonResults);
export const useIsComparing = () => useQuarryStore((s) => s.isComparing);
export const useRecommendation = () => useQuarryStore((s) => s.recommendation);

export const useTestQuery = () => useQuarryStore((s) => s.testQuery);
export const useQueryResults = () => useQuarryStore((s) => s.queryResults);
export const useIsTesting = () => useQuarryStore((s) => s.isTesting);
