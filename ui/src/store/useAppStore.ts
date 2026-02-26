/**
 * App-level state — active tool, global UI, backend health.
 */

import { create } from 'zustand';

export type ToolId = 'quarry' | 'forge' | 'foundry' | 'hearth';

interface AppState {
  activeTool: ToolId;
  backendReady: boolean;
  backendChecking: boolean;
  globalError: string | null;
  sidebarCollapsed: boolean;

  setActiveTool: (tool: ToolId) => void;
  setBackendReady: (ready: boolean) => void;
  setBackendChecking: (checking: boolean) => void;
  setGlobalError: (error: string | null) => void;
  toggleSidebar: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTool: 'quarry',
  backendReady: false,
  backendChecking: true,
  globalError: null,
  sidebarCollapsed: false,

  setActiveTool: (activeTool) => set({ activeTool }),
  setBackendReady: (backendReady) => set({ backendReady }),
  setBackendChecking: (backendChecking) => set({ backendChecking }),
  setGlobalError: (globalError) => set({ globalError }),
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
