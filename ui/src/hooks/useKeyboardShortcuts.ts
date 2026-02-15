/**
 * Keyboard shortcuts hook for CHONK.
 *
 * Provides global keyboard shortcuts for common actions.
 */

import { useEffect, useCallback } from 'react';
import { useStore } from '../store/useStore';
import { projectAPI, documentAPI } from '../api/chonk';

interface KeyboardShortcutsOptions {
  onOpenFile?: () => void;
  onSave?: () => void;
  onExport?: () => void;
  onSettings?: () => void;
}

export function useKeyboardShortcuts(options: KeyboardShortcutsOptions = {}) {
  const {
    project,
    setProject,
    setLoading,
    setError,
    selectedChunkIds,
    clearChunkSelection,
    selectChunks,
    toggleSidebar,
    toggleTestPanel,
    selectedDocumentId,
  } = useStore();

  // Get current document's chunks for navigation
  const getCurrentChunks = useCallback(() => {
    if (!project || !selectedDocumentId) return [];
    const doc = project.documents.find((d) => d.id === selectedDocumentId);
    return doc?.chunks || [];
  }, [project, selectedDocumentId]);

  // Handle keyboard events
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      const isMod = e.ctrlKey || e.metaKey;
      const isShift = e.shiftKey;

      // Global shortcuts (Ctrl/Cmd + key)
      if (isMod) {
        switch (e.key.toLowerCase()) {
          // File operations
          case 'o':
            e.preventDefault();
            options.onOpenFile?.();
            break;

          case 's':
            e.preventDefault();
            options.onSave?.();
            break;

          case 'e':
            e.preventDefault();
            options.onExport?.();
            break;

          case ',':
            e.preventDefault();
            options.onSettings?.();
            break;

          // Selection
          case 'a':
            if (selectedDocumentId) {
              e.preventDefault();
              const chunks = getCurrentChunks();
              selectChunks(chunks.map((c) => c.id));
            }
            break;

          // Panel toggles
          case 'b':
            e.preventDefault();
            toggleSidebar();
            break;

          case 't':
            e.preventDefault();
            toggleTestPanel();
            break;

          default:
            break;
        }
      }

      // Non-modifier shortcuts
      switch (e.key) {
        case 'Escape':
          e.preventDefault();
          clearChunkSelection();
          break;

        // Arrow navigation for chunks
        case 'ArrowUp':
        case 'ArrowDown':
          if (selectedDocumentId && selectedChunkIds.length > 0) {
            e.preventDefault();
            const chunks = getCurrentChunks();
            const currentIndex = chunks.findIndex(
              (c) => c.id === selectedChunkIds[selectedChunkIds.length - 1]
            );

            if (currentIndex !== -1) {
              const newIndex =
                e.key === 'ArrowUp'
                  ? Math.max(0, currentIndex - 1)
                  : Math.min(chunks.length - 1, currentIndex + 1);

              if (isShift) {
                // Extend selection
                const newChunkId = chunks[newIndex].id;
                if (!selectedChunkIds.includes(newChunkId)) {
                  selectChunks([...selectedChunkIds, newChunkId]);
                }
              } else {
                // Single selection
                selectChunks([chunks[newIndex].id]);
              }
            }
          }
          break;

        // Quick actions
        case 'Delete':
        case 'Backspace':
          // Would handle chunk deletion if implemented
          break;

        default:
          break;
      }

      // Number keys 1-9 for quick document selection
      if (!isMod && /^[1-9]$/.test(e.key)) {
        const docIndex = parseInt(e.key) - 1;
        if (project && project.documents[docIndex]) {
          e.preventDefault();
          useStore.getState().selectDocument(project.documents[docIndex].id);
        }
      }
    },
    [
      options,
      selectedChunkIds,
      clearChunkSelection,
      selectChunks,
      toggleSidebar,
      toggleTestPanel,
      selectedDocumentId,
      getCurrentChunks,
      project,
    ]
  );

  // Add event listener
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Return list of shortcuts for help display
  return {
    shortcuts: [
      { keys: ['Ctrl', 'O'], description: 'Open file' },
      { keys: ['Ctrl', 'S'], description: 'Save project' },
      { keys: ['Ctrl', 'E'], description: 'Export chunks' },
      { keys: ['Ctrl', ','], description: 'Open settings' },
      { keys: ['Ctrl', 'A'], description: 'Select all chunks' },
      { keys: ['Ctrl', 'B'], description: 'Toggle sidebar' },
      { keys: ['Ctrl', 'T'], description: 'Toggle test panel' },
      { keys: ['Esc'], description: 'Clear selection' },
      { keys: ['↑', '↓'], description: 'Navigate chunks' },
      { keys: ['Shift', '↑/↓'], description: 'Extend selection' },
      { keys: ['1-9'], description: 'Select document' },
    ],
  };
}
