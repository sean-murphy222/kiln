import { useState, useCallback, useEffect } from 'react';
import { HelpCircle, Workflow, Grid, AlertTriangle, Loader2 } from 'lucide-react';
import { useStore } from '../store/useStore';
import { Sidebar } from './Sidebar';
import { DocumentViewer } from './DocumentViewer';
import { ChunkPanel } from './ChunkPanel';
import { TestPanel } from './TestPanel';
import { Toolbar } from './Toolbar';
import { DropZone } from './DropZone';
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal';
import { WorkflowPanel } from './WorkflowPanel';
import { DiagnosticDashboard } from './DiagnosticDashboard';
import { DiagnosticWelcome } from './DiagnosticWelcome';
import { OnboardingTour } from './OnboardingTour';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';

type ViewMode = 'diagnostic' | 'workflow' | 'chunks';

export function Layout() {
  const { sidebarOpen, testPanelOpen, project, selectedDocumentId, isLoading, error, setError } = useStore();
  const [showShortcutsModal, setShowShortcutsModal] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('diagnostic');
  const [showTour, setShowTour] = useState(false);

  // Check if user has seen tour on mount
  useEffect(() => {
    const hasSeenTour = localStorage.getItem('chonk_tour_completed');
    if (!hasSeenTour) {
      setShowTour(true);
    }
  }, []);

  const handleCompleteTour = () => {
    localStorage.setItem('chonk_tour_completed', 'true');
    setShowTour(false);
  };

  const handleSkipTour = () => {
    localStorage.setItem('chonk_tour_completed', 'true');
    setShowTour(false);
  };

  // Get selected document for diagnostic dashboard
  const selectedDocument = selectedDocumentId
    ? project?.documents.find(d => d.id === selectedDocumentId)
    : project?.documents[0];

  // Initialize keyboard shortcuts
  useKeyboardShortcuts({
    // Callbacks handled by Toolbar component for now
  });

  return (
    <DropZone>
      <div className="h-screen flex flex-col bg-surface-bg">
        {/* Toolbar */}
        <Toolbar />

        {/* View Mode Switcher */}
        <div className="flex border-b border-gray-700 bg-gray-800">
          <button
            onClick={() => setViewMode('diagnostic')}
            className={`flex items-center gap-2 px-6 py-3 font-medium transition-colors ${
              viewMode === 'diagnostic'
                ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
            }`}
          >
            <AlertTriangle size={18} />
            <span>Diagnostic</span>
            <span className="text-xs px-2 py-0.5 bg-accent-primary rounded">MVP</span>
          </button>
          <button
            onClick={() => setViewMode('workflow')}
            className={`flex items-center gap-2 px-6 py-3 font-medium transition-colors ${
              viewMode === 'workflow'
                ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
            }`}
          >
            <Workflow size={18} />
            <span>Visual Workflow</span>
          </button>
          <button
            onClick={() => setViewMode('chunks')}
            className={`flex items-center gap-2 px-6 py-3 font-medium transition-colors ${
              viewMode === 'chunks'
                ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
            }`}
          >
            <Grid size={18} />
            <span>Chunks View</span>
          </button>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-900/30 border-b-2 border-red-400 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              <span className="text-sm text-red-400">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-300 text-sm font-bold"
            >
              DISMISS
            </button>
          </div>
        )}

        {/* Main content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Sidebar - Document list */}
          {sidebarOpen && (
            <div className="w-64 flex-shrink-0">
              <Sidebar />
            </div>
          )}

          {/* Diagnostic View (MVP Phase 1) */}
          {viewMode === 'diagnostic' && (
            <div className="flex-1 overflow-hidden">
              {selectedDocument ? (
                <DiagnosticDashboard document={selectedDocument} />
              ) : (
                <DiagnosticWelcome />
              )}
            </div>
          )}

          {/* Workflow View */}
          {viewMode === 'workflow' && (
            <div className="flex-1 overflow-hidden">
              <WorkflowPanel />
            </div>
          )}

          {/* Chunks View (Original) */}
          {viewMode === 'chunks' && (
            <>
              {/* Main area - Document viewer + Chunk panel */}
              <div className="flex-1 flex overflow-hidden">
                {/* Document viewer */}
                <div className="flex-1 overflow-hidden">
                  <DocumentViewer />
                </div>

                {/* Chunk panel */}
                <div className="w-80 flex-shrink-0 border-l border-chonk-slate">
                  <ChunkPanel />
                </div>
              </div>

              {/* Test panel */}
              {testPanelOpen && (
                <div className="w-96 flex-shrink-0 border-l border-chonk-slate">
                  <TestPanel />
                </div>
              )}
            </>
          )}
        </div>

        {/* Help buttons */}
        <div className="fixed bottom-4 right-4 flex flex-col gap-2">
          <button
            className="p-3 rounded-full bg-accent-primary border-2 border-black text-black hover:bg-accent-primary/80 transition-colors shadow-lg font-bold"
            onClick={() => setShowTour(true)}
            title="Show workflow guide"
          >
            ?
          </button>
          <button
            className="p-2 rounded-full bg-surface-panel border border-chonk-slate text-chonk-gray hover:text-accent-primary hover:border-accent-primary transition-colors shadow-lg"
            onClick={() => setShowShortcutsModal(true)}
            title="Keyboard shortcuts"
          >
            <HelpCircle size={20} />
          </button>
        </div>

        {/* Modals */}
        {showShortcutsModal && (
          <KeyboardShortcutsModal onClose={() => setShowShortcutsModal(false)} />
        )}
        {showTour && (
          <OnboardingTour
            onComplete={handleCompleteTour}
            onSkip={handleSkipTour}
          />
        )}

        {/* Global Loading Overlay */}
        {isLoading && (
          <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center">
            <div className="bg-surface-panel border-4 border-accent-primary p-8 rounded-lg shadow-2xl">
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="w-12 h-12 text-accent-primary animate-spin" />
                <p className="text-pixel text-lg text-chonk-light">Processing...</p>
                <p className="text-xs text-chonk-gray">
                  Uploading and extracting document
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </DropZone>
  );
}
