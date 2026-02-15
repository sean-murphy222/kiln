import { useState, useCallback } from 'react';
import {
  PanelLeft,
  TestTube,
  Save,
  Download,
  Upload,
  Settings,
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { projectAPI, documentAPI } from '../api/chonk';
import { SettingsModal } from './SettingsModal';
import { ExportModal } from './ExportModal';

export function Toolbar() {
  const {
    project,
    setProject,
    setLoading,
    setError,
    toggleSidebar,
    toggleTestPanel,
    sidebarOpen,
    testPanelOpen,
    selectedDocumentId,
    selectDocument,
  } = useStore();

  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);

  // Get selected document info for export
  const selectedDocument = project?.documents.find((d) => d.id === selectedDocumentId);

  // Handle file upload
  const handleUpload = useCallback(async () => {
    if (!window.electronAPI) {
      // Web fallback - use file input
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.pdf,.docx,.md,.txt';
      input.onchange = async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (!file) return;

        setLoading(true);
        setError(null);
        try {
          const result = await documentAPI.upload(file);
          const updatedProject = await projectAPI.get();
          setProject(updatedProject);
          selectDocument(result.document_id);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to upload document');
        } finally {
          setLoading(false);
        }
      };
      input.click();
      return;
    }

    const result = await window.electronAPI.openFile();
    if (result.canceled || !result.filePaths[0]) return;

    // For Electron, we need to read the file and upload it
    // This is simplified - in production you'd use IPC
    setError('File upload from Electron not fully implemented');
  }, [setProject, setLoading, setError, selectDocument]);

  // Handle save
  const handleSave = useCallback(async () => {
    if (!project) return;

    if (!project.project_path && window.electronAPI) {
      const result = await window.electronAPI.saveProject(project.name + '.chonk');
      if (result.canceled || !result.filePath) return;

      setLoading(true);
      try {
        await projectAPI.save(result.filePath);
        const updatedProject = await projectAPI.get();
        setProject(updatedProject);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to save project');
      } finally {
        setLoading(false);
      }
    } else {
      setLoading(true);
      try {
        await projectAPI.save();
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to save project');
      } finally {
        setLoading(false);
      }
    }
  }, [project, setProject, setLoading, setError]);

  // Handle export - now uses modal
  const handleExport = useCallback(() => {
    setShowExportModal(true);
  }, []);

  return (
    <div className="h-12 bg-surface-panel border-b border-chonk-slate flex items-center px-4 gap-2">
      {/* Left side - toggles */}
      <button
        className={`p-2 rounded hover:bg-surface-card transition-colors ${
          sidebarOpen ? 'text-accent-primary' : 'text-chonk-gray'
        }`}
        onClick={toggleSidebar}
        title="Toggle sidebar"
      >
        <PanelLeft size={20} />
      </button>

      {/* Project name */}
      <div className="text-pixel text-xs text-chonk-light ml-2">
        {project?.name ?? 'CHONK'}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Actions */}
      <button
        className="btn-pixel-primary flex items-center gap-2 py-1.5 px-3"
        onClick={handleUpload}
        title="Add document"
      >
        <Upload size={16} />
        <span className="text-sm">Add Doc</span>
      </button>

      <button
        className="p-2 rounded hover:bg-surface-card text-chonk-light transition-colors"
        onClick={handleSave}
        title="Save project"
      >
        <Save size={20} />
      </button>

      <button
        className="p-2 rounded hover:bg-surface-card text-chonk-light transition-colors"
        onClick={handleExport}
        title="Export chunks"
      >
        <Download size={20} />
      </button>

      <div className="w-px h-6 bg-chonk-slate mx-2" />

      {/* Test panel toggle */}
      <button
        className={`p-2 rounded hover:bg-surface-card transition-colors ${
          testPanelOpen ? 'text-accent-primary' : 'text-chonk-gray'
        }`}
        onClick={toggleTestPanel}
        title="Toggle test panel"
      >
        <TestTube size={20} />
      </button>

      <button
        className="p-2 rounded hover:bg-surface-card text-chonk-gray transition-colors"
        title="Settings"
        onClick={() => setShowSettingsModal(true)}
      >
        <Settings size={20} />
      </button>

      {/* Modals */}
      {showSettingsModal && (
        <SettingsModal onClose={() => setShowSettingsModal(false)} />
      )}
      {showExportModal && (
        <ExportModal
          documentId={selectedDocumentId ?? undefined}
          documentName={selectedDocument?.source_path.split(/[/\\]/).pop()}
          onClose={() => setShowExportModal(false)}
        />
      )}
    </div>
  );
}
