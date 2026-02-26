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
    <div className="h-12 bg-kiln-800 border-b border-kiln-600 flex items-center px-4 gap-2">
      {/* Left side - toggles */}
      <button
        className={`p-2 rounded hover:bg-kiln-700 transition-colors ${
          sidebarOpen ? 'text-ember' : 'text-kiln-500'
        }`}
        onClick={toggleSidebar}
        title="Toggle sidebar"
      >
        <PanelLeft size={20} />
      </button>

      {/* Project name */}
      <div className="text-pixel text-xs text-kiln-300 ml-2">
        {project?.name ?? 'CHONK'}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Actions */}
      <button
        className="btn-primary flex items-center gap-2 py-1.5 px-3"
        onClick={handleUpload}
        title="Add document"
      >
        <Upload size={16} />
        <span className="text-sm">Add Doc</span>
      </button>

      <button
        className="p-2 rounded hover:bg-kiln-700 text-kiln-300 transition-colors"
        onClick={handleSave}
        title="Save project"
      >
        <Save size={20} />
      </button>

      <button
        className="p-2 rounded hover:bg-kiln-700 text-kiln-300 transition-colors"
        onClick={handleExport}
        title="Export chunks"
      >
        <Download size={20} />
      </button>

      <div className="w-px h-6 bg-kiln-600 mx-2" />

      {/* Test panel toggle */}
      <button
        className={`p-2 rounded hover:bg-kiln-700 transition-colors ${
          testPanelOpen ? 'text-ember' : 'text-kiln-500'
        }`}
        onClick={toggleTestPanel}
        title="Toggle test panel"
      >
        <TestTube size={20} />
      </button>

      <button
        className="p-2 rounded hover:bg-kiln-700 text-kiln-500 transition-colors"
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
