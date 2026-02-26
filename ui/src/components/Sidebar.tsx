import { FileText, Trash2, ChevronRight } from 'lucide-react';
import { useStore } from '../store/useStore';
import { documentAPI, projectAPI } from '../api/chonk';

export function Sidebar() {
  const {
    project,
    selectedDocumentId,
    selectDocument,
    setProject,
    setLoading,
    setError,
  } = useStore();

  const handleDeleteDocument = async (e: React.MouseEvent, docId: string) => {
    e.stopPropagation();

    if (!confirm('Delete this document? This cannot be undone.')) return;

    setLoading(true);
    try {
      await documentAPI.delete(docId);
      const updatedProject = await projectAPI.get();
      setProject(updatedProject);

      if (selectedDocumentId === docId) {
        selectDocument(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full bg-kiln-800 flex flex-col border-r border-kiln-600">
      {/* Header */}
      <div className="panel-header flex items-center justify-between">
        <span>Documents</span>
        <span className="text-kiln-500">
          {project?.documents.length ?? 0}
        </span>
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto">
        {project?.documents.length === 0 ? (
          <div className="p-4 text-center text-kiln-500 text-sm">
            <p>No documents yet.</p>
            <p className="mt-2">Click "Add Doc" to get started.</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {project?.documents.map((doc) => {
              const isSelected = doc.id === selectedDocumentId;
              const filename = doc.source_path.split(/[/\\]/).pop() ?? 'Unknown';

              return (
                <div
                  key={doc.id}
                  className={`
                    group flex items-center gap-2 px-3 py-2 rounded cursor-pointer
                    transition-colors
                    ${isSelected
                      ? 'bg-ember/20 text-ember'
                      : 'hover:bg-kiln-700 text-kiln-300'
                    }
                  `}
                  onClick={() => selectDocument(doc.id)}
                >
                  <ChevronRight
                    size={14}
                    className={`transition-transform ${isSelected ? 'rotate-90' : ''}`}
                  />
                  <FileText size={16} className="flex-shrink-0" />
                  <span className="flex-1 truncate text-sm" title={filename}>
                    {filename}
                  </span>
                  <span className="text-xs text-kiln-500">
                    {doc.chunks.length}
                  </span>
                  <button
                    className="opacity-0 group-hover:opacity-100 p-1 hover:text-error transition-opacity"
                    onClick={(e) => handleDeleteDocument(e, doc.id)}
                    title="Delete document"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Project info */}
      <div className="border-t border-kiln-600 p-3 text-xs text-kiln-500">
        <div className="flex justify-between">
          <span>Chunks</span>
          <span>
            {project?.documents.reduce((sum, d) => sum + d.chunks.length, 0) ?? 0}
          </span>
        </div>
      </div>
    </div>
  );
}
