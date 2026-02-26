import { useState } from 'react';
import { FolderOpen, Plus, FileText } from 'lucide-react';

interface WelcomeScreenProps {
  onNewProject: (name: string) => void;
  onOpenProject: () => void;
  isLoading: boolean;
}

export function WelcomeScreen({
  onNewProject,
  onOpenProject,
  isLoading,
}: WelcomeScreenProps) {
  const [projectName, setProjectName] = useState('');
  const [showNewProject, setShowNewProject] = useState(false);

  const handleCreate = () => {
    if (projectName.trim()) {
      onNewProject(projectName.trim());
    }
  };

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-kiln-900">
      {/* Logo */}
      <div className="mb-8">
        <h1 className="text-pixel text-4xl text-ember mb-2">
          CHONK
        </h1>
        <p className="text-kiln-300 text-sm text-center">
          Visual Document Chunking Studio
        </p>
      </div>

      {/* Tagline */}
      <div className="mb-12 text-center max-w-md">
        <p className="text-lg text-kiln-100 mb-2">
          Know your chunks work before you embed them.
        </p>
        <p className="text-sm text-kiln-500">
          Drop docs → See chunks → Test queries → Export JSON
        </p>
      </div>

      {/* Action buttons */}
      <div className="flex flex-col gap-4 w-80">
        {showNewProject ? (
          <div className="card p-4 animate-slide-up">
            <label className="block text-sm text-kiln-300 mb-2">
              Project Name
            </label>
            <input
              type="text"
              className="input-field mb-4"
              placeholder="My RAG Project"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              autoFocus
            />
            <div className="flex gap-2">
              <button
                className="btn-secondary flex-1"
                onClick={() => setShowNewProject(false)}
                disabled={isLoading}
              >
                Cancel
              </button>
              <button
                className="btn-primary flex-1"
                onClick={handleCreate}
                disabled={isLoading || !projectName.trim()}
              >
                {isLoading ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        ) : (
          <>
            <button
              className="btn-primary flex items-center justify-center gap-3 py-4"
              onClick={() => setShowNewProject(true)}
              disabled={isLoading}
            >
              <Plus size={20} />
              <span>New Project</span>
            </button>

            <button
              className="btn-secondary flex items-center justify-center gap-3 py-4"
              onClick={onOpenProject}
              disabled={isLoading}
            >
              <FolderOpen size={20} />
              <span>Open Project</span>
            </button>
          </>
        )}
      </div>

      {/* Features list */}
      <div className="mt-12 grid grid-cols-3 gap-8 text-center">
        <div>
          <div className="text-2xl mb-2">📄</div>
          <div className="text-sm text-kiln-300">PDF, DOCX, MD, TXT</div>
        </div>
        <div>
          <div className="text-2xl mb-2">🔍</div>
          <div className="text-sm text-kiln-300">Test Before Export</div>
        </div>
        <div>
          <div className="text-2xl mb-2">🔒</div>
          <div className="text-sm text-kiln-300">100% Local</div>
        </div>
      </div>

      {/* Version */}
      <div className="absolute bottom-4 text-xs text-kiln-600">
        v0.1.0
      </div>
    </div>
  );
}
