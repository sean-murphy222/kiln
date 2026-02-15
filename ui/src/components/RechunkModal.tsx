import { useState } from 'react';
import { X, RefreshCw, Sliders } from 'lucide-react';
import { useStore } from '../store/useStore';
import { documentAPI, projectAPI } from '../api/chonk';

interface RechunkModalProps {
  documentId: string;
  documentName: string;
  currentChunker: string;
  onClose: () => void;
}

export function RechunkModal({
  documentId,
  documentName,
  currentChunker,
  onClose,
}: RechunkModalProps) {
  const { setProject, setError } = useStore();

  const [chunker, setChunker] = useState(currentChunker);
  const [targetTokens, setTargetTokens] = useState(400);
  const [maxTokens, setMaxTokens] = useState(600);
  const [minTokens, setMinTokens] = useState(100);
  const [overlapTokens, setOverlapTokens] = useState(50);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const chunkers = [
    {
      id: 'hierarchy',
      name: 'Hierarchy',
      description: 'Respects headings and document structure (recommended)',
    },
    {
      id: 'recursive',
      name: 'Recursive',
      description: 'Splits on natural boundaries (paragraphs, sentences)',
    },
    {
      id: 'fixed',
      name: 'Fixed Size',
      description: 'Simple fixed-size chunks with overlap',
    },
  ];

  const handleRechunk = async () => {
    setIsSubmitting(true);
    try {
      const result = await documentAPI.rechunk(documentId, {
        chunker,
        target_tokens: targetTokens,
        max_tokens: maxTokens,
        min_tokens: minTokens,
        overlap_tokens: overlapTokens,
      });

      const updatedProject = await projectAPI.get();
      setProject(updatedProject);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rechunk document');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="card-pixel w-full max-w-lg animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-chonk-slate">
          <div className="flex items-center gap-2">
            <Sliders size={18} className="text-accent-primary" />
            <h2 className="text-sm font-medium text-chonk-white">
              Rechunk Document
            </h2>
          </div>
          <button
            className="p-1 text-chonk-gray hover:text-chonk-white"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6">
          <p className="text-sm text-chonk-gray">
            Adjust chunking settings for{' '}
            <span className="text-chonk-white">{documentName}</span>
          </p>

          {/* Chunker selection */}
          <div>
            <label className="block text-xs text-chonk-gray mb-2">
              Chunking Strategy
            </label>
            <div className="space-y-2">
              {chunkers.map((c) => (
                <label
                  key={c.id}
                  className={`
                    flex items-start gap-3 p-3 rounded cursor-pointer transition-colors
                    ${chunker === c.id
                      ? 'bg-accent-primary/20 border border-accent-primary'
                      : 'bg-surface-card border border-transparent hover:border-chonk-slate'
                    }
                  `}
                >
                  <input
                    type="radio"
                    name="chunker"
                    value={c.id}
                    checked={chunker === c.id}
                    onChange={(e) => setChunker(e.target.value)}
                    className="mt-1"
                  />
                  <div>
                    <div className="text-sm font-medium text-chonk-white">
                      {c.name}
                    </div>
                    <div className="text-xs text-chonk-gray">{c.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Token settings */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-chonk-gray mb-2">
                Target Tokens
              </label>
              <input
                type="number"
                className="input-pixel"
                value={targetTokens}
                onChange={(e) => setTargetTokens(parseInt(e.target.value) || 400)}
                min={50}
                max={2000}
              />
              <p className="text-[10px] text-chonk-slate mt-1">
                Ideal chunk size (200-500 recommended)
              </p>
            </div>
            <div>
              <label className="block text-xs text-chonk-gray mb-2">
                Max Tokens
              </label>
              <input
                type="number"
                className="input-pixel"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 600)}
                min={100}
                max={4000}
              />
              <p className="text-[10px] text-chonk-slate mt-1">
                Maximum chunk size
              </p>
            </div>
            <div>
              <label className="block text-xs text-chonk-gray mb-2">
                Min Tokens
              </label>
              <input
                type="number"
                className="input-pixel"
                value={minTokens}
                onChange={(e) => setMinTokens(parseInt(e.target.value) || 100)}
                min={10}
                max={500}
              />
              <p className="text-[10px] text-chonk-slate mt-1">
                Merge chunks smaller than this
              </p>
            </div>
            <div>
              <label className="block text-xs text-chonk-gray mb-2">
                Overlap Tokens
              </label>
              <input
                type="number"
                className="input-pixel"
                value={overlapTokens}
                onChange={(e) => setOverlapTokens(parseInt(e.target.value) || 50)}
                min={0}
                max={200}
              />
              <p className="text-[10px] text-chonk-slate mt-1">
                Content shared between chunks
              </p>
            </div>
          </div>

          {/* Warning about locked chunks */}
          <div className="p-3 rounded bg-surface-card border border-chonk-slate">
            <p className="text-xs text-chonk-gray">
              <span className="text-accent-warning">Note:</span> Locked chunks
              will be preserved and re-added after rechunking.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-4 py-3 border-t border-chonk-slate">
          <button className="btn-pixel" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </button>
          <button
            className="btn-pixel-primary flex items-center gap-2"
            onClick={handleRechunk}
            disabled={isSubmitting}
          >
            <RefreshCw size={14} className={isSubmitting ? 'animate-spin' : ''} />
            {isSubmitting ? 'Processing...' : 'Rechunk'}
          </button>
        </div>
      </div>
    </div>
  );
}
