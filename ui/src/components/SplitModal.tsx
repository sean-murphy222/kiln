import { useState, useEffect, useRef } from 'react';
import { X, Scissors } from 'lucide-react';
import { useStore } from '../store/useStore';
import { chunkAPI, projectAPI } from '../api/chonk';
import type { Chunk } from '../api/chonk';

interface SplitModalProps {
  chunk: Chunk;
  onClose: () => void;
}

export function SplitModal({ chunk, onClose }: SplitModalProps) {
  const { setProject, setError, clearChunkSelection } = useStore();
  const [splitPosition, setSplitPosition] = useState(
    Math.floor(chunk.content.length / 2)
  );
  const [isSubmitting, setIsSubmitting] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  // Handle text selection to set split position
  const handleTextSelect = () => {
    const selection = window.getSelection();
    if (selection && selection.rangeCount > 0 && contentRef.current) {
      const range = selection.getRangeAt(0);
      const preRange = document.createRange();
      preRange.selectNodeContents(contentRef.current);
      preRange.setEnd(range.startContainer, range.startOffset);
      const position = preRange.toString().length;
      if (position > 0 && position < chunk.content.length) {
        setSplitPosition(position);
      }
    }
  };

  const handleSplit = async () => {
    if (splitPosition <= 0 || splitPosition >= chunk.content.length) {
      setError('Invalid split position');
      return;
    }

    setIsSubmitting(true);
    try {
      await chunkAPI.split(chunk.id, splitPosition);
      const updatedProject = await projectAPI.get();
      setProject(updatedProject);
      clearChunkSelection();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to split chunk');
    } finally {
      setIsSubmitting(false);
    }
  };

  const beforeText = chunk.content.slice(0, splitPosition);
  const afterText = chunk.content.slice(splitPosition);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="card w-full max-w-3xl max-h-[80vh] flex flex-col animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-kiln-600">
          <div className="flex items-center gap-2">
            <Scissors size={18} className="text-ember" />
            <h2 className="text-sm font-medium text-kiln-100">Split Chunk</h2>
          </div>
          <button
            className="p-1 text-kiln-500 hover:text-kiln-100"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <p className="text-sm text-kiln-500 mb-4">
            Click or select text to choose where to split. The chunk will be
            divided at the cursor position.
          </p>

          {/* Split position slider */}
          <div className="mb-4">
            <div className="flex items-center justify-between text-xs text-kiln-500 mb-2">
              <span>Position: {splitPosition}</span>
              <span>
                {Math.round((splitPosition / chunk.content.length) * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={chunk.content.length - 1}
              value={splitPosition}
              onChange={(e) => setSplitPosition(parseInt(e.target.value))}
              className="w-full h-2 bg-kiln-900 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Preview */}
          <div className="grid grid-cols-2 gap-4">
            {/* First chunk */}
            <div className="card p-3 border-success">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-success">
                  Chunk A
                </span>
                <span className="text-xs text-kiln-500">
                  ~{Math.round(beforeText.length / 4)} tokens
                </span>
              </div>
              <div
                ref={contentRef}
                className="text-sm text-kiln-300 max-h-48 overflow-y-auto whitespace-pre-wrap"
                onMouseUp={handleTextSelect}
              >
                {beforeText}
              </div>
            </div>

            {/* Second chunk */}
            <div className="card p-3 border-warning">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-warning">
                  Chunk B
                </span>
                <span className="text-xs text-kiln-500">
                  ~{Math.round(afterText.length / 4)} tokens
                </span>
              </div>
              <div className="text-sm text-kiln-300 max-h-48 overflow-y-auto whitespace-pre-wrap">
                {afterText}
              </div>
            </div>
          </div>

          {/* Warnings */}
          {(beforeText.length < 100 || afterText.length < 100) && (
            <div className="mt-4 p-3 rounded bg-warning/10 border border-warning/30">
              <p className="text-xs text-warning">
                Warning: One or both chunks will be very small. Consider
                adjusting the split position.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-4 py-3 border-t border-kiln-600">
          <button className="btn-secondary" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </button>
          <button
            className="btn-primary flex items-center gap-2"
            onClick={handleSplit}
            disabled={isSubmitting}
          >
            <Scissors size={14} />
            {isSubmitting ? 'Splitting...' : 'Split Chunk'}
          </button>
        </div>
      </div>
    </div>
  );
}
