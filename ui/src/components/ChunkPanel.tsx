import { useState } from 'react';
import {
  Lock,
  Unlock,
  Merge,
  Scissors,
  Tag,
  MessageSquare,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { chunkAPI, projectAPI } from '../api/chonk';
import type { Chunk } from '../api/chonk';
import { SplitModal } from './SplitModal';
import { RechunkModal } from './RechunkModal';

export function ChunkPanel() {
  const {
    project,
    selectedDocumentId,
    selectedChunkIds,
    selectChunk,
    toggleChunkSelection,
    clearChunkSelection,
    setProject,
    setError,
  } = useStore();

  const [splitModalChunk, setSplitModalChunk] = useState<Chunk | null>(null);
  const [showRechunkModal, setShowRechunkModal] = useState(false);

  const document = project?.documents.find((d) => d.id === selectedDocumentId);
  const selectedChunks = document?.chunks.filter((c) =>
    selectedChunkIds.includes(c.id)
  );

  // Handle merge
  const handleMerge = async () => {
    if (selectedChunkIds.length < 2) return;

    try {
      await chunkAPI.merge(selectedChunkIds);
      const updatedProject = await projectAPI.get();
      setProject(updatedProject);
      clearChunkSelection();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to merge chunks');
    }
  };

  if (!document) {
    return (
      <div className="h-full flex items-center justify-center text-kiln-500 text-sm">
        No document selected
      </div>
    );
  }

  return (
    <>
    <div className="h-full flex flex-col bg-kiln-800">
      {/* Header */}
      <div className="panel-header flex items-center justify-between">
        <span>Chunks</span>
        <div className="flex items-center gap-2">
          <button
            className="text-kiln-500 hover:text-ember"
            onClick={() => setShowRechunkModal(true)}
            title="Rechunk document"
          >
            <RefreshCw size={14} />
          </button>
          <span className="text-kiln-500">{document.chunks.length}</span>
        </div>
      </div>

      {/* Actions */}
      {selectedChunkIds.length > 0 && (
        <div className="px-3 py-2 border-b border-kiln-600 flex items-center gap-2">
          <span className="text-xs text-kiln-500">
            {selectedChunkIds.length} selected
          </span>
          <div className="flex-1" />
          {selectedChunkIds.length === 1 && (
            <button
              className="btn-secondary py-1 px-2 text-xs flex items-center gap-1"
              onClick={() => setSplitModalChunk(selectedChunks![0])}
              title="Split selected chunk"
            >
              <Scissors size={12} />
              Split
            </button>
          )}
          {selectedChunkIds.length >= 2 && (
            <button
              className="btn-secondary py-1 px-2 text-xs flex items-center gap-1"
              onClick={handleMerge}
              title="Merge selected chunks"
            >
              <Merge size={12} />
              Merge
            </button>
          )}
          <button
            className="text-xs text-kiln-500 hover:text-kiln-300"
            onClick={clearChunkSelection}
          >
            Clear
          </button>
        </div>
      )}

      {/* Chunk list */}
      <div className="flex-1 overflow-y-auto">
        {document.chunks.map((chunk, index) => (
          <ChunkListItem
            key={chunk.id}
            chunk={chunk}
            index={index}
            isSelected={selectedChunkIds.includes(chunk.id)}
            onSelect={() => selectChunk(chunk.id)}
            onToggle={() => toggleChunkSelection(chunk.id)}
          />
        ))}
      </div>

      {/* Selected chunk detail */}
      {selectedChunks?.length === 1 && (
        <ChunkDetail
          chunk={selectedChunks[0]}
          document={document}
        />
      )}
    </div>
    </>
  );
}

interface ChunkListItemProps {
  chunk: Chunk;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
  onToggle: () => void;
}

function ChunkListItem({
  chunk,
  index,
  isSelected,
  onSelect,
  onToggle,
}: ChunkListItemProps) {
  const qualityColor =
    chunk.quality.overall >= 0.85
      ? 'bg-success'
      : chunk.quality.overall >= 0.7
      ? 'bg-warning'
      : 'bg-error';

  return (
    <div
      className={`
        px-3 py-2 border-b border-kiln-600/50 cursor-pointer
        transition-colors
        ${isSelected ? 'bg-ember/20' : 'hover:bg-kiln-700'}
      `}
      onClick={onSelect}
    >
      <div className="flex items-center gap-2">
        {/* Selection checkbox */}
        <input
          type="checkbox"
          checked={isSelected}
          onChange={(e) => {
            e.stopPropagation();
            onToggle();
          }}
          className="w-4 h-4 rounded border-kiln-600 bg-kiln-900"
        />

        {/* Chunk info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-kiln-500">
              #{index + 1}
            </span>
            {chunk.is_locked && (
              <Lock size={12} className="text-warning" />
            )}
            {chunk.is_modified && (
              <span className="w-1.5 h-1.5 rounded-full bg-ember" />
            )}
          </div>
          <p className="text-xs text-kiln-300 truncate mt-0.5">
            {chunk.content.slice(0, 60)}...
          </p>
        </div>

        {/* Quality indicator */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-kiln-500">
            {chunk.token_count}
          </span>
          <div
            className={`w-2 h-2 rounded-full ${qualityColor}`}
            title={`Quality: ${Math.round(chunk.quality.overall * 100)}%`}
          />
        </div>
      </div>

      {/* Tags */}
      {chunk.user_metadata.tags.length > 0 && (
        <div className="flex gap-1 mt-1 ml-6">
          {chunk.user_metadata.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="text-[10px] px-1.5 py-0.5 rounded bg-kiln-700 text-kiln-300"
            >
              {tag}
            </span>
          ))}
          {chunk.user_metadata.tags.length > 3 && (
            <span className="text-[10px] text-kiln-500">
              +{chunk.user_metadata.tags.length - 3}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

interface ChunkDetailProps {
  chunk: Chunk;
  document: { id: string };
}

function ChunkDetail({ chunk, document }: ChunkDetailProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [newTag, setNewTag] = useState('');
  const { setProject, setError } = useStore();

  const handleAddTag = async () => {
    if (!newTag.trim()) return;

    try {
      await chunkAPI.update(chunk.id, {
        tags: [...chunk.user_metadata.tags, newTag.trim()],
      });
      const updatedProject = await projectAPI.get();
      setProject(updatedProject);
      setNewTag('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add tag');
    }
  };

  const handleRemoveTag = async (tag: string) => {
    try {
      await chunkAPI.update(chunk.id, {
        tags: chunk.user_metadata.tags.filter((t) => t !== tag),
      });
      const updatedProject = await projectAPI.get();
      setProject(updatedProject);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove tag');
    }
  };

  const handleToggleLock = async () => {
    try {
      await chunkAPI.update(chunk.id, {
        is_locked: !chunk.is_locked,
      });
      const updatedProject = await projectAPI.get();
      setProject(updatedProject);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update chunk');
    }
  };

  return (
    <div className="border-t border-kiln-600 bg-kiln-900">
      {/* Header */}
      <button
        className="w-full px-3 py-2 flex items-center gap-2 text-left hover:bg-kiln-700"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? (
          <ChevronDown size={14} />
        ) : (
          <ChevronRight size={14} />
        )}
        <span className="text-xs font-medium text-kiln-300">
          Chunk Details
        </span>
      </button>

      {isExpanded && (
        <div className="px-3 pb-3 space-y-3">
          {/* Stats */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-kiln-500">Tokens:</span>
              <span className="ml-2 text-kiln-100">{chunk.token_count}</span>
            </div>
            <div>
              <span className="text-kiln-500">Quality:</span>
              <span className="ml-2 text-kiln-100">
                {Math.round(chunk.quality.overall * 100)}%
              </span>
            </div>
          </div>

          {/* Quality breakdown */}
          {chunk.quality.overall < 0.85 && (
            <div className="p-2 rounded bg-warning/10 border border-warning/30">
              <div className="flex items-center gap-2 text-xs text-warning">
                <AlertTriangle size={12} />
                <span>Quality issues detected</span>
              </div>
            </div>
          )}

          {/* Tags */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Tag size={12} className="text-kiln-500" />
              <span className="text-xs text-kiln-500">Tags</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {chunk.user_metadata.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs px-2 py-0.5 rounded bg-kiln-700 text-kiln-300 flex items-center gap-1"
                >
                  {tag}
                  <button
                    className="hover:text-error"
                    onClick={() => handleRemoveTag(tag)}
                  >
                    ×
                  </button>
                </span>
              ))}
              <input
                type="text"
                className="text-xs px-2 py-0.5 w-20 bg-transparent border-b border-kiln-600 text-kiln-100 placeholder:text-kiln-500 focus:outline-none focus:border-ember"
                placeholder="Add..."
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAddTag()}
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <button
              className="btn-secondary py-1 px-2 text-xs flex items-center gap-1"
              onClick={handleToggleLock}
            >
              {chunk.is_locked ? (
                <>
                  <Unlock size={12} />
                  Unlock
                </>
              ) : (
                <>
                  <Lock size={12} />
                  Lock
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
