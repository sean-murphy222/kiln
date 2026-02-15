import { useState, useCallback } from 'react';
import { FileText, ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react';
import { useStore } from '../store/useStore';
import type { Chunk, Block } from '../api/chonk';

export function DocumentViewer() {
  const {
    project,
    selectedDocumentId,
    selectedChunkIds,
    selectChunk,
    toggleChunkSelection,
  } = useStore();

  const [viewMode, setViewMode] = useState<'chunks' | 'document'>('chunks');
  const [currentPage, setCurrentPage] = useState(1);
  const [hoveredChunkId, setHoveredChunkId] = useState<string | null>(null);

  const document = project?.documents.find((d) => d.id === selectedDocumentId);

  // No document selected
  if (!document) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-chonk-gray">
        <FileText size={48} className="mb-4 opacity-50" />
        <p>Select a document to view</p>
        <p className="text-sm mt-2">or drag and drop a file here</p>
      </div>
    );
  }

  const pageCount = document.metadata.page_count || 1;

  return (
    <div className="h-full flex flex-col">
      {/* Document header */}
      <div className="px-4 py-3 border-b border-chonk-slate bg-surface-panel">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-medium text-chonk-white">
              {document.source_path.split(/[/\\]/).pop()}
            </h2>
            <p className="text-xs text-chonk-gray mt-0.5">
              {document.metadata.page_count > 0 && `${document.metadata.page_count} pages · `}
              {document.metadata.word_count.toLocaleString()} words · {document.chunks.length} chunks
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* View mode toggle */}
            <div className="flex rounded overflow-hidden border border-chonk-slate">
              <button
                className={`px-3 py-1 text-xs transition-colors ${
                  viewMode === 'chunks'
                    ? 'bg-accent-primary text-chonk-white'
                    : 'bg-surface-panel text-chonk-gray hover:text-chonk-light'
                }`}
                onClick={() => setViewMode('chunks')}
              >
                Chunks
              </button>
              <button
                className={`px-3 py-1 text-xs transition-colors ${
                  viewMode === 'document'
                    ? 'bg-accent-primary text-chonk-white'
                    : 'bg-surface-panel text-chonk-gray hover:text-chonk-light'
                }`}
                onClick={() => setViewMode('document')}
              >
                Blocks
              </button>
            </div>
            <span className="badge badge-info">
              {document.chunker_used}
            </span>
          </div>
        </div>
      </div>

      {/* Page navigation (if multi-page) */}
      {pageCount > 1 && (
        <div className="px-4 py-2 border-b border-chonk-slate/50 flex items-center justify-center gap-4">
          <button
            className="p-1 text-chonk-gray hover:text-chonk-white disabled:opacity-30"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
          >
            <ChevronLeft size={18} />
          </button>
          <span className="text-sm text-chonk-light">
            Page {currentPage} of {pageCount}
          </span>
          <button
            className="p-1 text-chonk-gray hover:text-chonk-white disabled:opacity-30"
            onClick={() => setCurrentPage((p) => Math.min(pageCount, p + 1))}
            disabled={currentPage === pageCount}
          >
            <ChevronRight size={18} />
          </button>
        </div>
      )}

      {/* Document content */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-3xl mx-auto space-y-4">
          {viewMode === 'chunks' ? (
            // Chunk view
            document.chunks.map((chunk, index) => (
              <ChunkCard
                key={chunk.id}
                chunk={chunk}
                index={index}
                isSelected={selectedChunkIds.includes(chunk.id)}
                isHovered={hoveredChunkId === chunk.id}
                onClick={() => selectChunk(chunk.id)}
                onCtrlClick={() => toggleChunkSelection(chunk.id)}
                onMouseEnter={() => setHoveredChunkId(chunk.id)}
                onMouseLeave={() => setHoveredChunkId(null)}
                blocks={document.blocks.filter((b) => chunk.block_ids.includes(b.id))}
              />
            ))
          ) : (
            // Block view - shows document structure
            document.blocks
              .filter((b) => pageCount <= 1 || b.page === currentPage)
              .map((block, index) => (
                <BlockCard key={block.id} block={block} index={index} />
              ))
          )}
        </div>
      </div>
    </div>
  );
}

interface ChunkCardProps {
  chunk: Chunk;
  index: number;
  isSelected: boolean;
  isHovered: boolean;
  onClick: () => void;
  onCtrlClick: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  blocks: Block[];
}

function ChunkCard({
  chunk,
  index,
  isSelected,
  isHovered,
  onClick,
  onCtrlClick,
  onMouseEnter,
  onMouseLeave,
  blocks,
}: ChunkCardProps) {
  const qualityColor =
    chunk.quality.overall >= 0.85
      ? 'border-accent-success'
      : chunk.quality.overall >= 0.7
      ? 'border-accent-warning'
      : 'border-accent-error';

  const qualityBg =
    chunk.quality.overall >= 0.85
      ? 'bg-accent-success/5'
      : chunk.quality.overall >= 0.7
      ? 'bg-accent-warning/5'
      : 'bg-accent-error/5';

  const handleClick = (e: React.MouseEvent) => {
    if (e.ctrlKey || e.metaKey) {
      onCtrlClick();
    } else {
      onClick();
    }
  };

  return (
    <div
      className={`
        relative rounded-lg border-2 transition-all cursor-pointer
        ${isSelected ? 'ring-2 ring-accent-primary ring-offset-2 ring-offset-surface-bg' : ''}
        ${isHovered && !isSelected ? 'border-accent-primary/50' : qualityColor}
        ${qualityBg}
      `}
      onClick={handleClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {/* Chunk number badge */}
      <div
        className={`
          absolute -top-3 -left-3 w-7 h-7 rounded-full flex items-center justify-center
          text-xs font-mono font-bold border-2
          ${isSelected ? 'bg-accent-primary border-accent-primary text-white' : 'bg-surface-panel border-chonk-slate text-chonk-light'}
        `}
      >
        {index + 1}
      </div>

      {/* Lock indicator */}
      {chunk.is_locked && (
        <div className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-accent-warning flex items-center justify-center">
          <span className="text-[10px]">🔒</span>
        </div>
      )}

      <div className="p-4 pt-5">
        {/* Chunk header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {chunk.hierarchy_path && (
              <span className="text-xs text-chonk-light bg-surface-panel px-2 py-0.5 rounded">
                {chunk.hierarchy_path}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-chonk-gray">
              {chunk.token_count} tokens
            </span>
            <span
              className={`font-mono font-bold ${
                chunk.quality.overall >= 0.85
                  ? 'text-accent-success'
                  : chunk.quality.overall >= 0.7
                  ? 'text-accent-warning'
                  : 'text-accent-error'
              }`}
            >
              {Math.round(chunk.quality.overall * 100)}%
            </span>
          </div>
        </div>

        {/* Block type indicators */}
        <div className="flex gap-1 mb-3">
          {getBlockTypeSummary(blocks).map(({ type, count }) => (
            <span
              key={type}
              className="text-[10px] px-1.5 py-0.5 rounded bg-surface-card text-chonk-gray"
            >
              {type}×{count}
            </span>
          ))}
        </div>

        {/* Chunk content with visual block separation */}
        <div className="space-y-2">
          {blocks.map((block) => (
            <div
              key={block.id}
              className={`
                text-sm leading-relaxed
                ${block.type === 'heading' ? 'font-semibold text-chonk-white' : 'text-chonk-light'}
                ${block.type === 'code' ? 'font-mono bg-surface-bg p-2 rounded text-xs' : ''}
                ${block.type === 'table' ? 'font-mono text-xs overflow-x-auto' : ''}
              `}
            >
              {block.type === 'heading' && block.heading_level && (
                <span className="text-xs text-chonk-gray mr-2">
                  H{block.heading_level}
                </span>
              )}
              <span className="whitespace-pre-wrap">{block.content}</span>
            </div>
          ))}
        </div>

        {/* Tags */}
        {chunk.user_metadata.tags.length > 0 && (
          <div className="flex gap-1 mt-3 pt-3 border-t border-chonk-slate/30">
            {chunk.user_metadata.tags.map((tag) => (
              <span
                key={tag}
                className="text-[10px] px-1.5 py-0.5 rounded bg-accent-primary/20 text-accent-primary"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Selection hint */}
      {isHovered && !isSelected && (
        <div className="absolute bottom-2 right-2 text-[10px] text-chonk-gray">
          Ctrl+click to multi-select
        </div>
      )}
    </div>
  );
}

function getBlockTypeSummary(blocks: Block[]): Array<{ type: string; count: number }> {
  const counts: Record<string, number> = {};
  for (const block of blocks) {
    counts[block.type] = (counts[block.type] || 0) + 1;
  }
  return Object.entries(counts)
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count);
}

interface BlockCardProps {
  block: Block;
  index: number;
}

function BlockCard({ block, index }: BlockCardProps) {
  const typeColors: Record<string, string> = {
    heading: 'border-accent-primary bg-accent-primary/10',
    text: 'border-chonk-gray bg-surface-panel',
    table: 'border-accent-warning bg-accent-warning/10',
    code: 'border-chonk-teal bg-chonk-teal/10',
    list: 'border-chonk-green bg-chonk-green/10',
    list_item: 'border-chonk-green/50 bg-chonk-green/5',
    image: 'border-chonk-purple bg-chonk-purple/10',
  };

  const colorClass = typeColors[block.type] || typeColors.text;

  return (
    <div className={`rounded border-l-4 p-3 ${colorClass}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-chonk-gray">
            #{index + 1}
          </span>
          <span className="text-xs font-medium text-chonk-light uppercase">
            {block.type}
            {block.heading_level && ` ${block.heading_level}`}
          </span>
        </div>
        {block.bbox && (
          <span className="text-[10px] text-chonk-gray">
            p.{block.page}
          </span>
        )}
      </div>
      <div
        className={`
          text-sm whitespace-pre-wrap
          ${block.type === 'heading' ? 'font-semibold' : ''}
          ${block.type === 'code' ? 'font-mono text-xs' : ''}
        `}
      >
        {block.content.length > 500
          ? block.content.slice(0, 500) + '...'
          : block.content}
      </div>
    </div>
  );
}
