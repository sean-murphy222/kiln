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
      <div className="h-full flex flex-col items-center justify-center text-kiln-500">
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
      <div className="px-4 py-3 border-b border-kiln-600 bg-kiln-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-medium text-kiln-100">
              {document.source_path.split(/[/\\]/).pop()}
            </h2>
            <p className="text-xs text-kiln-500 mt-0.5">
              {document.metadata.page_count > 0 && `${document.metadata.page_count} pages · `}
              {document.metadata.word_count.toLocaleString()} words · {document.chunks.length} chunks
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* View mode toggle */}
            <div className="flex rounded overflow-hidden border border-kiln-600">
              <button
                className={`px-3 py-1 text-xs transition-colors ${
                  viewMode === 'chunks'
                    ? 'bg-ember text-kiln-100'
                    : 'bg-kiln-800 text-kiln-500 hover:text-kiln-300'
                }`}
                onClick={() => setViewMode('chunks')}
              >
                Chunks
              </button>
              <button
                className={`px-3 py-1 text-xs transition-colors ${
                  viewMode === 'document'
                    ? 'bg-ember text-kiln-100'
                    : 'bg-kiln-800 text-kiln-500 hover:text-kiln-300'
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
        <div className="px-4 py-2 border-b border-kiln-600/50 flex items-center justify-center gap-4">
          <button
            className="p-1 text-kiln-500 hover:text-kiln-100 disabled:opacity-30"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
          >
            <ChevronLeft size={18} />
          </button>
          <span className="text-sm text-kiln-300">
            Page {currentPage} of {pageCount}
          </span>
          <button
            className="p-1 text-kiln-500 hover:text-kiln-100 disabled:opacity-30"
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
      ? 'border-success'
      : chunk.quality.overall >= 0.7
      ? 'border-warning'
      : 'border-error';

  const qualityBg =
    chunk.quality.overall >= 0.85
      ? 'bg-success/5'
      : chunk.quality.overall >= 0.7
      ? 'bg-warning/5'
      : 'bg-error/5';

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
        ${isSelected ? 'ring-2 ring-ember ring-offset-2 ring-offset-kiln-900' : ''}
        ${isHovered && !isSelected ? 'border-ember/50' : qualityColor}
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
          ${isSelected ? 'bg-ember border-ember text-white' : 'bg-kiln-800 border-kiln-600 text-kiln-300'}
        `}
      >
        {index + 1}
      </div>

      {/* Lock indicator */}
      {chunk.is_locked && (
        <div className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-warning flex items-center justify-center">
          <span className="text-[10px]">🔒</span>
        </div>
      )}

      <div className="p-4 pt-5">
        {/* Chunk header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {chunk.hierarchy_path && (
              <span className="text-xs text-kiln-300 bg-kiln-800 px-2 py-0.5 rounded">
                {chunk.hierarchy_path}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-kiln-500">
              {chunk.token_count} tokens
            </span>
            <span
              className={`font-mono font-bold ${
                chunk.quality.overall >= 0.85
                  ? 'text-success'
                  : chunk.quality.overall >= 0.7
                  ? 'text-warning'
                  : 'text-error'
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
              className="text-[10px] px-1.5 py-0.5 rounded bg-kiln-700 text-kiln-500"
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
                ${block.type === 'heading' ? 'font-semibold text-kiln-100' : 'text-kiln-300'}
                ${block.type === 'code' ? 'font-mono bg-kiln-900 p-2 rounded text-xs' : ''}
                ${block.type === 'table' ? 'font-mono text-xs overflow-x-auto' : ''}
              `}
            >
              {block.type === 'heading' && block.heading_level && (
                <span className="text-xs text-kiln-500 mr-2">
                  H{block.heading_level}
                </span>
              )}
              <span className="whitespace-pre-wrap">{block.content}</span>
            </div>
          ))}
        </div>

        {/* Tags */}
        {chunk.user_metadata.tags.length > 0 && (
          <div className="flex gap-1 mt-3 pt-3 border-t border-kiln-600/30">
            {chunk.user_metadata.tags.map((tag) => (
              <span
                key={tag}
                className="text-[10px] px-1.5 py-0.5 rounded bg-ember/20 text-ember"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Selection hint */}
      {isHovered && !isSelected && (
        <div className="absolute bottom-2 right-2 text-[10px] text-kiln-500">
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
    heading: 'border-ember bg-ember/10',
    text: 'border-kiln-500 bg-kiln-800',
    table: 'border-warning bg-warning/10',
    code: 'border-foundry-cast bg-foundry-cast/10',
    list: 'border-success bg-success/10',
    list_item: 'border-success/50 bg-success/5',
    image: 'border-forge-heat bg-forge-heat/10',
  };

  const colorClass = typeColors[block.type] || typeColors.text;

  return (
    <div className={`rounded border-l-4 p-3 ${colorClass}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-kiln-500">
            #{index + 1}
          </span>
          <span className="text-xs font-medium text-kiln-300 uppercase">
            {block.type}
            {block.heading_level && ` ${block.heading_level}`}
          </span>
        </div>
        {block.bbox && (
          <span className="text-[10px] text-kiln-500">
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
