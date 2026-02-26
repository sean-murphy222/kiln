/**
 * Document Overview - Shows uploaded document chunks before diagnostics
 */

import { FileText, Hash, Layers } from 'lucide-react';
import type { Document, Chunk } from '../api/chonk';

interface DocumentOverviewProps {
  document: Document;
  onSelectChunk: (chunk: Chunk) => void;
  selectedChunkId?: string;
}

export function DocumentOverview({ document, onSelectChunk, selectedChunkId }: DocumentOverviewProps) {
  const totalTokens = document.chunks.reduce((sum, c) => sum + c.token_count, 0);
  const avgTokens = Math.round(totalTokens / document.chunks.length);

  return (
    <div className="h-full flex flex-col bg-kiln-900">
      {/* Header */}
      <div className="border-b-2 border-black p-4 bg-kiln-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-pixel text-lg text-kiln-300">DOCUMENT CHUNKS</h2>
          <span className="text-xs text-kiln-500">
            Click any chunk to view details
          </span>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-kiln-900 border border-black p-2">
            <div className="flex items-center gap-2 mb-1">
              <Layers size={14} className="text-ember" />
              <span className="text-xs text-kiln-500">Total Chunks</span>
            </div>
            <div className="text-pixel text-xl text-kiln-300">{document.chunks.length}</div>
          </div>

          <div className="bg-kiln-900 border border-black p-2">
            <div className="flex items-center gap-2 mb-1">
              <Hash size={14} className="text-ember" />
              <span className="text-xs text-kiln-500">Avg Tokens</span>
            </div>
            <div className="text-pixel text-xl text-kiln-300">{avgTokens}</div>
          </div>

          <div className="bg-kiln-900 border border-black p-2">
            <div className="flex items-center gap-2 mb-1">
              <FileText size={14} className="text-ember" />
              <span className="text-xs text-kiln-500">Total Tokens</span>
            </div>
            <div className="text-pixel text-xl text-kiln-300">{totalTokens.toLocaleString()}</div>
          </div>
        </div>

        <div className="mt-3 bg-blue-900/20 border border-blue-400 p-2">
          <p className="text-xs text-blue-300">
            <strong>Next Step:</strong> Click "RUN DIAGNOSTICS" above to analyze these chunks for problems
          </p>
        </div>
      </div>

      {/* Chunk List */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          {document.chunks.map((chunk, index) => {
            const isSelected = selectedChunkId === chunk.id;
            return (
            <div
              key={chunk.id}
              onClick={() => onSelectChunk(chunk)}
              className={`
                border-2 p-3 cursor-pointer transition-colors
                ${isSelected
                  ? 'border-ember bg-ember/10'
                  : 'border-black bg-kiln-800 hover:bg-kiln-900'
                }
              `}
            >
              {/* Chunk Header */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-ember">
                    Chunk {index + 1}
                  </span>
                  {chunk.hierarchy_path && (
                    <span className="text-xs text-kiln-500 truncate max-w-xs">
                      {chunk.hierarchy_path}
                    </span>
                  )}
                </div>
                <span className="text-xs text-kiln-500">
                  {chunk.token_count} tokens
                </span>
              </div>

              {/* Content Preview */}
              <p className="text-sm text-kiln-300 line-clamp-3">
                {chunk.content}
              </p>

              {/* Tags */}
              {chunk.user_metadata.tags.length > 0 && (
                <div className="flex gap-1 mt-2">
                  {chunk.user_metadata.tags.map((tag, i) => (
                    <span
                      key={i}
                      className="text-xs px-2 py-0.5 bg-ember/20 border border-ember text-ember"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
          })}
        </div>
      </div>
    </div>
  );
}
