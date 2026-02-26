/**
 * Chunk Tree View - Displays chunks in hierarchical structure
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText, Layers } from 'lucide-react';
import type { Chunk } from '../api/chonk';

interface ChunkNode {
  path: string;
  level: number;
  chunks: Chunk[];
  children: Map<string, ChunkNode>;
}

interface ChunkTreeViewProps {
  chunks: Chunk[];
  onSelectChunk: (chunk: Chunk) => void;
  selectedChunkId?: string;
}

function buildTree(chunks: Chunk[]): ChunkNode {
  const root: ChunkNode = {
    path: '',
    level: 0,
    chunks: [],
    children: new Map(),
  };

  // Group chunks by hierarchy path
  for (const chunk of chunks) {
    if (!chunk.hierarchy_path) {
      // No hierarchy - add to root
      root.chunks.push(chunk);
      continue;
    }

    // Split path into parts (e.g., "Section 2 > 2.1 General > 2.1.1 Safety")
    const parts = chunk.hierarchy_path.split(' > ').filter(p => p.trim());
    let current = root;

    // Build tree structure
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i].trim();
      const fullPath = parts.slice(0, i + 1).join(' > ');

      if (!current.children.has(part)) {
        current.children.set(part, {
          path: fullPath,
          level: i + 1,
          chunks: [],
          children: new Map(),
        });
      }

      current = current.children.get(part)!;
    }

    // Add chunk to the deepest level
    current.chunks.push(chunk);
  }

  return root;
}

function TreeNode({
  node,
  label,
  onSelectChunk,
  selectedChunkId,
  defaultExpanded = false,
}: {
  node: ChunkNode;
  label: string;
  onSelectChunk: (chunk: Chunk) => void;
  selectedChunkId?: string;
  defaultExpanded?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded || node.level === 0 || node.level === 1);

  const hasChildren = node.children.size > 0;
  const hasChunks = node.chunks.length > 0;
  const totalChunks = node.chunks.length +
    Array.from(node.children.values()).reduce((sum, child) => sum + child.chunks.length, 0);

  // Calculate indent based on level
  const indent = node.level * 16;

  return (
    <div>
      {/* Section Header (if not root) */}
      {label && (
        <div
          className="flex items-center gap-2 py-2 px-2 cursor-pointer hover:bg-kiln-900 transition-colors"
          style={{ paddingLeft: `${indent}px` }}
          onClick={() => setExpanded(!expanded)}
        >
          {hasChildren ? (
            expanded ? (
              <ChevronDown size={16} className="text-ember flex-shrink-0" />
            ) : (
              <ChevronRight size={16} className="text-kiln-500 flex-shrink-0" />
            )
          ) : (
            <div className="w-4" />
          )}

          <Layers size={14} className="text-ember flex-shrink-0" />

          <span className="text-sm font-bold text-kiln-300 flex-1 min-w-0">
            {label}
          </span>

          <span className="text-xs text-kiln-500 flex-shrink-0">
            {totalChunks} {totalChunks === 1 ? 'chunk' : 'chunks'}
          </span>
        </div>
      )}

      {/* Expanded Content */}
      {expanded && (
        <div>
          {/* Direct chunks at this level */}
          {hasChunks && node.chunks.map((chunk, index) => {
            const isSelected = selectedChunkId === chunk.id;
            return (
              <div
                key={chunk.id}
                onClick={() => onSelectChunk(chunk)}
                className={`
                  flex items-start gap-2 py-2 px-2 cursor-pointer transition-colors
                  ${isSelected
                    ? 'bg-ember/10 border-l-4 border-ember'
                    : 'hover:bg-kiln-900 border-l-4 border-transparent'
                  }
                `}
                style={{ paddingLeft: `${indent + 24}px` }}
              >
                <FileText size={14} className="text-kiln-500 flex-shrink-0 mt-0.5" />

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-bold text-ember">
                      Chunk {index + 1}
                    </span>
                    <span className="text-xs text-kiln-500">
                      {chunk.token_count} tokens
                    </span>
                  </div>
                  <p className="text-xs text-kiln-300 line-clamp-2">
                    {chunk.content}
                  </p>
                </div>
              </div>
            );
          })}

          {/* Child sections */}
          {Array.from(node.children.entries()).map(([childLabel, childNode]) => (
            <TreeNode
              key={childLabel}
              node={childNode}
              label={childLabel}
              onSelectChunk={onSelectChunk}
              selectedChunkId={selectedChunkId}
              defaultExpanded={childNode.level <= 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function ChunkTreeView({ chunks, onSelectChunk, selectedChunkId }: ChunkTreeViewProps) {
  const tree = buildTree(chunks);

  const totalTokens = chunks.reduce((sum, c) => sum + c.token_count, 0);
  const avgTokens = Math.round(totalTokens / chunks.length);
  const hasHierarchy = chunks.some(c => c.hierarchy_path);

  return (
    <div className="h-full flex flex-col bg-kiln-900">
      {/* Header */}
      <div className="border-b-2 border-black p-4 bg-kiln-800">
        <h2 className="text-pixel text-lg text-kiln-300 mb-3">
          {hasHierarchy ? 'DOCUMENT STRUCTURE' : 'DOCUMENT CHUNKS'}
        </h2>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-3">
          <div className="bg-kiln-900 border border-black p-2">
            <div className="text-xs text-kiln-500 mb-1">Total Chunks</div>
            <div className="text-pixel text-xl text-kiln-300">{chunks.length}</div>
          </div>

          <div className="bg-kiln-900 border border-black p-2">
            <div className="text-xs text-kiln-500 mb-1">Avg Tokens</div>
            <div className="text-pixel text-xl text-kiln-300">{avgTokens}</div>
          </div>

          <div className="bg-kiln-900 border border-black p-2">
            <div className="text-xs text-kiln-500 mb-1">Total Tokens</div>
            <div className="text-pixel text-xl text-kiln-300">{totalTokens.toLocaleString()}</div>
          </div>
        </div>

        {hasHierarchy ? (
          <div className="bg-blue-900/20 border border-blue-400 p-2">
            <p className="text-xs text-blue-300">
              <strong>Hierarchical view:</strong> Chunks are organized by document structure.
              Click sections to expand/collapse. Click chunks to view details.
            </p>
          </div>
        ) : (
          <div className="bg-yellow-900/20 border border-yellow-400 p-2">
            <p className="text-xs text-yellow-300">
              <strong>Flat view:</strong> No hierarchy detected. Click chunks to view details.
            </p>
          </div>
        )}
      </div>

      {/* Tree View */}
      <div className="flex-1 overflow-y-auto">
        {hasHierarchy ? (
          <TreeNode
            node={tree}
            label=""
            onSelectChunk={onSelectChunk}
            selectedChunkId={selectedChunkId}
            defaultExpanded={true}
          />
        ) : (
          // Fallback to flat list if no hierarchy
          <div className="p-4 space-y-2">
            {chunks.map((chunk, index) => {
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
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-ember">
                      Chunk {index + 1}
                    </span>
                    <span className="text-xs text-kiln-500">
                      {chunk.token_count} tokens
                    </span>
                  </div>
                  <p className="text-sm text-kiln-300 line-clamp-3">
                    {chunk.content}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
