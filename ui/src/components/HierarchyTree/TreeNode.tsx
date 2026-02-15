/**
 * TreeNode - Individual node in the hierarchy tree
 *
 * Renders a single section with expand/collapse, metadata, and children.
 */

import React from 'react';
import type { HierarchyNode } from './TreeView';

interface TreeNodeProps {
  node: HierarchyNode;
  depth: number;
  isExpanded: boolean;
  onToggle: () => void;
  onSelect?: (node: HierarchyNode) => void;
  isSelected: boolean;
  matchesSearch: (node: HierarchyNode) => boolean;
  searchQuery: string;
}

export const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  depth,
  isExpanded,
  onToggle,
  onSelect,
  isSelected,
  matchesSearch,
  searchQuery,
}) => {
  const hasChildren = node.children.length > 0;
  const matches = matchesSearch(node);

  // Skip nodes that don't match search
  if (searchQuery && !matches && !hasChildrenMatching(node, matchesSearch)) {
    return null;
  }

  const handleClick = () => {
    if (hasChildren) {
      onToggle();
    }
    onSelect?.(node);
  };

  // Indent based on depth
  const indentClass = `pl-${depth * 4}`;

  // Icon based on state
  const icon = hasChildren ? (isExpanded ? '▼' : '▶') : '📄';

  // Highlight search matches
  const highlightText = (text: string) => {
    if (!searchQuery) return text;
    const parts = text.split(new RegExp(`(${searchQuery})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === searchQuery.toLowerCase() ? (
        <span key={i} className="bg-yellow-500 text-black">
          {part}
        </span>
      ) : (
        part
      )
    );
  };

  return (
    <div>
      {/* Node */}
      <div
        className={`
          group flex items-start gap-2 px-3 py-2 rounded cursor-pointer
          hover:bg-gray-700 transition-colors
          ${isSelected ? 'bg-blue-600' : ''}
          ${indentClass}
        `}
        onClick={handleClick}
      >
        {/* Expand/Collapse Icon */}
        <span className="text-sm flex-shrink-0 w-4">{icon}</span>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Heading */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium truncate">
              {node.heading ? highlightText(node.heading) : '(No heading)'}
            </span>

            {/* Badges */}
            <div className="flex gap-1 flex-shrink-0">
              {/* Level badge */}
              <span className="px-1.5 py-0.5 text-xs bg-gray-600 rounded">
                L{node.heading_level}
              </span>

              {/* Token count badge */}
              <span
                className={`px-1.5 py-0.5 text-xs rounded ${
                  node.token_count > 500
                    ? 'bg-red-600'
                    : node.token_count > 300
                    ? 'bg-yellow-600'
                    : 'bg-green-600'
                }`}
              >
                {node.token_count} tok
              </span>

              {/* Page range */}
              {node.page_range[0] > 0 && (
                <span className="px-1.5 py-0.5 text-xs bg-gray-600 rounded">
                  p{node.page_range[0]}
                  {node.page_range[1] !== node.page_range[0] &&
                    `-${node.page_range[1]}`}
                </span>
              )}

              {/* Children count */}
              {hasChildren && (
                <span className="px-1.5 py-0.5 text-xs bg-blue-600 rounded">
                  {node.child_count} sub
                </span>
              )}
            </div>
          </div>

          {/* Content preview (only show when not expanded or no children) */}
          {node.content && (!hasChildren || !isExpanded) && (
            <div className="text-sm text-gray-400 mt-1 truncate">
              {highlightText(node.content.slice(0, 100))}
              {node.content.length > 100 && '...'}
            </div>
          )}

          {/* Section ID (subtle) */}
          <div className="text-xs text-gray-500 mt-1 font-mono">
            {node.section_id}
          </div>
        </div>
      </div>

      {/* Children (recursive) */}
      {isExpanded && hasChildren && (
        <div className="mt-1">
          {node.children.map((child) => (
            <TreeNode
              key={child.section_id}
              node={child}
              depth={depth + 1}
              isExpanded={isExpanded}
              onToggle={onToggle}
              onSelect={onSelect}
              isSelected={isSelected}
              matchesSearch={matchesSearch}
              searchQuery={searchQuery}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Helper function to check if any children match search
function hasChildrenMatching(
  node: HierarchyNode,
  matchesSearch: (node: HierarchyNode) => boolean
): boolean {
  if (matchesSearch(node)) return true;
  return node.children.some((child) => hasChildrenMatching(child, matchesSearch));
}
