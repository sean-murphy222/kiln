/**
 * TreeView - The centerpiece of CHONK
 *
 * Visual document structure explorer showing hierarchy tree.
 * This is what makes CHONK different from other chunkers.
 */

import React, { useState } from 'react';
import { TreeNode } from './TreeNode';
import { TreeStats } from './TreeStats';

export interface HierarchyNode {
  section_id: string;
  heading: string | null;
  heading_level: number;
  content: string;
  token_count: number;
  page_range: number[];
  hierarchy_path: string;
  depth: number;
  is_leaf: boolean;
  child_count: number;
  children: HierarchyNode[];
  heading_block_id: string | null;
  content_block_ids: string[];
}

export interface HierarchyTree {
  document_id: string;
  statistics: {
    total_nodes: number;
    nodes_with_content: number;
    nodes_with_children: number;
    leaf_nodes: number;
    max_depth: number;
    avg_tokens_per_node: number;
    min_tokens: number;
    max_tokens: number;
    level_distribution: Record<number, number>;
  };
  root: HierarchyNode;
}

interface TreeViewProps {
  tree: HierarchyTree | null;
  onNodeSelect?: (node: HierarchyNode) => void;
  selectedNodeId?: string;
  className?: string;
}

export const TreeView: React.FC<TreeViewProps> = ({
  tree,
  onNodeSelect,
  selectedNodeId,
  className = '',
}) => {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');

  if (!tree) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center p-8">
          <div className="text-6xl mb-4">🌳</div>
          <h3 className="text-xl font-bold mb-2">No Document Loaded</h3>
          <p className="text-gray-400">
            Upload a document to see its structure
          </p>
        </div>
      </div>
    );
  }

  const toggleNode = (nodeId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  const expandAll = () => {
    const allIds = new Set<string>();
    const collectIds = (node: HierarchyNode) => {
      if (node.section_id !== 'root') {
        allIds.add(node.section_id);
      }
      node.children.forEach(collectIds);
    };
    tree.root.children.forEach(collectIds);
    setExpandedNodes(allIds);
  };

  const collapseAll = () => {
    setExpandedNodes(new Set());
  };

  // Filter nodes based on search
  const matchesSearch = (node: HierarchyNode): boolean => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      node.heading?.toLowerCase().includes(query) ||
      node.section_id.toLowerCase().includes(query) ||
      node.content.toLowerCase().includes(query)
    );
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700 bg-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-bold">📊 Document Structure</h2>
          <div className="flex gap-2">
            <button
              onClick={expandAll}
              className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-700 rounded"
            >
              Expand All
            </button>
            <button
              onClick={collapseAll}
              className="px-3 py-1 text-sm bg-gray-600 hover:bg-gray-700 rounded"
            >
              Collapse All
            </button>
          </div>
        </div>

        {/* Search */}
        <input
          type="text"
          placeholder="🔍 Search sections..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Stats Summary */}
      <TreeStats tree={tree} />

      {/* Tree */}
      <div className="flex-1 overflow-y-auto p-4">
        {tree.root.children.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <p>No sections found in document</p>
            <p className="text-sm mt-2">
              The document may not have clear heading structure
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            {tree.root.children.map((node) => (
              <TreeNode
                key={node.section_id}
                node={node}
                depth={0}
                isExpanded={expandedNodes.has(node.section_id)}
                onToggle={() => toggleNode(node.section_id)}
                onSelect={onNodeSelect}
                isSelected={selectedNodeId === node.section_id}
                matchesSearch={matchesSearch}
                searchQuery={searchQuery}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
