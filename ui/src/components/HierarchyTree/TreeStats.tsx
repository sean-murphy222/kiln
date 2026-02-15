/**
 * TreeStats - Statistics panel for hierarchy tree
 *
 * Shows key metrics about document structure.
 */

import React from 'react';
import type { HierarchyTree } from './TreeView';

interface TreeStatsProps {
  tree: HierarchyTree;
}

export const TreeStats: React.FC<TreeStatsProps> = ({ tree }) => {
  const { statistics: stats } = tree;

  return (
    <div className="px-4 py-3 bg-gray-800 border-b border-gray-700">
      <div className="grid grid-cols-2 gap-2 text-sm">
        {/* Total Sections */}
        <div className="flex justify-between">
          <span className="text-gray-400">Sections:</span>
          <span className="font-mono font-bold">{stats.total_nodes}</span>
        </div>

        {/* Max Depth */}
        <div className="flex justify-between">
          <span className="text-gray-400">Depth:</span>
          <span className="font-mono font-bold">{stats.max_depth}</span>
        </div>

        {/* Avg Tokens */}
        <div className="flex justify-between">
          <span className="text-gray-400">Avg Tokens:</span>
          <span className="font-mono font-bold">
            {stats.avg_tokens_per_node.toFixed(0)}
          </span>
        </div>

        {/* Leaf Nodes */}
        <div className="flex justify-between">
          <span className="text-gray-400">Leaves:</span>
          <span className="font-mono font-bold">{stats.leaf_nodes}</span>
        </div>
      </div>

      {/* Quality Indicator */}
      <div className="mt-3 pt-3 border-t border-gray-700">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Structure Quality:</span>
          <StructureQualityBadge stats={stats} />
        </div>
      </div>
    </div>
  );
};

// Quality badge based on structure metrics
const StructureQualityBadge: React.FC<{ stats: HierarchyTree['statistics'] }> = ({
  stats,
}) => {
  // Simple quality heuristic
  const hasGoodDepth = stats.max_depth >= 2 && stats.max_depth <= 6;
  const hasReasonableSize =
    stats.avg_tokens_per_node >= 50 && stats.avg_tokens_per_node <= 800;
  const hasGoodContent =
    stats.nodes_with_content / stats.total_nodes > 0.7;

  const score =
    (hasGoodDepth ? 1 : 0) +
    (hasReasonableSize ? 1 : 0) +
    (hasGoodContent ? 1 : 0);

  const quality =
    score === 3
      ? { label: 'Excellent', color: 'bg-green-600' }
      : score === 2
      ? { label: 'Good', color: 'bg-blue-600' }
      : score === 1
      ? { label: 'Fair', color: 'bg-yellow-600' }
      : { label: 'Poor', color: 'bg-red-600' };

  return (
    <span
      className={`px-2 py-1 text-xs font-bold rounded ${quality.color}`}
      title={getQualityTooltip(stats, hasGoodDepth, hasReasonableSize, hasGoodContent)}
    >
      {quality.label}
    </span>
  );
};

function getQualityTooltip(
  stats: HierarchyTree['statistics'],
  hasGoodDepth: boolean,
  hasReasonableSize: boolean,
  hasGoodContent: boolean
): string {
  const issues = [];
  if (!hasGoodDepth) {
    issues.push(
      stats.max_depth === 1
        ? 'Very flat structure (only 1 level)'
        : `Very deep structure (${stats.max_depth} levels)`
    );
  }
  if (!hasReasonableSize) {
    issues.push(
      stats.avg_tokens_per_node < 50
        ? 'Sections are very small'
        : 'Sections are very large'
    );
  }
  if (!hasGoodContent) {
    const ratio = ((stats.nodes_with_content / stats.total_nodes) * 100).toFixed(0);
    issues.push(`Only ${ratio}% of sections have content`);
  }

  return issues.length > 0
    ? issues.join('. ')
    : 'Good hierarchy for chunking!';
}
