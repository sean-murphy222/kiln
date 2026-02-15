/**
 * SideBySide - Compare two chunking strategies side-by-side
 *
 * This is a killer feature - showing users concrete differences
 * before they commit to a strategy.
 */

import React from 'react';
import type { ChunkingStrategy } from '../StrategySelector';

export interface StrategyResult {
  strategy_name: string;
  chunks_count: number;
  avg_tokens: number;
  min_tokens: number;
  max_tokens: number;
  avg_quality_score: number;
  hierarchy_preservation: number;
  chunks_with_context: number;
  processing_time_ms: number;
}

interface SideBySideProps {
  results: StrategyResult[];
  onSelectStrategy?: (strategy: string) => void;
}

export const SideBySide: React.FC<SideBySideProps> = ({
  results,
  onSelectStrategy,
}) => {
  if (results.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <p>Run comparison to see results</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {results.map((result) => (
        <StrategyResultCard
          key={result.strategy_name}
          result={result}
          onSelect={() => onSelectStrategy?.(result.strategy_name)}
        />
      ))}
    </div>
  );
};

interface StrategyResultCardProps {
  result: StrategyResult;
  onSelect: () => void;
}

const StrategyResultCard: React.FC<StrategyResultCardProps> = ({
  result,
  onSelect,
}) => {
  // Determine quality badge
  const qualityBadge =
    result.avg_quality_score >= 0.9
      ? { label: 'Excellent', color: 'bg-green-600' }
      : result.avg_quality_score >= 0.7
      ? { label: 'Good', color: 'bg-blue-600' }
      : result.avg_quality_score >= 0.5
      ? { label: 'Fair', color: 'bg-yellow-600' }
      : { label: 'Poor', color: 'bg-red-600' };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 hover:border-blue-500 transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold capitalize">
          {result.strategy_name}
        </h3>
        <span
          className={`px-3 py-1 text-xs font-bold rounded ${qualityBadge.color}`}
        >
          {qualityBadge.label}
        </span>
      </div>

      {/* Metrics Grid */}
      <div className="space-y-3">
        {/* Total Chunks */}
        <MetricRow
          label="Total Chunks"
          value={result.chunks_count.toLocaleString()}
          icon="📦"
        />

        {/* Avg Tokens */}
        <MetricRow
          label="Avg Tokens"
          value={result.avg_tokens.toFixed(1)}
          subtext={`${result.min_tokens} - ${result.max_tokens}`}
          icon="📝"
        />

        {/* Quality Score */}
        <MetricRow
          label="Quality Score"
          value={result.avg_quality_score.toFixed(3)}
          progress={result.avg_quality_score}
          icon="⭐"
        />

        {/* Hierarchy Preservation */}
        <MetricRow
          label="Hierarchy"
          value={`${(result.hierarchy_preservation * 100).toFixed(0)}%`}
          progress={result.hierarchy_preservation}
          icon="🌳"
          badge={
            result.hierarchy_preservation > 0.8 ? (
              <span className="text-xs text-green-400">✓ Preserved</span>
            ) : (
              <span className="text-xs text-red-400">✗ Lost</span>
            )
          }
        />

        {/* Context */}
        <MetricRow
          label="With Context"
          value={`${result.chunks_with_context}/${result.chunks_count}`}
          icon="📍"
        />

        {/* Processing Time */}
        <MetricRow
          label="Processing Time"
          value={`${result.processing_time_ms}ms`}
          icon="⏱️"
        />
      </div>

      {/* Select Button */}
      <button
        onClick={onSelect}
        className="w-full mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors"
      >
        Use This Strategy
      </button>
    </div>
  );
};

interface MetricRowProps {
  label: string;
  value: string | number;
  subtext?: string;
  progress?: number;
  icon?: string;
  badge?: React.ReactNode;
}

const MetricRow: React.FC<MetricRowProps> = ({
  label,
  value,
  subtext,
  progress,
  icon,
  badge,
}) => {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          {icon && <span>{icon}</span>}
          <span className="text-sm text-gray-400">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold">{value}</span>
          {badge}
        </div>
      </div>

      {subtext && (
        <div className="text-xs text-gray-500 ml-6">{subtext}</div>
      )}

      {progress !== undefined && (
        <div className="ml-6 mt-1">
          <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                progress > 0.8
                  ? 'bg-green-500'
                  : progress > 0.5
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
