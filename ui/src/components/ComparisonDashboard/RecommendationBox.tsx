/**
 * RecommendationBox - Shows recommended strategy based on comparison
 */

import React from 'react';
import type { StrategyResult } from './SideBySide';

interface RecommendationBoxProps {
  results: StrategyResult[];
  recommendation: string;
}

export const RecommendationBox: React.FC<RecommendationBoxProps> = ({
  results,
  recommendation,
}) => {
  if (results.length === 0) return null;

  // Find best strategy
  const best = results.reduce((prev, current) =>
    current.avg_quality_score > prev.avg_quality_score ? current : prev
  );

  return (
    <div className="bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg p-6 border border-blue-500">
      <div className="flex items-start gap-4">
        <div className="text-4xl">💡</div>
        <div className="flex-1">
          <h3 className="text-xl font-bold mb-2">Recommendation</h3>
          <div className="space-y-2 text-sm">
            <p className="text-lg font-medium">
              ✅ Use <span className="text-blue-300 font-bold capitalize">{best.strategy_name}</span> strategy
            </p>
            <div className="text-gray-300">
              <p className="font-medium mb-1">Why?</p>
              <ul className="space-y-1 ml-4">
                {best.hierarchy_preservation > 0.8 && (
                  <li>• Preserves document structure ({(best.hierarchy_preservation * 100).toFixed(0)}% hierarchy)</li>
                )}
                {best.avg_quality_score > 0.9 && (
                  <li>• High quality chunks (score: {best.avg_quality_score.toFixed(2)})</li>
                )}
                {best.avg_tokens >= 300 && best.avg_tokens <= 600 && (
                  <li>• Optimal token size ({best.avg_tokens.toFixed(0)} tokens avg)</li>
                )}
                {best.chunks_with_context / best.chunks_count > 0.9 && (
                  <li>• Includes hierarchy context for {best.chunks_with_context} chunks</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
