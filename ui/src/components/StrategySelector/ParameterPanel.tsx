/**
 * ParameterPanel - Configure strategy parameters
 *
 * Allows users to tweak chunking parameters.
 */

import React from 'react';
import type { ChunkingStrategy } from './StrategyPicker';

export interface ChunkingParameters {
  max_tokens: number;
  overlap_tokens: number;
  preserve_tables: boolean;
  preserve_code: boolean;
  group_under_headings: boolean;
  heading_weight: number;
}

interface ParameterPanelProps {
  strategy: ChunkingStrategy;
  parameters: ChunkingParameters;
  onChange: (parameters: ChunkingParameters) => void;
}

export const ParameterPanel: React.FC<ParameterPanelProps> = ({
  strategy,
  parameters,
  onChange,
}) => {
  const updateParam = <K extends keyof ChunkingParameters>(
    key: K,
    value: ChunkingParameters[K]
  ) => {
    onChange({ ...parameters, [key]: value });
  };

  // Different parameters shown for different strategies
  const showGroupUnderHeadings = strategy === 'hierarchical';
  const showHeadingWeight = strategy === 'hierarchical';

  return (
    <div className="space-y-4 p-4 bg-gray-800 rounded-lg">
      <h4 className="font-bold mb-3">Parameters</h4>

      {/* Max Tokens */}
      <div>
        <label className="block text-sm font-medium mb-2">
          Max Tokens per Chunk
        </label>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min="100"
            max="2000"
            step="50"
            value={parameters.max_tokens}
            onChange={(e) => updateParam('max_tokens', parseInt(e.target.value))}
            className="flex-1"
          />
          <input
            type="number"
            min="100"
            max="2000"
            value={parameters.max_tokens}
            onChange={(e) => updateParam('max_tokens', parseInt(e.target.value))}
            className="w-20 px-2 py-1 bg-gray-900 border border-gray-600 rounded text-sm"
          />
        </div>
        <p className="text-xs text-gray-400 mt-1">
          Recommended: 400-600 tokens for most use cases
        </p>
      </div>

      {/* Overlap Tokens */}
      <div>
        <label className="block text-sm font-medium mb-2">
          Overlap Tokens
        </label>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min="0"
            max="200"
            step="10"
            value={parameters.overlap_tokens}
            onChange={(e) => updateParam('overlap_tokens', parseInt(e.target.value))}
            className="flex-1"
          />
          <input
            type="number"
            min="0"
            max="200"
            value={parameters.overlap_tokens}
            onChange={(e) => updateParam('overlap_tokens', parseInt(e.target.value))}
            className="w-20 px-2 py-1 bg-gray-900 border border-gray-600 rounded text-sm"
          />
        </div>
        <p className="text-xs text-gray-400 mt-1">
          Overlap between chunks for context continuity
        </p>
      </div>

      {/* Preserve Tables */}
      <div className="flex items-center justify-between">
        <div>
          <label className="block text-sm font-medium">Preserve Tables</label>
          <p className="text-xs text-gray-400">
            Keep tables as complete units
          </p>
        </div>
        <button
          onClick={() => updateParam('preserve_tables', !parameters.preserve_tables)}
          className={`
            relative w-12 h-6 rounded-full transition-colors
            ${parameters.preserve_tables ? 'bg-blue-500' : 'bg-gray-600'}
          `}
        >
          <div
            className={`
              absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform
              ${parameters.preserve_tables ? 'translate-x-6' : 'translate-x-0.5'}
            `}
          />
        </button>
      </div>

      {/* Preserve Code */}
      <div className="flex items-center justify-between">
        <div>
          <label className="block text-sm font-medium">Preserve Code</label>
          <p className="text-xs text-gray-400">
            Keep code blocks intact
          </p>
        </div>
        <button
          onClick={() => updateParam('preserve_code', !parameters.preserve_code)}
          className={`
            relative w-12 h-6 rounded-full transition-colors
            ${parameters.preserve_code ? 'bg-blue-500' : 'bg-gray-600'}
          `}
        >
          <div
            className={`
              absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform
              ${parameters.preserve_code ? 'translate-x-6' : 'translate-x-0.5'}
            `}
          />
        </button>
      </div>

      {/* Group Under Headings (hierarchical only) */}
      {showGroupUnderHeadings && (
        <div className="flex items-center justify-between">
          <div>
            <label className="block text-sm font-medium">
              Group Under Headings
            </label>
            <p className="text-xs text-gray-400">
              Keep content with its heading
            </p>
          </div>
          <button
            onClick={() =>
              updateParam('group_under_headings', !parameters.group_under_headings)
            }
            className={`
              relative w-12 h-6 rounded-full transition-colors
              ${parameters.group_under_headings ? 'bg-blue-500' : 'bg-gray-600'}
            `}
          >
            <div
              className={`
                absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform
                ${parameters.group_under_headings ? 'translate-x-6' : 'translate-x-0.5'}
              `}
            />
          </button>
        </div>
      )}

      {/* Heading Weight (hierarchical only) */}
      {showHeadingWeight && (
        <div>
          <label className="block text-sm font-medium mb-2">
            Heading Weight
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min="1"
              max="3"
              step="0.5"
              value={parameters.heading_weight}
              onChange={(e) =>
                updateParam('heading_weight', parseFloat(e.target.value))
              }
              className="flex-1"
            />
            <span className="w-12 text-sm font-mono">
              {parameters.heading_weight.toFixed(1)}
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            How strongly to prefer breaking at headings (higher = stricter)
          </p>
        </div>
      )}

      {/* Reset to Defaults */}
      <div className="pt-3 border-t border-gray-700">
        <button
          onClick={() =>
            onChange(getDefaultParameters(strategy))
          }
          className="w-full px-3 py-2 text-sm bg-gray-700 hover:bg-gray-600 rounded"
        >
          Reset to Defaults
        </button>
      </div>
    </div>
  );
};

// Default parameters for each strategy
function getDefaultParameters(strategy: ChunkingStrategy): ChunkingParameters {
  switch (strategy) {
    case 'hierarchical':
      return {
        max_tokens: 512,
        overlap_tokens: 50,
        preserve_tables: true,
        preserve_code: true,
        group_under_headings: true,
        heading_weight: 2.0,
      };
    case 'fixed':
      return {
        max_tokens: 512,
        overlap_tokens: 50,
        preserve_tables: true,
        preserve_code: true,
        group_under_headings: false,
        heading_weight: 1.0,
      };
    case 'semantic':
      return {
        max_tokens: 400,
        overlap_tokens: 0,
        preserve_tables: true,
        preserve_code: true,
        group_under_headings: false,
        heading_weight: 1.0,
      };
    default:
      return {
        max_tokens: 512,
        overlap_tokens: 50,
        preserve_tables: true,
        preserve_code: true,
        group_under_headings: true,
        heading_weight: 1.5,
      };
  }
}
