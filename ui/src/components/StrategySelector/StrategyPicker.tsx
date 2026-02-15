/**
 * StrategyPicker - Choose chunking strategy
 *
 * Radio selector for different chunking approaches.
 */

import React from 'react';

export type ChunkingStrategy = 'hierarchical' | 'fixed' | 'semantic' | 'custom';

export interface StrategyOption {
  id: ChunkingStrategy;
  name: string;
  description: string;
  recommended: boolean;
  icon: string;
  pros: string[];
  cons: string[];
}

const STRATEGIES: StrategyOption[] = [
  {
    id: 'hierarchical',
    name: 'Hierarchical',
    description: 'Section-based chunking that respects document structure',
    recommended: true,
    icon: '🌳',
    pros: [
      'Preserves document structure',
      'Complete sections, not fragments',
      'Hierarchy paths for context',
      'Best for structured documents',
    ],
    cons: [
      'Requires good heading structure',
      'May create variable-sized chunks',
    ],
  },
  {
    id: 'fixed',
    name: 'Fixed',
    description: 'Token-based sliding window chunking',
    recommended: false,
    icon: '📏',
    pros: [
      'Consistent chunk sizes',
      'Simple and predictable',
      'Good baseline for comparison',
    ],
    cons: [
      'Destroys document structure',
      'Mixes unrelated content',
      'No context preservation',
    ],
  },
  {
    id: 'semantic',
    name: 'Semantic',
    description: 'Embedding-based similarity chunking',
    recommended: false,
    icon: '🧠',
    pros: [
      'Groups related content',
      'Good for unstructured documents',
      'Semantic coherence',
    ],
    cons: [
      'Expensive (requires embeddings)',
      'Slower processing',
      'Less predictable',
    ],
  },
  {
    id: 'custom',
    name: 'Custom',
    description: 'User-defined chunking rules',
    recommended: false,
    icon: '⚙️',
    pros: ['Full control', 'Domain-specific rules'],
    cons: ['Requires expertise', 'Manual configuration'],
  },
];

interface StrategyPickerProps {
  selected: ChunkingStrategy;
  onSelect: (strategy: ChunkingStrategy) => void;
  disabled?: boolean;
}

export const StrategyPicker: React.FC<StrategyPickerProps> = ({
  selected,
  onSelect,
  disabled = false,
}) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold">Choose Chunking Strategy</h3>
        <span className="text-sm text-gray-400">
          Not sure? Try Hierarchical ⭐
        </span>
      </div>

      {STRATEGIES.map((strategy) => (
        <div
          key={strategy.id}
          className={`
            relative border-2 rounded-lg p-4 cursor-pointer transition-all
            ${
              selected === strategy.id
                ? 'border-blue-500 bg-blue-900 bg-opacity-20'
                : 'border-gray-700 hover:border-gray-600'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          onClick={() => !disabled && onSelect(strategy.id)}
        >
          {/* Recommended badge */}
          {strategy.recommended && (
            <div className="absolute top-2 right-2">
              <span className="px-2 py-1 text-xs font-bold bg-yellow-500 text-black rounded">
                RECOMMENDED
              </span>
            </div>
          )}

          {/* Header */}
          <div className="flex items-start gap-3">
            {/* Radio button */}
            <div className="flex-shrink-0 mt-1">
              <div
                className={`
                  w-5 h-5 rounded-full border-2 flex items-center justify-center
                  ${
                    selected === strategy.id
                      ? 'border-blue-500 bg-blue-500'
                      : 'border-gray-600'
                  }
                `}
              >
                {selected === strategy.id && (
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                )}
              </div>
            </div>

            {/* Content */}
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-2xl">{strategy.icon}</span>
                <h4 className="text-lg font-bold">{strategy.name}</h4>
              </div>
              <p className="text-sm text-gray-300 mb-3">
                {strategy.description}
              </p>

              {/* Pros/Cons */}
              {selected === strategy.id && (
                <div className="grid grid-cols-2 gap-4 mt-3 pt-3 border-t border-gray-700">
                  {/* Pros */}
                  <div>
                    <h5 className="text-xs font-bold text-green-400 mb-2">
                      ✓ Pros
                    </h5>
                    <ul className="space-y-1">
                      {strategy.pros.map((pro, i) => (
                        <li key={i} className="text-xs text-gray-300">
                          • {pro}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Cons */}
                  <div>
                    <h5 className="text-xs font-bold text-red-400 mb-2">
                      ✗ Cons
                    </h5>
                    <ul className="space-y-1">
                      {strategy.cons.map((con, i) => (
                        <li key={i} className="text-xs text-gray-300">
                          • {con}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
