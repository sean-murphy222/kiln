/**
 * ProblemCard - Individual chunk problem display
 */

import type { Chunk } from '../../api/chonk';

export type ProblemType = 'semantic_incomplete' | 'semantic_contamination' | 'structural_breakage' | 'reference_orphaning';
export type ProblemSeverity = 'high' | 'medium' | 'low';

export interface Problem {
  id: string;
  chunkId: string;
  type: ProblemType;
  severity: ProblemSeverity;
  description: string;
  suggestedFix?: string;
  annotatedAt?: string;
}

interface ProblemCardProps {
  problem: Problem;
  chunk?: Chunk;
  isSelected: boolean;
  onClick: () => void;
}

const PROBLEM_TYPES: Record<ProblemType, { label: string; description: string; color: string }> = {
  semantic_incomplete: {
    label: 'Semantic Incompleteness',
    description: 'Chunk contains partial idea (dangling connectives, incomplete sentences)',
    color: 'text-red-400',
  },
  semantic_contamination: {
    label: 'Semantic Contamination',
    description: 'Chunk contains multiple unrelated ideas',
    color: 'text-orange-400',
  },
  structural_breakage: {
    label: 'Structural Breakage',
    description: 'Chunk splits logical unit (lists, tables, procedures)',
    color: 'text-yellow-400',
  },
  reference_orphaning: {
    label: 'Reference Orphaning',
    description: 'Chunk contains broken references ("see above", "as follows")',
    color: 'text-blue-400',
  },
};

export function ProblemCard({ problem, chunk, isSelected, onClick }: ProblemCardProps) {
  const typeInfo = PROBLEM_TYPES[problem.type];

  return (
    <div
      onClick={onClick}
      className={`
        border-2 border-black p-3 cursor-pointer
        hover:bg-kiln-900 transition-colors
        ${isSelected ? 'bg-ember/10' : 'bg-kiln-900'}
      `}
    >
      <div className="flex items-start justify-between mb-2">
        <div className={`text-xs font-bold ${typeInfo.color}`}>
          {typeInfo.label}
        </div>
        <div className={`
          text-xs px-2 py-1 border border-black
          ${problem.severity === 'high' ? 'bg-red-900/30 text-red-400' : ''}
          ${problem.severity === 'medium' ? 'bg-orange-900/30 text-orange-400' : ''}
          ${problem.severity === 'low' ? 'bg-yellow-900/30 text-yellow-400' : ''}
        `}>
          {problem.severity.toUpperCase()}
        </div>
      </div>
      <p className="text-xs text-kiln-500 mb-2">{problem.description}</p>
      {chunk && (
        <p className="text-xs text-kiln-500 truncate">
          Chunk: {chunk.content.substring(0, 60)}...
        </p>
      )}
      {problem.suggestedFix && (
        <div className="mt-2 pt-2 border-t border-kiln-500/20">
          <p className="text-xs text-green-400">
            <span className="font-bold">Suggested Fix:</span> {problem.suggestedFix}
          </p>
        </div>
      )}
    </div>
  );
}

export { PROBLEM_TYPES };
