import { useState } from 'react';
import { BarChart3, Play, ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useFoundryStore } from '@/store/useFoundryStore';
import type { EvaluationResult, CompetencyScore } from '@/store/useFoundryStore';

const RATING_CONFIG = {
  strong: { label: 'Strong', className: 'bg-success/15 text-success' },
  adequate: { label: 'Adequate', className: 'bg-info/15 text-info' },
  weak: { label: 'Weak', className: 'bg-error/15 text-error' },
  untested: { label: 'Untested', className: 'bg-kiln-600 text-kiln-400' },
} as const;

function ScoreBar({ score }: { score: CompetencyScore }) {
  const rating = RATING_CONFIG[score.rating];
  const pct = score.total > 0 ? Math.round((score.correct / score.total) * 100) : 0;

  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="text-sm text-kiln-300 w-48 truncate">{score.competency_name}</span>
      <div className="flex-1 h-2 bg-kiln-700 rounded-full">
        <div
          className={cn(
            'h-full rounded-full transition-all',
            pct >= 80 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-error',
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-2xs text-kiln-400 tabular-nums w-12 text-right">
        {score.correct}/{score.total}
      </span>
      <span className={cn('px-1.5 py-0.5 rounded text-2xs font-medium w-20 text-center', rating.className)}>
        {rating.label}
      </span>
    </div>
  );
}

function EvalDetail({ result }: { result: EvaluationResult }) {
  const pct = result.overall_total > 0
    ? Math.round((result.overall_correct / result.overall_total) * 100)
    : 0;

  return (
    <div className="p-4 space-y-4 animate-fade-in">
      {/* Summary */}
      <div className="flex items-center gap-6">
        <div className="text-center">
          <div className="text-3xl font-display font-bold text-kiln-100 tabular-nums">{pct}%</div>
          <div className="text-2xs text-kiln-500">Overall Score</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-display font-semibold text-kiln-200 tabular-nums">
            {result.overall_correct}/{result.overall_total}
          </div>
          <div className="text-2xs text-kiln-500">Correct</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-display font-semibold text-kiln-200">
            {result.model_name}
          </div>
          <div className="text-2xs text-kiln-500">Model</div>
        </div>
      </div>

      {/* Competency breakdown */}
      <div>
        <h4 className="text-xs font-medium text-kiln-400 uppercase tracking-wide mb-2">
          Competency Breakdown
        </h4>
        <div className="space-y-0.5">
          {result.competency_scores.map((cs) => (
            <ScoreBar key={cs.competency_name} score={cs} />
          ))}
        </div>
      </div>
    </div>
  );
}

export function EvaluationPanel() {
  const { evaluations, trainingRuns, addEvaluation, selectedEvalId, selectEvaluation } =
    useFoundryStore();
  const [isRunning, setIsRunning] = useState(false);
  const [selectedRunForEval, setSelectedRunForEval] = useState('');

  const completedRuns = trainingRuns.filter((r) => r.status === 'completed');
  const selectedEval = evaluations.find((e) => e.id === selectedEvalId);

  const handleRunEval = () => {
    if (!selectedRunForEval) return;
    setIsRunning(true);
    setTimeout(() => {
      const demoScores: CompetencyScore[] = [
        { competency_name: 'Procedural Comprehension', correct: 9, total: 10, score: 0.9, rating: 'strong' },
        { competency_name: 'Fault Isolation', correct: 7, total: 10, score: 0.7, rating: 'adequate' },
        { competency_name: 'Parts Interpretation', correct: 4, total: 10, score: 0.4, rating: 'weak' },
        { competency_name: 'Safety Compliance', correct: 8, total: 10, score: 0.8, rating: 'strong' },
        { competency_name: 'Preventive Maintenance', correct: 6, total: 10, score: 0.6, rating: 'adequate' },
      ];
      const eval_: EvaluationResult = {
        id: `eval-${Date.now()}`,
        training_run_id: selectedRunForEval,
        model_name: completedRuns.find((r) => r.id === selectedRunForEval)?.name ?? 'Unknown',
        overall_score: 0.68,
        overall_correct: 34,
        overall_total: 50,
        competency_scores: demoScores,
        created_at: new Date().toISOString(),
      };
      addEvaluation(eval_);
      selectEvaluation(eval_.id);
      setIsRunning(false);
    }, 2500);
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Run evaluation */}
      <div className="p-4 border-b border-kiln-600">
        <div className="card p-4">
          <h4 className="text-sm font-medium text-kiln-200 flex items-center gap-2 mb-3">
            <BarChart3 size={14} className="text-foundry-cast" />
            Run Evaluation
          </h4>
          <div className="flex gap-2">
            <select
              value={selectedRunForEval}
              onChange={(e) => setSelectedRunForEval(e.target.value)}
              className="input-field-sm flex-1"
            >
              <option value="">Select a completed training run...</option>
              {completedRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
            <button
              onClick={handleRunEval}
              disabled={!selectedRunForEval || isRunning}
              className={cn('btn-primary btn-sm', (!selectedRunForEval || isRunning) && 'opacity-50')}
            >
              <Play size={12} />
              {isRunning ? 'Evaluating...' : 'Evaluate'}
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Eval list */}
        <div className="w-72 border-r border-kiln-600 overflow-y-auto">
          <div className="px-3 py-2 text-xs font-medium text-kiln-400 uppercase tracking-wide">
            History ({evaluations.length})
          </div>
          {evaluations.map((ev) => {
            const pct = ev.overall_total > 0
              ? Math.round((ev.overall_correct / ev.overall_total) * 100)
              : 0;
            return (
              <button
                key={ev.id}
                onClick={() => selectEvaluation(ev.id)}
                className={cn(
                  'w-full text-left px-3 py-2.5 border-b border-kiln-600/50 transition-colors',
                  ev.id === selectedEvalId ? 'bg-foundry-cast-faint' : 'hover:bg-kiln-700/30',
                )}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-kiln-200 truncate">{ev.model_name}</span>
                  <span className="text-sm font-semibold text-kiln-100 tabular-nums">{pct}%</span>
                </div>
                <div className="text-2xs text-kiln-500">
                  {new Date(ev.created_at).toLocaleDateString()}
                </div>
              </button>
            );
          })}
          {evaluations.length === 0 && (
            <div className="text-center py-8 text-2xs text-kiln-500">No evaluations yet</div>
          )}
        </div>

        {/* Detail */}
        <div className="flex-1 overflow-y-auto">
          {selectedEval ? (
            <EvalDetail result={selectedEval} />
          ) : (
            <div className="flex-1 flex items-center justify-center h-full">
              <p className="text-sm text-kiln-500">Select an evaluation to view results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
