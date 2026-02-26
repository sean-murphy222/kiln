import { useState } from 'react';
import {
  Wrench,
  Activity,
  AlertTriangle,
  AlertCircle,
  Info,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  GitBranch,
  Merge,
} from 'lucide-react';
import { cn } from '@/lib/cn';
import { useFoundryStore } from '@/store/useFoundryStore';
import type { DiagnosticReport, DiagnosticIssue, ModelVersion, MergeResult } from '@/store/useFoundryStore';

const SEVERITY_CONFIG = {
  high: { label: 'High', icon: AlertCircle, className: 'bg-error/15 text-error' },
  medium: { label: 'Medium', icon: AlertTriangle, className: 'bg-warning/15 text-warning' },
  low: { label: 'Low', icon: Info, className: 'bg-info/15 text-info' },
} as const;

const CONVERGENCE_CONFIG = {
  converging: { label: 'Converging', icon: TrendingDown, className: 'text-success' },
  diverging: { label: 'Diverging', icon: TrendingUp, className: 'text-error' },
  plateau: { label: 'Plateau', icon: Minus, className: 'text-warning' },
  unknown: { label: 'Unknown', icon: Activity, className: 'text-kiln-500' },
} as const;

const OVERFIT_CONFIG = {
  high: { label: 'High Risk', className: 'bg-error/15 text-error' },
  medium: { label: 'Medium Risk', className: 'bg-warning/15 text-warning' },
  low: { label: 'Low Risk', className: 'bg-info/15 text-info' },
  none: { label: 'No Risk', className: 'bg-success/15 text-success' },
} as const;

function DiagnosticsSection() {
  const { diagnosticReport, trainingRuns, setDiagnosticReport } = useFoundryStore();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState('');

  const handleAnalyze = () => {
    if (!selectedRunId) return;
    setIsAnalyzing(true);
    setTimeout(() => {
      const report: DiagnosticReport = {
        run_id: selectedRunId,
        issues: [
          {
            type: 'Learning Rate',
            severity: 'medium',
            description: 'Loss oscillation detected in final 20% of training. Learning rate may be too high for fine-tuning phase.',
            recommendation: 'Consider using learning rate scheduler with cosine decay, or reduce base learning rate by 50%.',
          },
          {
            type: 'Data Distribution',
            severity: 'low',
            description: 'Training examples are unevenly distributed across competencies. "Parts Interpretation" has 3x fewer examples than average.',
            recommendation: 'Balance training data by adding more examples for underrepresented competencies.',
          },
        ],
        convergence_status: 'converging',
        overfit_risk: 'low',
        analyzed_at: new Date().toISOString(),
      };
      setDiagnosticReport(report);
      setIsAnalyzing(false);
    }, 2000);
  };

  const conv = diagnosticReport ? CONVERGENCE_CONFIG[diagnosticReport.convergence_status] : null;
  const overfit = diagnosticReport ? OVERFIT_CONFIG[diagnosticReport.overfit_risk] : null;
  const ConvIcon = conv?.icon ?? Activity;

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <h4 className="text-sm font-medium text-kiln-200 flex items-center gap-2 mb-3">
          <Wrench size={14} className="text-foundry-cast" />
          Training Diagnostics
        </h4>
        <div className="flex gap-2">
          <select
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            className="input-field-sm flex-1"
          >
            <option value="">Select a training run...</option>
            {trainingRuns.map((r) => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </select>
          <button
            onClick={handleAnalyze}
            disabled={!selectedRunId || isAnalyzing}
            className={cn('btn-secondary btn-sm', (!selectedRunId || isAnalyzing) && 'opacity-50')}
          >
            <RefreshCw size={12} className={cn(isAnalyzing && 'animate-spin')} />
            {isAnalyzing ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
      </div>

      {diagnosticReport && (
        <>
          {/* Status indicators */}
          <div className="grid grid-cols-2 gap-3">
            <div className="card p-3">
              <div className="text-2xs text-kiln-500 mb-1">Convergence</div>
              <div className={cn('flex items-center gap-1.5 text-sm font-medium', conv?.className)}>
                <ConvIcon size={14} />
                {conv?.label}
              </div>
            </div>
            <div className="card p-3">
              <div className="text-2xs text-kiln-500 mb-1">Overfit Risk</div>
              <span className={cn('px-2 py-0.5 rounded text-xs font-medium', overfit?.className)}>
                {overfit?.label}
              </span>
            </div>
          </div>

          {/* Issues */}
          <div className="space-y-2">
            {diagnosticReport.issues.map((issue, i) => {
              const sev = SEVERITY_CONFIG[issue.severity];
              const SevIcon = sev.icon;
              return (
                <div key={i} className="card p-4">
                  <div className="flex items-start gap-3">
                    <SevIcon size={16} className={cn('mt-0.5 shrink-0', sev.className.split(' ')[1])} />
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-2xs font-medium text-kiln-400 uppercase">{issue.type}</span>
                        <span className={cn('px-1.5 py-0.5 rounded text-2xs font-medium', sev.className)}>
                          {sev.label}
                        </span>
                      </div>
                      <p className="text-sm text-kiln-200 mb-2">{issue.description}</p>
                      <p className="text-sm text-foundry-cast">{issue.recommendation}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function VersionsSection() {
  const { modelVersions } = useFoundryStore();

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-kiln-200 flex items-center gap-2">
        <GitBranch size={14} className="text-foundry-cast" />
        Model Versions ({modelVersions.length})
      </h4>

      {modelVersions.length === 0 ? (
        <div className="text-center py-8 text-2xs text-kiln-500">
          No model versions registered. Complete a training run and register the resulting adapter.
        </div>
      ) : (
        <div className="space-y-1">
          {modelVersions.map((v) => (
            <div key={v.id} className="card p-3 flex items-center gap-3">
              <div className="flex-1">
                <div className="text-sm text-kiln-200">{v.name}</div>
                <div className="text-2xs text-kiln-500">{v.adapter_path}</div>
              </div>
              {v.evaluation_score !== null && (
                <span className="text-sm font-semibold text-kiln-200 tabular-nums">
                  {Math.round(v.evaluation_score * 100)}%
                </span>
              )}
              <span className="text-2xs text-kiln-500">
                {new Date(v.created_at).toLocaleDateString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MergingSection() {
  const { mergeResults, modelVersions, addMergeResult } = useFoundryStore();
  const [method, setMethod] = useState<'linear' | 'ties'>('linear');
  const [selectedAdapters, setSelectedAdapters] = useState<string[]>([]);
  const [outputName, setOutputName] = useState('');
  const [isMerging, setIsMerging] = useState(false);

  const handleMerge = () => {
    if (selectedAdapters.length < 2 || !outputName) return;
    setIsMerging(true);
    setTimeout(() => {
      addMergeResult({
        id: `merge-${Date.now()}`,
        output_path: `/models/merged/${outputName}`,
        method,
        adapters_merged: selectedAdapters,
        created_at: new Date().toISOString(),
      });
      setIsMerging(false);
      setOutputName('');
      setSelectedAdapters([]);
    }, 2000);
  };

  const toggleAdapter = (id: string) => {
    setSelectedAdapters((prev) =>
      prev.includes(id) ? prev.filter((a) => a !== id) : [...prev, id],
    );
  };

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <h4 className="text-sm font-medium text-kiln-200 flex items-center gap-2 mb-3">
          <Merge size={14} className="text-foundry-cast" />
          Merge Adapters
        </h4>

        <div className="space-y-3">
          <div>
            <label className="text-2xs text-kiln-400 mb-1 block">Method</label>
            <div className="flex gap-2">
              {(['linear', 'ties'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMethod(m)}
                  className={cn(
                    'btn-sm text-2xs',
                    method === m ? 'bg-foundry-cast-faint text-foundry-cast border border-foundry-cast/20' : 'btn-ghost',
                  )}
                >
                  {m === 'linear' ? 'Linear' : 'TIES'}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-2xs text-kiln-400 mb-1 block">Select Adapters (2+)</label>
            <div className="space-y-1">
              {modelVersions.map((v) => (
                <label
                  key={v.id}
                  className={cn(
                    'flex items-center gap-2 px-2 py-1.5 rounded-kiln cursor-pointer transition-colors',
                    selectedAdapters.includes(v.id) ? 'bg-foundry-cast-faint' : 'hover:bg-kiln-700/50',
                  )}
                >
                  <input
                    type="checkbox"
                    checked={selectedAdapters.includes(v.id)}
                    onChange={() => toggleAdapter(v.id)}
                    className="accent-foundry-cast"
                  />
                  <span className="text-sm text-kiln-300">{v.name}</span>
                </label>
              ))}
              {modelVersions.length < 2 && (
                <p className="text-2xs text-kiln-500 py-2">
                  Register at least 2 model versions to enable merging.
                </p>
              )}
            </div>
          </div>

          <div>
            <label className="text-2xs text-kiln-400 mb-1 block">Output Name</label>
            <input
              type="text"
              value={outputName}
              onChange={(e) => setOutputName(e.target.value)}
              placeholder="e.g. merged-maintenance-v2"
              className="input-field-sm"
            />
          </div>

          <button
            onClick={handleMerge}
            disabled={selectedAdapters.length < 2 || !outputName || isMerging}
            className={cn(
              'btn-primary btn-sm w-full',
              (selectedAdapters.length < 2 || !outputName || isMerging) && 'opacity-50',
            )}
          >
            <Merge size={12} />
            {isMerging ? 'Merging...' : 'Run Merge'}
          </button>
        </div>
      </div>

      {mergeResults.length > 0 && (
        <div>
          <h5 className="text-xs font-medium text-kiln-400 uppercase tracking-wide mb-2">
            Merge History
          </h5>
          {mergeResults.map((mr) => (
            <div key={mr.id} className="card p-3 mb-1">
              <div className="text-sm text-kiln-200">{mr.output_path}</div>
              <div className="text-2xs text-kiln-500">
                {mr.method} — {mr.adapters_merged.length} adapters — {new Date(mr.created_at).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DiagnosticsPanel() {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-6">
      <DiagnosticsSection />
      <div className="divider" />
      <VersionsSection />
      <div className="divider" />
      <MergingSection />
    </div>
  );
}
