import { useState } from 'react';
import { Play, Square, Clock, CheckCircle2, XCircle, AlertCircle, Cpu } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useFoundryStore } from '@/store/useFoundryStore';
import type { TrainingRun } from '@/store/useFoundryStore';

const STATUS_CONFIG = {
  pending: { label: 'Pending', icon: Clock, className: 'bg-kiln-600 text-kiln-300' },
  running: { label: 'Running', icon: Play, className: 'bg-info/15 text-info' },
  completed: { label: 'Completed', icon: CheckCircle2, className: 'bg-success/15 text-success' },
  failed: { label: 'Failed', icon: XCircle, className: 'bg-error/15 text-error' },
  cancelled: { label: 'Cancelled', icon: AlertCircle, className: 'bg-warning/15 text-warning' },
} as const;

function TrainingConfigForm() {
  const { addTrainingRun } = useFoundryStore();
  const [baseModel, setBaseModel] = useState('meta-llama/Llama-3.1-8B');
  const [adapterName, setAdapterName] = useState('');
  const [loraRank, setLoraRank] = useState(16);
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(0.0002);

  const handleStart = () => {
    const run: TrainingRun = {
      id: `run-${Date.now()}`,
      name: adapterName || `Training ${new Date().toLocaleDateString()}`,
      status: 'running',
      progress: 0,
      base_model: baseModel,
      metrics: {},
      created_at: new Date().toISOString(),
      completed_at: null,
      error: null,
    };
    addTrainingRun(run);
    setAdapterName('');
  };

  return (
    <div className="card p-4 space-y-3">
      <h4 className="text-sm font-medium text-kiln-200 flex items-center gap-2">
        <Cpu size={14} className="text-foundry-cast" />
        New Training Configuration
      </h4>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-2xs text-kiln-400 mb-1 block">Base Model</label>
          <select
            value={baseModel}
            onChange={(e) => setBaseModel(e.target.value)}
            className="input-field-sm"
          >
            <option value="meta-llama/Llama-3.1-8B">Llama 3.1 8B</option>
            <option value="mistralai/Mistral-7B-v0.3">Mistral 7B v0.3</option>
            <option value="microsoft/phi-3-mini">Phi-3 Mini</option>
          </select>
        </div>
        <div>
          <label className="text-2xs text-kiln-400 mb-1 block">Adapter Name</label>
          <input
            type="text"
            value={adapterName}
            onChange={(e) => setAdapterName(e.target.value)}
            placeholder="e.g. maintenance-v1"
            className="input-field-sm"
          />
        </div>
        <div>
          <label className="text-2xs text-kiln-400 mb-1 block">LoRA Rank</label>
          <input
            type="number"
            value={loraRank}
            onChange={(e) => setLoraRank(parseInt(e.target.value) || 16)}
            min={4}
            max={128}
            className="input-field-sm"
          />
        </div>
        <div>
          <label className="text-2xs text-kiln-400 mb-1 block">Epochs</label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value) || 3)}
            min={1}
            max={20}
            className="input-field-sm"
          />
        </div>
        <div className="col-span-2">
          <label className="text-2xs text-kiln-400 mb-1 block">
            Learning Rate: {learningRate.toExponential(1)}
          </label>
          <input
            type="range"
            min={0.00001}
            max={0.001}
            step={0.00001}
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="w-full accent-foundry-cast"
          />
        </div>
      </div>

      <div className="flex justify-end gap-2 pt-1">
        <button className="btn-secondary btn-sm">Auto-Configure</button>
        <button onClick={handleStart} className="btn-primary btn-sm">
          <Play size={12} />
          Start Training
        </button>
      </div>
    </div>
  );
}

interface RunRowProps {
  run: TrainingRun;
  isSelected: boolean;
  onSelect: () => void;
}

function RunRow({ run, isSelected, onSelect }: RunRowProps) {
  const status = STATUS_CONFIG[run.status];
  const StatusIcon = status.icon;
  const { updateTrainingRun } = useFoundryStore();

  return (
    <button
      onClick={onSelect}
      className={cn(
        'w-full text-left px-4 py-3 flex items-center gap-3 border-b border-kiln-600/50 transition-colors',
        isSelected ? 'bg-foundry-cast-faint' : 'hover:bg-kiln-700/30',
      )}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-medium text-kiln-200 truncate">{run.name}</span>
          <span className={cn('inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-2xs font-medium', status.className)}>
            <StatusIcon size={10} />
            {status.label}
          </span>
        </div>
        <div className="text-2xs text-kiln-500">
          {run.base_model} — {new Date(run.created_at).toLocaleDateString()}
        </div>
      </div>

      {run.status === 'running' && (
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-20 h-1.5 bg-kiln-700 rounded-full">
            <div
              className="h-full bg-foundry-cast rounded-full transition-all"
              style={{ width: `${run.progress}%` }}
            />
          </div>
          <span className="text-2xs text-foundry-cast tabular-nums">{run.progress}%</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              updateTrainingRun(run.id, { status: 'cancelled' });
            }}
            className="p-1 rounded hover:bg-kiln-600 text-kiln-500 hover:text-error transition-colors"
            title="Cancel"
          >
            <Square size={12} />
          </button>
        </div>
      )}
    </button>
  );
}

export function TrainingPanel() {
  const { trainingRuns, selectedRunId, selectRun } = useFoundryStore();

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="p-4">
        <TrainingConfigForm />
      </div>

      <div className="px-4 py-2 border-t border-kiln-600">
        <h4 className="text-xs font-medium text-kiln-400 uppercase tracking-wide">
          Training History ({trainingRuns.length})
        </h4>
      </div>

      <div className="flex-1 overflow-y-auto">
        {trainingRuns.length === 0 ? (
          <div className="text-center py-12 text-2xs text-kiln-500">
            No training runs yet. Configure and start a training above.
          </div>
        ) : (
          trainingRuns.map((run) => (
            <RunRow
              key={run.id}
              run={run}
              isSelected={run.id === selectedRunId}
              onSelect={() => selectRun(run.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}
