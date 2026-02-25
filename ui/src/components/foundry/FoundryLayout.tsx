import { Factory, Cpu, BarChart3, Wrench, GitBranch } from 'lucide-react';
import { ToolHeader } from '@/components/shell/ToolHeader';

export function FoundryLayout() {
  return (
    <div className="flex flex-col h-full">
      <ToolHeader
        icon={Factory}
        title="Foundry"
        color="#6BA089"
      />

      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-md animate-fade-in">
          <div
            className="w-16 h-16 rounded-2xl mx-auto mb-6 flex items-center justify-center"
            style={{ background: 'rgba(107, 160, 137, 0.08)' }}
          >
            <Factory size={28} className="text-foundry-cast" strokeWidth={1.5} />
          </div>

          <h2 className="font-display text-xl font-semibold text-kiln-200 mb-2">
            Training & Evaluation
          </h2>
          <p className="text-sm text-kiln-400 mb-8 leading-relaxed">
            LoRA training pipeline with competency-based evaluation.
            Results in SME language, not ML metrics.
          </p>

          <div className="grid grid-cols-2 gap-3 text-left">
            {[
              { icon: Cpu, label: 'LoRA Training', desc: 'Sensible defaults' },
              { icon: BarChart3, label: 'Evaluation', desc: 'Competency scoring' },
              { icon: Wrench, label: 'Diagnostics', desc: 'Failure analysis' },
              { icon: GitBranch, label: 'Model Merging', desc: 'Linear & TIES' },
            ].map(({ icon: Icon, label, desc }) => (
              <div
                key={label}
                className="card p-3 flex items-start gap-3"
              >
                <Icon size={16} className="text-foundry-cast mt-0.5 flex-shrink-0" strokeWidth={1.5} />
                <div>
                  <div className="text-xs font-medium text-kiln-200">{label}</div>
                  <div className="text-2xs text-kiln-500">{desc}</div>
                </div>
              </div>
            ))}
          </div>

          <p className="text-2xs text-kiln-600 mt-8">
            Backend API required — Sprint 13
          </p>
        </div>
      </div>
    </div>
  );
}
