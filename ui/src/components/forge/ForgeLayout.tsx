import { Hammer, BookOpen, Target, FileCheck, Users } from 'lucide-react';
import { ToolHeader } from '@/components/shell/ToolHeader';

export function ForgeLayout() {
  return (
    <div className="flex flex-col h-full">
      <ToolHeader
        icon={Hammer}
        title="Forge"
        color="#D4915C"
      />

      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-md animate-fade-in">
          <div
            className="w-16 h-16 rounded-2xl mx-auto mb-6 flex items-center justify-center"
            style={{ background: 'rgba(212, 145, 92, 0.08)' }}
          >
            <Hammer size={28} className="text-forge-heat" strokeWidth={1.5} />
          </div>

          <h2 className="font-display text-xl font-semibold text-kiln-200 mb-2">
            Curriculum Builder
          </h2>
          <p className="text-sm text-kiln-400 mb-8 leading-relaxed">
            Guide domain experts through creating human-validated training data.
            Discipline-level methodology with multi-contributor support.
          </p>

          <div className="grid grid-cols-2 gap-3 text-left">
            {[
              { icon: BookOpen, label: 'Discovery Interview', desc: 'Map discipline structure' },
              { icon: Target, label: 'Competency Mapping', desc: 'Define skill coverage' },
              { icon: FileCheck, label: 'Example Elicitation', desc: 'Gather training pairs' },
              { icon: Users, label: 'Multi-Contributor', desc: 'Consistency checking' },
            ].map(({ icon: Icon, label, desc }) => (
              <div
                key={label}
                className="card p-3 flex items-start gap-3"
              >
                <Icon size={16} className="text-forge-heat mt-0.5 flex-shrink-0" strokeWidth={1.5} />
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
