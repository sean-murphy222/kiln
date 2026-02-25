import { Settings, Info } from 'lucide-react';
import { ToolHeader } from '@/components/shell/ToolHeader';

export function SettingsPage() {
  return (
    <div className="flex flex-col h-full">
      <ToolHeader
        icon={Settings}
        title="Settings"
        color="#8892A8"
      />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto animate-fade-in">
          <div className="card p-6 mb-4">
            <h3 className="font-display text-sm font-semibold text-kiln-200 mb-4">
              About Kiln
            </h3>
            <div className="flex items-start gap-4">
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ background: 'rgba(232, 115, 74, 0.08)' }}
              >
                <Info size={18} className="text-ember" />
              </div>
              <div className="text-sm text-kiln-400 leading-relaxed">
                <p className="text-kiln-200 font-medium mb-1">
                  Kiln v0.2.0
                </p>
                <p>
                  A complete pipeline for trustworthy domain-specific AI.
                  Process documents, build curricula, train models, and
                  query with citations — all locally.
                </p>
              </div>
            </div>
          </div>

          <p className="text-2xs text-kiln-600 text-center">
            Full settings page — Sprint 17
          </p>
        </div>
      </div>
    </div>
  );
}
