import { Factory, Cpu, BarChart3, Wrench } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useFoundryStore } from '@/store/useFoundryStore';
import type { FoundryTab } from '@/store/useFoundryStore';
import { ToolHeader } from '@/components/shell/ToolHeader';
import { TrainingPanel } from './TrainingPanel';
import { EvaluationPanel } from './EvaluationPanel';
import { DiagnosticsPanel } from './DiagnosticsPanel';

const TABS: { id: FoundryTab; label: string; icon: typeof Cpu }[] = [
  { id: 'training', label: 'Training', icon: Cpu },
  { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
  { id: 'diagnostics', label: 'Diagnostics & Merging', icon: Wrench },
];

const TAB_COMPONENTS: Record<FoundryTab, React.ComponentType> = {
  training: TrainingPanel,
  evaluation: EvaluationPanel,
  diagnostics: DiagnosticsPanel,
  versions: DiagnosticsPanel,
  merging: DiagnosticsPanel,
};

export function FoundryLayout() {
  const { activeTab, setActiveTab } = useFoundryStore();
  const ActiveComponent = TAB_COMPONENTS[activeTab];

  // Normalize versions/merging to diagnostics tab
  const displayTab = ['versions', 'merging'].includes(activeTab) ? 'diagnostics' : activeTab;

  return (
    <div className="flex flex-col h-full">
      <ToolHeader
        icon={Factory}
        title="Foundry"
        color="#6BA089"
        breadcrumb={[TABS.find((t) => t.id === displayTab)?.label ?? '']}
      />

      {/* Tab bar */}
      <div className="flex items-center gap-0.5 px-4 py-1.5 border-b border-kiln-600 bg-kiln-800/30">
        {TABS.map((tab) => {
          const isActive = displayTab === tab.id;
          const TabIcon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-kiln text-xs font-medium transition-all duration-150',
                isActive
                  ? 'bg-foundry-cast-faint text-foundry-cast'
                  : 'text-kiln-400 hover:text-kiln-200 hover:bg-kiln-700/50',
              )}
            >
              <TabIcon size={13} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Active panel */}
      <ActiveComponent />
    </div>
  );
}
