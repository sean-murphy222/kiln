import {
  Hammer,
  BookOpen,
  Target,
  FileCheck,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/store/useForgeStore";
import type { ForgeView } from "@/store/useForgeStore";
import { ToolHeader } from "@/components/shell/ToolHeader";
import { DisciplineList } from "./DisciplineList";
import { DiscoveryWizard } from "./DiscoveryWizard";
import { CompetencyTree } from "./CompetencyTree";
import { ExampleList } from "./ExampleList";
import { ConsistencyReport } from "./ConsistencyReport";

const VIEW_TABS: { id: ForgeView; label: string; icon: typeof BookOpen }[] = [
  { id: "discovery", label: "Discovery", icon: BookOpen },
  { id: "competencies", label: "Competencies", icon: Target },
  { id: "examples", label: "Examples", icon: FileCheck },
  { id: "consistency", label: "Consistency", icon: AlertTriangle },
];

const VIEW_COMPONENTS: Record<ForgeView, React.ComponentType> = {
  discovery: DiscoveryWizard,
  competencies: CompetencyTree,
  examples: ExampleList,
  consistency: ConsistencyReport,
};

export function ForgeLayout() {
  const { activeView, setActiveView, selectedDisciplineId, disciplines } =
    useForgeStore();
  const selectedDiscipline = disciplines.find(
    (d) => d.id === selectedDisciplineId,
  );
  const ActiveComponent = VIEW_COMPONENTS[activeView];

  return (
    <div className="flex flex-col h-full">
      <ToolHeader
        icon={Hammer}
        title="Forge"
        color="#D4915C"
        breadcrumb={
          selectedDiscipline
            ? [
                selectedDiscipline.name,
                VIEW_TABS.find((t) => t.id === activeView)?.label ?? "",
              ]
            : undefined
        }
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar: Discipline list */}
        <DisciplineList />

        {/* Right: Tab bar + active view */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Tab bar */}
          {selectedDisciplineId && (
            <div className="flex items-center gap-0.5 px-4 py-1.5 border-b border-kiln-600 bg-kiln-800/30">
              {VIEW_TABS.map((tab) => {
                const isActive = activeView === tab.id;
                const TabIcon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveView(tab.id)}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-1.5 rounded-kiln text-xs font-medium transition-all duration-150",
                      isActive
                        ? "bg-forge-heat-faint text-forge-heat"
                        : "text-kiln-400 hover:text-kiln-200 hover:bg-kiln-700/50",
                    )}
                  >
                    <TabIcon size={13} />
                    {tab.label}
                  </button>
                );
              })}
            </div>
          )}

          {/* Active view */}
          <ActiveComponent />
        </div>
      </div>
    </div>
  );
}
