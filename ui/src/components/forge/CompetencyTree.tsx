import { useState, useCallback } from "react";
import {
  ChevronRight,
  ChevronDown,
  Plus,
  Pencil,
  Trash2,
  Target,
  X,
  Check,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/store/useForgeStore";
import type { ForgeCompetency } from "@/store/useForgeStore";

const LEVEL_CONFIG = {
  foundational: { label: "Foundation", className: "bg-info/15 text-info" },
  intermediate: {
    label: "Intermediate",
    className: "bg-success/15 text-success",
  },
  advanced: { label: "Advanced", className: "bg-warning/15 text-warning" },
  expert: { label: "Expert", className: "bg-forge-heat-faint text-forge-heat" },
} as const;

type CompetencyLevel = keyof typeof LEVEL_CONFIG;

function coverageClass(current: number, target: number): string {
  if (target === 0) return "text-kiln-500";
  const pct = (current / target) * 100;
  if (pct >= 80) return "text-success";
  if (pct >= 40) return "text-warning";
  return "text-error";
}

function coverageBarClass(current: number, target: number): string {
  if (target === 0) return "bg-kiln-600";
  const pct = (current / target) * 100;
  if (pct >= 80) return "bg-success";
  if (pct >= 40) return "bg-warning";
  return "bg-error";
}

interface InlineFormProps {
  initial?: {
    name: string;
    description: string;
    level: CompetencyLevel;
    target_count: number;
  };
  onSave: (data: {
    name: string;
    description: string;
    level: CompetencyLevel;
    target_count: number;
  }) => void;
  onCancel: () => void;
}

function InlineForm({ initial, onSave, onCancel }: InlineFormProps) {
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [level, setLevel] = useState<CompetencyLevel>(
    initial?.level ?? "foundational",
  );
  const [targetCount, setTargetCount] = useState(initial?.target_count ?? 10);

  const handleSubmit = () => {
    if (!name.trim()) return;
    onSave({
      name: name.trim(),
      description: description.trim(),
      level,
      target_count: targetCount,
    });
  };

  return (
    <div className="card p-3 space-y-2 animate-slide-down">
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
        placeholder="Competency name..."
        className="input-field-sm"
        autoFocus
      />
      <input
        type="text"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Brief description..."
        className="input-field-sm"
      />
      <div className="flex gap-2">
        <select
          value={level}
          onChange={(e) => setLevel(e.target.value as CompetencyLevel)}
          className="input-field-sm flex-1"
        >
          <option value="foundational">Foundational</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
          <option value="expert">Expert</option>
        </select>
        <input
          type="number"
          value={targetCount}
          onChange={(e) =>
            setTargetCount(Math.max(1, parseInt(e.target.value) || 1))
          }
          min={1}
          className="input-field-sm w-20"
          title="Target example count"
        />
      </div>
      <div className="flex justify-end gap-1.5">
        <button onClick={onCancel} className="btn-ghost btn-sm text-2xs">
          <X size={12} />
          Cancel
        </button>
        <button onClick={handleSubmit} className="btn-primary btn-sm text-2xs">
          <Check size={12} />
          {initial ? "Update" : "Add"}
        </button>
      </div>
    </div>
  );
}

interface TreeNodeProps {
  competency: ForgeCompetency;
  children: ForgeCompetency[];
  depth: number;
  allCompetencies: ForgeCompetency[];
}

function TreeNode({
  competency,
  children,
  depth,
  allCompetencies,
}: TreeNodeProps) {
  const [expanded, setExpanded] = useState(true);
  const [editing, setEditing] = useState(false);
  const [addingChild, setAddingChild] = useState(false);
  const {
    updateCompetency,
    removeCompetency,
    addCompetency,
    selectedDisciplineId,
  } = useForgeStore();

  const levelConfig = LEVEL_CONFIG[competency.level];
  const hasChildren = children.length > 0;
  const pct =
    competency.target_count > 0
      ? Math.round((competency.example_count / competency.target_count) * 100)
      : 0;

  const handleSaveEdit = useCallback(
    (data: {
      name: string;
      description: string;
      level: CompetencyLevel;
      target_count: number;
    }) => {
      updateCompetency(competency.id, data);
      setEditing(false);
    },
    [competency.id, updateCompetency],
  );

  const handleAddChild = useCallback(
    (data: {
      name: string;
      description: string;
      level: CompetencyLevel;
      target_count: number;
    }) => {
      addCompetency({
        id: `comp-${Date.now()}`,
        discipline_id: selectedDisciplineId ?? "",
        name: data.name,
        description: data.description,
        level: data.level,
        parent_id: competency.id,
        example_count: 0,
        target_count: data.target_count,
      });
      setAddingChild(false);
    },
    [competency.id, selectedDisciplineId, addCompetency],
  );

  if (editing) {
    return (
      <div style={{ paddingLeft: `${depth * 20}px` }}>
        <InlineForm
          initial={{
            name: competency.name,
            description: competency.description,
            level: competency.level,
            target_count: competency.target_count,
          }}
          onSave={handleSaveEdit}
          onCancel={() => setEditing(false)}
        />
      </div>
    );
  }

  return (
    <div>
      <div
        className={cn(
          "group flex items-center gap-2 py-1.5 px-2 rounded-kiln hover:bg-kiln-700/50 transition-colors",
        )}
        style={{ paddingLeft: `${depth * 20 + 8}px` }}
      >
        {/* Expand toggle */}
        <button
          onClick={() => setExpanded(!expanded)}
          className={cn(
            "p-0.5 rounded transition-colors",
            hasChildren
              ? "text-kiln-400 hover:text-kiln-200"
              : "text-transparent pointer-events-none",
          )}
        >
          {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        </button>

        {/* Name + level badge */}
        <div className="flex-1 flex items-center gap-2 min-w-0">
          <span className="text-sm text-kiln-200 truncate">
            {competency.name}
          </span>
          <span
            className={cn(
              "px-1.5 py-0.5 rounded text-2xs font-medium shrink-0",
              levelConfig.className,
            )}
          >
            {levelConfig.label}
          </span>
        </div>

        {/* Example count / target */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-16 h-1 bg-kiln-700 rounded-full">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                coverageBarClass(
                  competency.example_count,
                  competency.target_count,
                ),
              )}
              style={{ width: `${Math.min(100, pct)}%` }}
            />
          </div>
          <span
            className={cn(
              "text-2xs tabular-nums w-12 text-right",
              coverageClass(competency.example_count, competency.target_count),
            )}
          >
            {competency.example_count}/{competency.target_count}
          </span>
        </div>

        {/* Actions (visible on hover) */}
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => setAddingChild(true)}
            className="p-1 rounded hover:bg-kiln-600 text-kiln-500 hover:text-forge-heat transition-colors"
            title="Add child competency"
          >
            <Plus size={12} />
          </button>
          <button
            onClick={() => setEditing(true)}
            className="p-1 rounded hover:bg-kiln-600 text-kiln-500 hover:text-kiln-200 transition-colors"
            title="Edit"
          >
            <Pencil size={12} />
          </button>
          <button
            onClick={() => removeCompetency(competency.id)}
            className="p-1 rounded hover:bg-kiln-600 text-kiln-500 hover:text-error transition-colors"
            title="Delete"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div>
          {children.map((child) => (
            <TreeNode
              key={child.id}
              competency={child}
              children={allCompetencies.filter((c) => c.parent_id === child.id)}
              depth={depth + 1}
              allCompetencies={allCompetencies}
            />
          ))}
        </div>
      )}

      {/* Inline add child */}
      {addingChild && (
        <div style={{ paddingLeft: `${(depth + 1) * 20}px` }}>
          <InlineForm
            onSave={handleAddChild}
            onCancel={() => setAddingChild(false)}
          />
        </div>
      )}
    </div>
  );
}

export function CompetencyTree() {
  const { competencies, addCompetency, selectedDisciplineId } = useForgeStore();
  const [addingRoot, setAddingRoot] = useState(false);

  const disciplineCompetencies = competencies.filter(
    (c) => c.discipline_id === selectedDisciplineId,
  );
  const roots = disciplineCompetencies.filter((c) => c.parent_id === null);

  const totalExamples = disciplineCompetencies.reduce(
    (s, c) => s + c.example_count,
    0,
  );
  const totalTarget = disciplineCompetencies.reduce(
    (s, c) => s + c.target_count,
    0,
  );
  const overallPct =
    totalTarget > 0 ? Math.round((totalExamples / totalTarget) * 100) : 0;

  const handleAddRoot = useCallback(
    (data: {
      name: string;
      description: string;
      level: CompetencyLevel;
      target_count: number;
    }) => {
      addCompetency({
        id: `comp-${Date.now()}`,
        discipline_id: selectedDisciplineId ?? "",
        name: data.name,
        description: data.description,
        level: data.level,
        parent_id: null,
        example_count: 0,
        target_count: data.target_count,
      });
      setAddingRoot(false);
    },
    [selectedDisciplineId, addCompetency],
  );

  if (!selectedDisciplineId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Target
            size={32}
            className="mx-auto mb-3 text-kiln-500"
            strokeWidth={1.5}
          />
          <p className="text-sm text-kiln-400">
            Select a discipline to view competencies
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Summary header */}
      <div className="px-4 py-3 border-b border-kiln-600 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-kiln-200">Competency Map</h3>
          <p className="text-2xs text-kiln-500">
            {disciplineCompetencies.length} competencies — {totalExamples}/
            {totalTarget} examples ({overallPct}% coverage)
          </p>
        </div>
        <button
          onClick={() => setAddingRoot(true)}
          className="btn-secondary btn-sm"
        >
          <Plus size={13} />
          Add Competency
        </button>
      </div>

      {/* Tree */}
      <div className="flex-1 overflow-y-auto p-2">
        {addingRoot && (
          <InlineForm
            onSave={handleAddRoot}
            onCancel={() => setAddingRoot(false)}
          />
        )}

        {roots.length === 0 && !addingRoot ? (
          <div className="text-center py-12 text-2xs text-kiln-500">
            No competencies defined yet. Click "Add Competency" to begin.
          </div>
        ) : (
          roots.map((root) => (
            <TreeNode
              key={root.id}
              competency={root}
              children={disciplineCompetencies.filter(
                (c) => c.parent_id === root.id,
              )}
              depth={0}
              allCompetencies={disciplineCompetencies}
            />
          ))
        )}
      </div>
    </div>
  );
}
