import { useState, useEffect } from "react";
import { Plus, Search, Archive, Zap, FileEdit } from "lucide-react";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/store/useForgeStore";
import type { ForgeDiscipline } from "@/store/useForgeStore";
import { disciplineAPI } from "@/api/forge";
import { mapDiscipline } from "@/lib/forgeMappers";
import { showToast } from "@/components/common/Toast";

const STATUS_CONFIG = {
  draft: {
    label: "Draft",
    icon: FileEdit,
    className: "bg-kiln-600 text-kiln-300",
  },
  active: {
    label: "Active",
    icon: Zap,
    className: "bg-forge-heat-faint text-forge-heat",
  },
  archived: {
    label: "Archived",
    icon: Archive,
    className: "bg-kiln-700 text-kiln-500",
  },
} as const;

function coverageColor(pct: number): string {
  if (pct >= 80) return "bg-success";
  if (pct >= 40) return "bg-warning";
  return "bg-error";
}

function coverageTrackColor(pct: number): string {
  if (pct >= 80) return "bg-success/15";
  if (pct >= 40) return "bg-warning/15";
  return "bg-error/15";
}

interface DisciplineItemProps {
  discipline: ForgeDiscipline;
  isActive: boolean;
  onClick: () => void;
}

function DisciplineItem({
  discipline,
  isActive,
  onClick,
}: DisciplineItemProps) {
  const status = STATUS_CONFIG[discipline.status];
  const StatusIcon = status.icon;

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full text-left px-3 py-2.5 rounded-kiln transition-all duration-150",
        "group relative",
        isActive
          ? "bg-forge-heat-faint border border-forge-heat/20"
          : "hover:bg-kiln-700 border border-transparent",
      )}
    >
      {/* Active indicator bar */}
      {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-forge-heat" />
      )}

      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span
          className={cn(
            "text-sm font-medium truncate",
            isActive
              ? "text-kiln-100"
              : "text-kiln-300 group-hover:text-kiln-200",
          )}
        >
          {discipline.name}
        </span>
        <span
          className={cn(
            "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-2xs font-medium shrink-0",
            status.className,
          )}
        >
          <StatusIcon size={10} />
          {status.label}
        </span>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-3 text-2xs text-kiln-500 mb-1.5">
        <span>{discipline.competency_count} competencies</span>
        <span className="text-kiln-600">|</span>
        <span>{discipline.example_count} examples</span>
      </div>

      {/* Coverage bar */}
      <div className="flex items-center gap-2">
        <div
          className={cn(
            "flex-1 h-1 rounded-full",
            coverageTrackColor(discipline.coverage_pct),
          )}
        >
          <div
            className={cn(
              "h-full rounded-full transition-all duration-300",
              coverageColor(discipline.coverage_pct),
            )}
            style={{ width: `${Math.min(100, discipline.coverage_pct)}%` }}
          />
        </div>
        <span className="text-2xs text-kiln-500 tabular-nums w-8 text-right">
          {discipline.coverage_pct}%
        </span>
      </div>
    </button>
  );
}

export function DisciplineList() {
  const {
    disciplines,
    selectedDisciplineId,
    selectDiscipline,
    addDiscipline,
    setDisciplines,
    setLoading,
    setError,
  } = useForgeStore();
  const [search, setSearch] = useState("");
  const [showNewForm, setShowNewForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  // Load disciplines from the backend on mount.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    disciplineAPI
      .list()
      .then((items) => {
        if (!cancelled) setDisciplines(items.map(mapDiscipline));
      })
      .catch((err) => {
        if (!cancelled) {
          const msg =
            err instanceof Error ? err.message : "Failed to load disciplines";
          setError(msg);
          showToast("error", msg);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [setDisciplines, setLoading, setError]);

  const filtered = disciplines.filter((d) =>
    d.name.toLowerCase().includes(search.toLowerCase()),
  );

  const handleCreate = async () => {
    if (!newName.trim() || creating) return;
    setCreating(true);
    try {
      const created = await disciplineAPI.create({
        name: newName.trim(),
        description: "",
      });
      const mapped: ForgeDiscipline = mapDiscipline(created);
      addDiscipline(mapped);
      selectDiscipline(mapped.id);
      setNewName("");
      setShowNewForm(false);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to create discipline";
      setError(msg);
      showToast("error", msg);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="w-60 flex flex-col border-r border-kiln-600 bg-kiln-800/50">
      {/* Header */}
      <div className="px-3 pt-3 pb-2">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-display text-xs font-semibold text-kiln-400 uppercase tracking-wider">
            Disciplines
          </h3>
          <button
            onClick={() => setShowNewForm(!showNewForm)}
            className="p-1 rounded hover:bg-kiln-700 text-forge-heat transition-colors"
            title="New discipline"
          >
            <Plus size={14} />
          </button>
        </div>

        {/* New discipline form */}
        {showNewForm && (
          <div className="mb-2 animate-slide-down">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              placeholder="Discipline name..."
              className="input-field-sm mb-1.5"
              autoFocus
            />
            <div className="flex gap-1.5">
              <button
                onClick={handleCreate}
                disabled={creating || !newName.trim()}
                className={cn(
                  "btn-primary btn-sm flex-1 text-2xs",
                  (creating || !newName.trim()) && "opacity-60",
                )}
              >
                {creating ? "Creating..." : "Create"}
              </button>
              <button
                onClick={() => {
                  setShowNewForm(false);
                  setNewName("");
                }}
                className="btn-ghost btn-sm text-2xs"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Search */}
        <div className="relative">
          <Search
            size={12}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-kiln-500"
          />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter..."
            className="input-field-sm pl-7"
          />
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {filtered.length === 0 ? (
          <div className="text-center py-8 text-2xs text-kiln-500">
            {disciplines.length === 0 ? "No disciplines yet" : "No matches"}
          </div>
        ) : (
          filtered.map((d) => (
            <DisciplineItem
              key={d.id}
              discipline={d}
              isActive={d.id === selectedDisciplineId}
              onClick={() => selectDiscipline(d.id)}
            />
          ))
        )}
      </div>

      {/* Summary footer */}
      <div className="px-3 py-2 border-t border-kiln-600 text-2xs text-kiln-500">
        {disciplines.length} discipline{disciplines.length !== 1 ? "s" : ""}
      </div>
    </div>
  );
}
