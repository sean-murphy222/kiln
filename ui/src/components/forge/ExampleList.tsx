import { useState, useMemo } from "react";
import {
  FileCheck,
  Filter,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Search,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/store/useForgeStore";
import type { ForgeExample } from "@/store/useForgeStore";

const STATUS_CONFIG = {
  draft: {
    label: "Draft",
    icon: Clock,
    className: "bg-kiln-600 text-kiln-300",
  },
  approved: {
    label: "Approved",
    icon: CheckCircle2,
    className: "bg-success/15 text-success",
  },
  rejected: {
    label: "Rejected",
    icon: XCircle,
    className: "bg-error/15 text-error",
  },
  needs_revision: {
    label: "Needs Revision",
    icon: AlertTriangle,
    className: "bg-warning/15 text-warning",
  },
} as const;

type ExampleStatus = keyof typeof STATUS_CONFIG;

interface ExampleRowProps {
  example: ForgeExample;
  isExpanded: boolean;
  onToggle: () => void;
  onApprove: () => void;
  onReject: () => void;
  onRevision: () => void;
}

function ExampleRow({
  example,
  isExpanded,
  onToggle,
  onApprove,
  onReject,
  onRevision,
}: ExampleRowProps) {
  const status = STATUS_CONFIG[example.status];
  const StatusIcon = status.icon;

  return (
    <div className={cn("border-b border-kiln-600/50 last:border-0")}>
      {/* Row */}
      <button
        onClick={onToggle}
        className={cn(
          "w-full text-left px-4 py-2.5 flex items-center gap-3 hover:bg-kiln-700/30 transition-colors",
          isExpanded && "bg-kiln-700/20",
        )}
      >
        <span className="text-kiln-500 shrink-0">
          {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>

        {/* Question preview */}
        <span className="flex-1 text-sm text-kiln-200 truncate min-w-0">
          {example.question}
        </span>

        {/* Competency */}
        <span className="text-2xs text-kiln-400 truncate max-w-[140px] shrink-0">
          {example.competency_name}
        </span>

        {/* Status badge */}
        <span
          className={cn(
            "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-2xs font-medium shrink-0",
            status.className,
          )}
        >
          <StatusIcon size={10} />
          {status.label}
        </span>

        {/* Contributor */}
        <span className="text-2xs text-kiln-500 w-20 text-right truncate shrink-0">
          {example.contributor_name}
        </span>
      </button>

      {/* Expanded detail */}
      {isExpanded && (
        <div className="px-4 pb-4 pt-1 animate-slide-down">
          <div className="ml-7 space-y-3">
            <div>
              <label className="text-2xs font-medium text-forge-heat uppercase tracking-wide mb-1 block">
                Question
              </label>
              <p className="text-sm text-kiln-200 leading-relaxed">
                {example.question}
              </p>
            </div>

            <div>
              <label className="text-2xs font-medium text-forge-heat uppercase tracking-wide mb-1 block">
                Answer
              </label>
              <p className="text-sm text-kiln-300 leading-relaxed whitespace-pre-wrap">
                {example.answer}
              </p>
            </div>

            {example.context && (
              <div>
                <label className="text-2xs font-medium text-kiln-500 uppercase tracking-wide mb-1 block">
                  Context
                </label>
                <p className="text-sm text-kiln-400 leading-relaxed italic">
                  {example.context}
                </p>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex items-center gap-2 pt-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onApprove();
                }}
                className="btn-sm inline-flex items-center gap-1.5 bg-success/10 text-success border border-success/20 rounded-kiln hover:bg-success/20 transition-colors text-2xs font-medium px-3 py-1.5"
              >
                <CheckCircle2 size={12} />
                Approve
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRevision();
                }}
                className="btn-sm inline-flex items-center gap-1.5 bg-warning/10 text-warning border border-warning/20 rounded-kiln hover:bg-warning/20 transition-colors text-2xs font-medium px-3 py-1.5"
              >
                <AlertTriangle size={12} />
                Needs Revision
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onReject();
                }}
                className="btn-sm inline-flex items-center gap-1.5 bg-error/10 text-error border border-error/20 rounded-kiln hover:bg-error/20 transition-colors text-2xs font-medium px-3 py-1.5"
              >
                <XCircle size={12} />
                Reject
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function ExampleList() {
  const { examples, competencies, selectedDisciplineId, updateExample } =
    useForgeStore();
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<ExampleStatus | "all">(
    "all",
  );
  const [competencyFilter, setCompetencyFilter] = useState<string | "all">(
    "all",
  );
  const [showFilters, setShowFilters] = useState(false);

  const disciplineCompetencies = competencies.filter(
    (c) => c.discipline_id === selectedDisciplineId,
  );
  const competencyIds = new Set(disciplineCompetencies.map((c) => c.id));

  const disciplineExamples = examples.filter((e) =>
    competencyIds.has(e.competency_id),
  );

  const filtered = useMemo(() => {
    return disciplineExamples.filter((e) => {
      if (statusFilter !== "all" && e.status !== statusFilter) return false;
      if (competencyFilter !== "all" && e.competency_id !== competencyFilter)
        return false;
      if (search) {
        const q = search.toLowerCase();
        return (
          e.question.toLowerCase().includes(q) ||
          e.answer.toLowerCase().includes(q) ||
          e.competency_name.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [disciplineExamples, statusFilter, competencyFilter, search]);

  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = { all: disciplineExamples.length };
    for (const e of disciplineExamples) {
      counts[e.status] = (counts[e.status] ?? 0) + 1;
    }
    return counts;
  }, [disciplineExamples]);

  if (!selectedDisciplineId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <FileCheck
            size={32}
            className="mx-auto mb-3 text-kiln-500"
            strokeWidth={1.5}
          />
          <p className="text-sm text-kiln-400">
            Select a discipline to view examples
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-kiln-600">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-sm font-medium text-kiln-200">
              Training Examples
            </h3>
            <p className="text-2xs text-kiln-500">
              {filtered.length} of {disciplineExamples.length} examples
            </p>
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              "btn-ghost btn-sm",
              showFilters && "bg-kiln-700 text-kiln-200",
            )}
          >
            <Filter size={13} />
            Filters
          </button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search
            size={13}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-kiln-500"
          />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search examples..."
            className="input-field-sm pl-8"
          />
        </div>

        {/* Filters panel */}
        {showFilters && (
          <div className="mt-2 flex gap-2 animate-slide-down">
            <select
              value={statusFilter}
              onChange={(e) =>
                setStatusFilter(e.target.value as ExampleStatus | "all")
              }
              className="input-field-sm flex-1"
            >
              <option value="all">
                All statuses ({statusCounts.all ?? 0})
              </option>
              <option value="draft">Draft ({statusCounts.draft ?? 0})</option>
              <option value="approved">
                Approved ({statusCounts.approved ?? 0})
              </option>
              <option value="needs_revision">
                Needs Revision ({statusCounts.needs_revision ?? 0})
              </option>
              <option value="rejected">
                Rejected ({statusCounts.rejected ?? 0})
              </option>
            </select>
            <select
              value={competencyFilter}
              onChange={(e) => setCompetencyFilter(e.target.value)}
              className="input-field-sm flex-1"
            >
              <option value="all">All competencies</option>
              {disciplineCompetencies.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Table header */}
      <div className="px-4 py-1.5 flex items-center gap-3 border-b border-kiln-600/50 bg-kiln-800/50 text-2xs font-medium text-kiln-500 uppercase tracking-wide">
        <span className="w-5" />
        <span className="flex-1">Question</span>
        <span className="w-[140px]">Competency</span>
        <span className="w-24">Status</span>
        <span className="w-20 text-right">By</span>
      </div>

      {/* Example rows */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="text-center py-12 text-2xs text-kiln-500">
            {disciplineExamples.length === 0
              ? "No examples yet. Begin elicitation to gather training pairs."
              : "No examples match the current filters."}
          </div>
        ) : (
          filtered.map((example) => (
            <ExampleRow
              key={example.id}
              example={example}
              isExpanded={expandedId === example.id}
              onToggle={() =>
                setExpandedId(expandedId === example.id ? null : example.id)
              }
              onApprove={() =>
                updateExample(example.id, { status: "approved" })
              }
              onReject={() => updateExample(example.id, { status: "rejected" })}
              onRevision={() =>
                updateExample(example.id, { status: "needs_revision" })
              }
            />
          ))
        )}
      </div>
    </div>
  );
}
