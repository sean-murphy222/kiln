import { useState } from "react";
import {
  AlertTriangle,
  AlertCircle,
  Info,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  FileWarning,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/store/useForgeStore";
import type { ConsistencyIssue } from "@/store/useForgeStore";

const SEVERITY_CONFIG = {
  high: {
    label: "High",
    icon: AlertCircle,
    className: "bg-error/15 text-error",
    dotClass: "bg-error",
  },
  medium: {
    label: "Medium",
    icon: AlertTriangle,
    className: "bg-warning/15 text-warning",
    dotClass: "bg-warning",
  },
  low: {
    label: "Low",
    icon: Info,
    className: "bg-info/15 text-info",
    dotClass: "bg-info",
  },
} as const;

interface IssueCardProps {
  issue: ConsistencyIssue;
}

function IssueCard({ issue }: IssueCardProps) {
  const [expanded, setExpanded] = useState(false);
  const severity = SEVERITY_CONFIG[issue.severity];
  const SeverityIcon = severity.icon;

  return (
    <div className="card overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-4 py-3 flex items-start gap-3 hover:bg-kiln-700/30 transition-colors"
      >
        <SeverityIcon
          size={16}
          className={cn("mt-0.5 shrink-0", severity.className.split(" ")[1])}
        />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-2xs font-medium text-kiln-400 uppercase tracking-wide">
              {issue.type}
            </span>
            <span
              className={cn(
                "px-1.5 py-0.5 rounded text-2xs font-medium",
                severity.className,
              )}
            >
              {severity.label}
            </span>
          </div>
          <p className="text-sm text-kiln-200 leading-relaxed">
            {issue.description}
          </p>
        </div>

        <span className="text-kiln-500 mt-1 shrink-0">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
      </button>

      {expanded && (
        <div className="px-4 pb-3 animate-slide-down">
          <div className="ml-7">
            {/* Affected examples */}
            {issue.affected_example_ids.length > 0 && (
              <div className="mb-3">
                <span className="text-2xs font-medium text-kiln-500 uppercase tracking-wide">
                  Affected Examples ({issue.affected_example_ids.length})
                </span>
                <div className="mt-1 flex flex-wrap gap-1">
                  {issue.affected_example_ids.map((id) => (
                    <span key={id} className="badge-neutral text-2xs">
                      {id}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Suggested fix */}
            {issue.suggested_fix && (
              <div>
                <span className="text-2xs font-medium text-kiln-500 uppercase tracking-wide">
                  Suggested Fix
                </span>
                <p className="text-sm text-kiln-400 mt-1 leading-relaxed">
                  {issue.suggested_fix}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function ConsistencyReport() {
  const {
    consistencyIssues,
    consistencyCheckedAt,
    selectedDisciplineId,
    setConsistencyIssues,
  } = useForgeStore();
  const [isRunning, setIsRunning] = useState(false);

  const bySeverity = {
    high: consistencyIssues.filter((i) => i.severity === "high").length,
    medium: consistencyIssues.filter((i) => i.severity === "medium").length,
    low: consistencyIssues.filter((i) => i.severity === "low").length,
  };

  const handleRunCheck = () => {
    setIsRunning(true);
    // Simulate consistency check (will be replaced by real API call)
    setTimeout(() => {
      const demoIssues: ConsistencyIssue[] = [
        {
          id: "issue-1",
          type: "Duplicate Content",
          severity: "high",
          description:
            "Two examples have nearly identical questions with different answers, creating contradictory training signals.",
          affected_example_ids: ["ex-001", "ex-015"],
          suggested_fix:
            "Merge the examples or differentiate the questions to cover distinct aspects of the topic.",
        },
        {
          id: "issue-2",
          type: "Coverage Gap",
          severity: "medium",
          description:
            'The "Fault Isolation" competency has only 2 examples against a target of 15. This will result in weak model performance.',
          affected_example_ids: [],
          suggested_fix:
            "Add more examples covering common fault isolation scenarios, starting with the most frequent equipment failures.",
        },
        {
          id: "issue-3",
          type: "Style Inconsistency",
          severity: "low",
          description:
            "Answer length varies significantly across examples. Some are 2 sentences while others are 2 paragraphs. Normalize for consistent training signal.",
          affected_example_ids: ["ex-003", "ex-007", "ex-012", "ex-019"],
          suggested_fix:
            "Establish a target answer length (3-5 sentences) and adjust outliers to match.",
        },
        {
          id: "issue-4",
          type: "Missing Context",
          severity: "medium",
          description:
            "Several examples reference specific equipment models without providing context about what type of equipment it is.",
          affected_example_ids: ["ex-005", "ex-008"],
          suggested_fix:
            "Add context fields explaining the equipment type and its role in the maintenance workflow.",
        },
      ];
      setConsistencyIssues(demoIssues, new Date().toISOString());
      setIsRunning(false);
    }, 2000);
  };

  if (!selectedDisciplineId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <FileWarning
            size={32}
            className="mx-auto mb-3 text-kiln-500"
            strokeWidth={1.5}
          />
          <p className="text-sm text-kiln-400">
            Select a discipline to check consistency
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-kiln-600">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-medium text-kiln-200">
              Consistency Analysis
            </h3>
            {consistencyCheckedAt && (
              <p className="text-2xs text-kiln-500">
                Last checked: {new Date(consistencyCheckedAt).toLocaleString()}
              </p>
            )}
          </div>
          <button
            onClick={handleRunCheck}
            disabled={isRunning}
            className={cn("btn-secondary btn-sm", isRunning && "opacity-60")}
          >
            <RefreshCw size={13} className={cn(isRunning && "animate-spin")} />
            {isRunning ? "Checking..." : "Run Check"}
          </button>
        </div>

        {/* Summary cards */}
        {consistencyIssues.length > 0 && (
          <div className="grid grid-cols-3 gap-2">
            {(["high", "medium", "low"] as const).map((sev) => {
              const config = SEVERITY_CONFIG[sev];
              return (
                <div
                  key={sev}
                  className={cn(
                    "rounded-kiln px-3 py-2 flex items-center gap-2",
                    config.className,
                  )}
                >
                  <div
                    className={cn("w-2 h-2 rounded-full", config.dotClass)}
                  />
                  <span className="text-2xs font-medium">
                    {bySeverity[sev]} {config.label}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Issue list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {consistencyIssues.length === 0 ? (
          <div className="text-center py-12">
            {consistencyCheckedAt ? (
              <>
                <div className="w-12 h-12 rounded-full bg-success/10 flex items-center justify-center mx-auto mb-3">
                  <AlertTriangle size={20} className="text-success" />
                </div>
                <p className="text-sm text-kiln-300 mb-1">No issues found</p>
                <p className="text-2xs text-kiln-500">
                  Your curriculum passed all consistency checks.
                </p>
              </>
            ) : (
              <>
                <FileWarning
                  size={28}
                  className="mx-auto mb-3 text-kiln-500"
                  strokeWidth={1.5}
                />
                <p className="text-sm text-kiln-400 mb-1">
                  No consistency check run yet
                </p>
                <p className="text-2xs text-kiln-500">
                  Click "Run Check" to analyze your curriculum for issues.
                </p>
              </>
            )}
          </div>
        ) : (
          consistencyIssues.map((issue) => (
            <IssueCard key={issue.id} issue={issue} />
          ))
        )}
      </div>
    </div>
  );
}
