import { useState } from 'react';
import { PanelRightClose, FileText, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { Citation } from '@/store/useHearthStore';

interface CitationPanelProps {
  citations: Citation[];
  activeCitationId?: string | null;
  onClose: () => void;
}

function relevanceBadge(score: number) {
  if (score >= 0.8) return { label: 'High', className: 'badge-success' };
  if (score >= 0.5) return { label: 'Medium', className: 'badge-warning' };
  return { label: 'Low', className: 'badge-neutral' };
}

function CitationCard({
  citation,
  index,
  isActive,
}: {
  citation: Citation;
  index: number;
  isActive: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const badge = relevanceBadge(citation.relevance_score);

  return (
    <div
      className={cn(
        'rounded-md border transition-all duration-200',
        isActive
          ? 'bg-hearth-glow/5 border-hearth-glow/20'
          : 'bg-kiln-800/50 border-kiln-600/40 hover:border-kiln-500/40',
      )}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-3 py-2.5"
      >
        <div className="flex items-start gap-2.5">
          {/* Citation number */}
          <span
            className={cn(
              'flex-shrink-0 w-5 h-5 rounded text-[10px] font-mono font-bold',
              'flex items-center justify-center',
              isActive
                ? 'bg-hearth-glow/20 text-hearth-glow'
                : 'bg-kiln-700 text-kiln-400',
            )}
          >
            {index + 1}
          </span>

          <div className="flex-1 min-w-0">
            {/* Document title */}
            <div className="flex items-center gap-2 mb-1">
              <FileText size={12} className="text-kiln-500 flex-shrink-0" />
              <span className="text-xs font-medium text-kiln-200 truncate">
                {citation.document_title}
              </span>
            </div>

            {/* Section + page */}
            <div className="flex items-center gap-2 text-2xs text-kiln-500">
              <span className="truncate">{citation.section}</span>
              <span className="flex-shrink-0">p.{citation.page}</span>
              <span className={badge.className}>{badge.label}</span>
            </div>
          </div>

          {/* Expand arrow */}
          {expanded ? (
            <ChevronDown size={14} className="text-kiln-500 flex-shrink-0 mt-0.5" />
          ) : (
            <ChevronRight size={14} className="text-kiln-500 flex-shrink-0 mt-0.5" />
          )}
        </div>
      </button>

      {/* Expanded snippet */}
      {expanded && (
        <div className="px-3 pb-3 pt-0">
          <div className="ml-7.5 pl-2 border-l-2 border-kiln-600/50">
            <p className="text-xs text-kiln-400 leading-relaxed whitespace-pre-wrap">
              {citation.snippet}
            </p>
            <div className="mt-2 flex items-center gap-1.5">
              <span className="text-2xs text-kiln-500 font-mono">
                Relevance: {(citation.relevance_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function CitationPanel({ citations, activeCitationId, onClose }: CitationPanelProps) {
  return (
    <div className="w-80 flex flex-col h-full bg-kiln-800/50 border-l border-kiln-600/50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-kiln-600/50">
        <div className="flex items-center gap-2">
          <FileText size={14} className="text-hearth-glow" />
          <span className="font-display text-xs font-semibold text-kiln-300 uppercase tracking-wider">
            Sources
          </span>
          {citations.length > 0 && (
            <span className="text-2xs text-kiln-500 bg-kiln-700 px-1.5 py-0.5 rounded-full">
              {citations.length}
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1 text-kiln-500 hover:text-kiln-300 transition-colors rounded"
          title="Close panel"
        >
          <PanelRightClose size={16} />
        </button>
      </div>

      {/* Citation list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {citations.length === 0 ? (
          <div className="text-center py-12">
            <FileText size={24} className="mx-auto mb-3 text-kiln-600" />
            <p className="text-xs text-kiln-500">
              Citations will appear here when<br />the assistant references documents.
            </p>
          </div>
        ) : (
          citations.map((citation, i) => (
            <CitationCard
              key={citation.id}
              citation={citation}
              index={i}
              isActive={citation.id === activeCitationId}
            />
          ))
        )}
      </div>
    </div>
  );
}
