/**
 * Diagnostic Dashboard - Main interface for chunk problem detection
 *
 * MVP Phase 1 (Weeks 1-2):
 * - Upload PDF + chunks
 * - Visualize chunks overlaid on document
 * - Manual problem annotation (<60 seconds)
 * - Export diagnostic report
 */

import { useState } from 'react';
import { AlertTriangle, CheckCircle, XCircle, FileText, Loader2, Zap } from 'lucide-react';
import type { Document, Chunk, ChunkProblem, FixPlan, DiagnosticStatistics } from '../../api/chonk';
import { diagnosticAPI } from '../../api/chonk';
import { WorkflowChecklist } from '../WorkflowChecklist';
import { ChunkTreeView } from '../ChunkTreeView';

interface DiagnosticDashboardProps {
  document: Document;
}

const PROBLEM_TYPES = {
  semantic_incomplete: {
    label: 'Semantic Incompleteness',
    description: 'Chunk contains partial idea (dangling connectives, incomplete sentences)',
    color: 'text-red-400',
  },
  semantic_contamination: {
    label: 'Semantic Contamination',
    description: 'Chunk contains multiple unrelated ideas',
    color: 'text-orange-400',
  },
  structural_breakage: {
    label: 'Structural Breakage',
    description: 'Chunk splits logical unit (lists, tables, procedures)',
    color: 'text-yellow-400',
  },
  reference_orphaning: {
    label: 'Reference Orphaning',
    description: 'Chunk contains broken references ("see above", "as follows")',
    color: 'text-blue-400',
  },
  small_chunk: {
    label: 'Small Chunk',
    description: 'Chunk is too small to be meaningful',
    color: 'text-purple-400',
  },
  large_chunk: {
    label: 'Large Chunk',
    description: 'Chunk is too large and may contain multiple topics',
    color: 'text-cyan-400',
  },
};

export function DiagnosticDashboard({ document }: DiagnosticDashboardProps) {
  const [problems, setProblems] = useState<ChunkProblem[]>([]);
  const [statistics, setStatistics] = useState<DiagnosticStatistics | null>(null);
  const [selectedChunk, setSelectedChunk] = useState<Chunk | null>(null);
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);

  // Fix-related state
  const [fixPlan, setFixPlan] = useState<FixPlan | null>(null);
  const [isLoadingFixes, setIsLoadingFixes] = useState(false);
  const [isApplyingFixes, setIsApplyingFixes] = useState(false);
  const [fixError, setFixError] = useState<string | null>(null);
  const [fixSuccess, setFixSuccess] = useState<{
    problems_fixed: number;
    reduction_rate: number;
    chunks_before: number;
    chunks_after: number;
  } | null>(null);


  // Run diagnostics
  const runDiagnostics = async () => {
    console.log('[DIAGNOSTICS] Starting analysis for document:', document.id);
    setIsAnalyzing(true);
    setAnalyzeError(null);
    setFixSuccess(null);

    try {
      console.log('[DIAGNOSTICS] Calling API...');
      const result = await diagnosticAPI.analyze(document.id, true, true);
      console.log('[DIAGNOSTICS] Got results:', {
        problems: result.problems.length,
        statistics: result.statistics,
        questions_generated: result.questions_generated,
      });
      setProblems(result.problems);
      setStatistics(result.statistics);

      if (result.problems.length === 0) {
        console.log('[DIAGNOSTICS] No problems detected - document is clean!');
      }
    } catch (error) {
      console.error('[DIAGNOSTICS] Error:', error);
      setAnalyzeError(error instanceof Error ? error.message : 'Failed to run diagnostics');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Preview fixes
  const previewFixes = async () => {
    setIsLoadingFixes(true);
    setFixError(null);

    try {
      const result = await diagnosticAPI.previewFixes(document.id, true);
      setFixPlan(result.fix_plan);
    } catch (error) {
      console.error('Failed to preview fixes:', error);
      setFixError(error instanceof Error ? error.message : 'Failed to preview fixes');
    } finally {
      setIsLoadingFixes(false);
    }
  };

  // Apply fixes
  const applyFixes = async () => {
    setIsApplyingFixes(true);
    setFixError(null);

    try {
      const result = await diagnosticAPI.applyFixes(document.id, true, true);

      // Update problems with new diagnostics
      setProblems(result.after.problems ? [] : problems); // Will get real data from backend

      // Show success metrics
      setFixSuccess({
        problems_fixed: result.improvement.problems_fixed,
        reduction_rate: result.improvement.reduction_rate,
        chunks_before: result.fix_result.chunks_before,
        chunks_after: result.fix_result.chunks_after,
      });

      // Clear fix plan
      setFixPlan(null);

      // Re-run diagnostics to get updated problems
      setTimeout(() => runDiagnostics(), 1000);
    } catch (error) {
      console.error('Failed to apply fixes:', error);
      setFixError(error instanceof Error ? error.message : 'Failed to apply fixes');
    } finally {
      setIsApplyingFixes(false);
    }
  };

  // Stats
  const totalChunks = document.chunks.length;
  const problemChunks = statistics?.unique_chunks_with_problems || 0;
  const healthyChunks = totalChunks - problemChunks;

  // Group problems by severity
  const highSeverity = statistics?.by_severity?.high || 0;
  const mediumSeverity = statistics?.by_severity?.medium || 0;
  const lowSeverity = statistics?.by_severity?.low || 0;

  return (
    <div className="h-full flex flex-col bg-kiln-900">
      {/* Header */}
      <div className="border-b-4 border-black p-4 bg-kiln-800">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-pixel text-2xl text-kiln-300">DIAGNOSTIC DASHBOARD</h1>
            <p className="text-sm text-kiln-500 mt-1">
              {document.source_path} • {totalChunks} chunks
            </p>
          </div>
          <button
            onClick={runDiagnostics}
            disabled={isAnalyzing}
            className="btn btn-primary flex items-center gap-2"
          >
            {isAnalyzing && <Loader2 className="w-4 h-4 animate-spin" />}
            {isAnalyzing ? 'ANALYZING...' : 'RUN DIAGNOSTICS'}
          </button>
        </div>

        {/* Error Display */}
        {analyzeError && (
          <div className="mt-4 bg-red-900/30 border-2 border-red-400 p-3">
            <p className="text-sm text-red-400">Error: {analyzeError}</p>
          </div>
        )}

        {/* Success Metrics */}
        {fixSuccess && (
          <div className="mt-4 bg-green-900/30 border-2 border-green-400 p-3">
            <h3 className="text-pixel text-green-400 mb-2">FIXES APPLIED SUCCESSFULLY!</h3>
            <div className="grid grid-cols-4 gap-3 text-xs">
              <div>
                <div className="text-kiln-500">Problems Fixed</div>
                <div className="text-green-400 font-bold">{fixSuccess.problems_fixed}</div>
              </div>
              <div>
                <div className="text-kiln-500">Improvement</div>
                <div className="text-green-400 font-bold">{(fixSuccess.reduction_rate * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div className="text-kiln-500">Chunks Before</div>
                <div className="text-green-400 font-bold">{fixSuccess.chunks_before}</div>
              </div>
              <div>
                <div className="text-kiln-500">Chunks After</div>
                <div className="text-green-400 font-bold">{fixSuccess.chunks_after}</div>
              </div>
            </div>
          </div>
        )}

        {/* Stats Bar */}
        <div className="grid grid-cols-4 gap-4 mt-4">
          <div className="bg-kiln-900 border-2 border-black p-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <div>
                <div className="text-xs text-kiln-500">HEALTHY</div>
                <div className="text-pixel text-xl text-kiln-300">{healthyChunks}</div>
              </div>
            </div>
          </div>

          <div className="bg-kiln-900 border-2 border-black p-3">
            <div className="flex items-center gap-2">
              <XCircle className="w-5 h-5 text-red-400" />
              <div>
                <div className="text-xs text-kiln-500">HIGH</div>
                <div className="text-pixel text-xl text-red-400">{highSeverity}</div>
              </div>
            </div>
          </div>

          <div className="bg-kiln-900 border-2 border-black p-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-orange-400" />
              <div>
                <div className="text-xs text-kiln-500">MEDIUM</div>
                <div className="text-pixel text-xl text-orange-400">{mediumSeverity}</div>
              </div>
            </div>
          </div>

          <div className="bg-kiln-900 border-2 border-black p-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              <div>
                <div className="text-xs text-kiln-500">LOW</div>
                <div className="text-pixel text-xl text-yellow-400">{lowSeverity}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Fix Section */}
        {problems.length > 0 && !fixPlan && (
          <div className="mt-4">
            <button
              onClick={previewFixes}
              disabled={isLoadingFixes}
              className="btn btn-secondary flex items-center gap-2"
            >
              {isLoadingFixes && <Loader2 className="w-4 h-4 animate-spin" />}
              <Zap className="w-4 h-4" />
              {isLoadingFixes ? 'ANALYZING FIXES...' : 'PREVIEW AUTOMATIC FIXES'}
            </button>
          </div>
        )}

        {/* Fix Plan Preview */}
        {fixPlan && (
          <div className="mt-4 bg-ember/10 border-2 border-ember p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-pixel text-ember">AUTOMATIC FIX PLAN</h3>
              <button
                onClick={() => setFixPlan(null)}
                className="text-xs text-kiln-500 hover:text-kiln-300"
              >
                CLOSE
              </button>
            </div>

            {/* Fix Error */}
            {fixError && (
              <div className="mb-3 bg-red-900/30 border border-red-400 p-2">
                <p className="text-xs text-red-400">{fixError}</p>
              </div>
            )}

            {/* Stats */}
            <div className="grid grid-cols-3 gap-3 mb-3 text-xs">
              <div className="bg-kiln-900 border border-black p-2">
                <div className="text-kiln-500">Proposed Fixes</div>
                <div className="text-pixel text-kiln-300">{fixPlan.total_actions}</div>
              </div>
              <div className="bg-kiln-900 border border-black p-2">
                <div className="text-kiln-500">Est. Improvement</div>
                <div className="text-pixel text-green-400">
                  {(fixPlan.estimated_improvement * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-kiln-900 border border-black p-2">
                <div className="text-kiln-500">Conflicts</div>
                <div className="text-pixel text-kiln-300">{fixPlan.conflicts.length}</div>
              </div>
            </div>

            {/* Warnings */}
            {fixPlan.warnings.length > 0 && (
              <div className="mb-3 bg-yellow-900/20 border border-yellow-400 p-2">
                <div className="text-xs text-yellow-400 font-bold mb-1">Warnings:</div>
                {fixPlan.warnings.map((warning, i) => (
                  <div key={i} className="text-xs text-yellow-300">{warning}</div>
                ))}
              </div>
            )}

            {/* Actions List */}
            <div className="mb-3 max-h-48 overflow-y-auto">
              <div className="text-xs text-kiln-500 mb-2">PLANNED ACTIONS:</div>
              <div className="space-y-2">
                {fixPlan.actions.slice(0, 10).map((action, i) => (
                  <div key={i} className="bg-kiln-900 border border-black p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-bold text-ember">
                        {action.action_type.toUpperCase()}
                      </span>
                      <span className="text-xs text-green-400">
                        {(action.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    <p className="text-xs text-kiln-500">{action.description}</p>
                  </div>
                ))}
                {fixPlan.actions.length > 10 && (
                  <div className="text-xs text-kiln-500 text-center">
                    +{fixPlan.actions.length - 10} more actions...
                  </div>
                )}
              </div>
            </div>

            {/* Apply Button */}
            <button
              onClick={applyFixes}
              disabled={isApplyingFixes}
              className="btn btn-primary w-full flex items-center justify-center gap-2"
            >
              {isApplyingFixes && <Loader2 className="w-4 h-4 animate-spin" />}
              {isApplyingFixes ? 'APPLYING FIXES...' : 'APPLY FIXES'}
            </button>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex min-h-0">
        {/* Left Sidebar - Workflow + Problems/Overview */}
        <div className="w-1/3 border-r-4 border-black bg-kiln-800 overflow-y-auto flex flex-col">
          {/* Workflow Checklist */}
          <div className="p-4 border-b-4 border-black">
            <WorkflowChecklist
              hasDocument={!!document}
              hasProblems={problems.length > 0}
              hasFixPlan={!!fixPlan}
              hasAppliedFixes={!!fixSuccess}
            />
          </div>

          {/* Show Problems if detected, otherwise show hint */}
          {problems.length > 0 ? (
            <div className="flex-1 overflow-y-auto p-4">
              <h2 className="text-pixel text-lg text-kiln-300 mb-4">DETECTED PROBLEMS</h2>
              <div className="space-y-2">
                {problems.map((problem) => {
                  const typeInfo = PROBLEM_TYPES[problem.problem_type as keyof typeof PROBLEM_TYPES] || {
                    label: problem.problem_type,
                    color: 'text-kiln-500',
                  };
                  const chunk = document.chunks.find(c => c.id === problem.chunk_id);

                  return (
                    <div
                      key={`${problem.chunk_id}-${problem.problem_type}`}
                      onClick={() => setSelectedChunk(chunk || null)}
                      className={`
                        border-2 border-black p-3 cursor-pointer
                        hover:bg-kiln-900 transition-colors
                        ${selectedChunk?.id === problem.chunk_id ? 'bg-ember/10' : 'bg-kiln-900'}
                      `}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className={`text-xs font-bold ${typeInfo.color}`}>
                          {typeInfo.label}
                        </div>
                        <div className={`
                          text-xs px-2 py-1 border border-black
                          ${problem.severity === 'high' ? 'bg-red-900/30 text-red-400' : ''}
                          ${problem.severity === 'medium' ? 'bg-orange-900/30 text-orange-400' : ''}
                          ${problem.severity === 'low' ? 'bg-yellow-900/30 text-yellow-400' : ''}
                        `}>
                          {problem.severity.toUpperCase()}
                        </div>
                      </div>
                      <p className="text-xs text-kiln-500 mb-2">{problem.description}</p>
                      {chunk && (
                        <p className="text-xs text-kiln-500 truncate">
                          Chunk: {chunk.content.substring(0, 60)}...
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-4">
              <h2 className="text-pixel text-lg text-kiln-300 mb-4">BEFORE DIAGNOSTICS</h2>

              {!statistics ? (
                // Haven't run diagnostics yet
                <div className="text-center py-8">
                  <FileText className="w-12 h-12 text-kiln-500 mx-auto mb-2" />
                  <p className="text-kiln-500 mb-2">Ready to analyze</p>
                  <p className="text-xs text-kiln-500 mb-4">Click "RUN DIAGNOSTICS" above to check for problems</p>
                  <div className="bg-kiln-900 border border-black p-4 text-left max-w-md mx-auto">
                    <p className="text-xs text-kiln-300 mb-2">
                      <strong>Diagnostics will check for:</strong>
                    </p>
                    <ul className="text-xs text-kiln-500 space-y-1">
                      <li>• Incomplete sentences and dangling connectives</li>
                      <li>• Chunks that are too small or too large</li>
                      <li>• Split lists, tables, and procedures</li>
                      <li>• Broken cross-references</li>
                      <li>• Mixed topics in single chunks</li>
                    </ul>
                  </div>
                </div>
              ) : (
                // Diagnostics ran but found no problems
                <div className="text-center py-8">
                  <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-3" />
                  <p className="text-green-400 font-bold mb-2">Great News!</p>
                  <p className="text-sm text-kiln-500 mb-4">No major problems detected in your chunks</p>
                  <div className="bg-green-900/20 border border-green-400 p-4 text-left max-w-md mx-auto">
                    <p className="text-xs text-green-300 mb-2">
                      <strong>What this means:</strong>
                    </p>
                    <ul className="text-xs text-kiln-500 space-y-1">
                      <li>✓ Chunks have good size distribution</li>
                      <li>✓ Sentences are complete</li>
                      <li>✓ No obvious structural breaks</li>
                      <li>✓ References appear intact</li>
                    </ul>
                    <p className="text-xs text-kiln-500 mt-3">
                      Your document was likely well-structured to begin with. You can still test retrieval quality with custom queries in the Test panel.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Side - Document Overview or Chunk Details */}
        <div className="flex-1 overflow-hidden">
          {selectedChunk ? (
            // Show selected chunk details
            <div className="h-full overflow-y-auto p-4">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setSelectedChunk(null)}
                    className="btn btn-secondary text-sm"
                  >
                    ← BACK TO CHUNKS
                  </button>
                  <h2 className="text-pixel text-lg text-kiln-300">CHUNK DETAILS</h2>
                </div>
                <button
                  onClick={() => setIsAnnotating(!isAnnotating)}
                  className="btn btn-secondary text-sm"
                >
                  {isAnnotating ? 'CANCEL ANNOTATION' : 'ANNOTATE PROBLEM'}
                </button>
              </div>

              {/* Chunk Info */}
              <div className="bg-kiln-800 border-2 border-black p-4 mb-4">
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div>
                    <div className="text-xs text-kiln-500">TOKENS</div>
                    <div className="text-pixel text-kiln-300">{selectedChunk.token_count}</div>
                  </div>
                  <div>
                    <div className="text-xs text-kiln-500">QUALITY</div>
                    <div className="text-pixel text-kiln-300">
                      {(selectedChunk.quality.overall * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-kiln-500">PATH</div>
                    <div className="text-xs text-kiln-300 truncate">{selectedChunk.hierarchy_path}</div>
                  </div>
                </div>

                <div className="border-t-2 border-black pt-4">
                  <div className="text-xs text-kiln-500 mb-2">CONTENT</div>
                  <div className="text-sm text-kiln-300 font-mono whitespace-pre-wrap max-h-96 overflow-y-auto p-3 bg-kiln-900 border border-black">
                    {selectedChunk.content}
                  </div>
                </div>
              </div>

              {/* Annotation Form */}
              {isAnnotating && (
                <div className="bg-error/10 border-2 border-error p-4">
                  <h3 className="text-pixel text-error mb-4">ANNOTATE PROBLEM</h3>

                  <div className="space-y-4">
                    <div>
                      <label className="text-xs text-kiln-500 block mb-2">PROBLEM TYPE</label>
                      <select className="w-full bg-kiln-900 border-2 border-black p-2 text-sm text-kiln-300">
                        {Object.entries(PROBLEM_TYPES).map(([key, info]) => (
                          <option key={key} value={key}>{info.label}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="text-xs text-kiln-500 block mb-2">SEVERITY</label>
                      <select className="w-full bg-kiln-900 border-2 border-black p-2 text-sm text-kiln-300">
                        <option value="high">HIGH</option>
                        <option value="medium">MEDIUM</option>
                        <option value="low">LOW</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-xs text-kiln-500 block mb-2">DESCRIPTION</label>
                      <textarea
                        className="w-full bg-kiln-900 border-2 border-black p-2 text-sm text-kiln-300 font-mono"
                        rows={3}
                        placeholder="Describe the problem..."
                      />
                    </div>

                    <div>
                      <label className="text-xs text-kiln-500 block mb-2">SUGGESTED FIX (Optional)</label>
                      <textarea
                        className="w-full bg-kiln-900 border-2 border-black p-2 text-sm text-kiln-300 font-mono"
                        rows={2}
                        placeholder="How should this be fixed?"
                      />
                    </div>

                    <div className="flex gap-2">
                      <button className="btn btn-primary flex-1">SAVE ANNOTATION</button>
                      <button
                        onClick={() => setIsAnnotating(false)}
                        className="btn btn-secondary"
                      >
                        CANCEL
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Existing Problems for this Chunk */}
              <div className="mt-4">
                <h3 className="text-pixel text-sm text-kiln-300 mb-2">PROBLEMS FOR THIS CHUNK</h3>
                {problems.filter(p => p.chunk_id === selectedChunk.id).length === 0 ? (
                  <p className="text-xs text-kiln-500">No problems detected</p>
                ) : (
                  <div className="space-y-2">
                    {problems.filter(p => p.chunk_id === selectedChunk.id).map((problem, idx) => {
                      const typeInfo = PROBLEM_TYPES[problem.problem_type as keyof typeof PROBLEM_TYPES] || {
                        label: problem.problem_type,
                        color: 'text-kiln-500',
                      };
                      return (
                        <div key={idx} className="bg-kiln-800 border border-black p-3">
                          <div className={`text-xs font-bold ${typeInfo.color} mb-1`}>
                            {typeInfo.label} ({problem.severity})
                          </div>
                          <p className="text-xs text-kiln-500">{problem.description}</p>
                          {problem.suggested_fix && (
                            <div className="mt-2 text-xs">
                              <span className="text-green-400">Fix: </span>
                              <span className="text-kiln-500">{problem.suggested_fix}</span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
            </div>
          ) : (
            // Show document overview when nothing selected
            <ChunkTreeView
              chunks={document.chunks}
              onSelectChunk={(chunk) => setSelectedChunk(chunk)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
