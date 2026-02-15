/**
 * WorkflowPanel - The new CHONK workflow
 *
 * DROP FILE → SEE HIERARCHY → CHOOSE STRATEGY → COMPARE → TEST → EXPORT
 */

import { useEffect, useState } from 'react';
import { useStore } from '../store/useStore';
import { hierarchyAPI, comparisonAPI, queryTestAPI } from '../api/chonk';
import { TreeView } from './HierarchyTree';
import { StrategyPicker, ParameterPanel } from './StrategySelector';
import { SideBySide, RecommendationBox } from './ComparisonDashboard';
import { QueryTester } from './QueryTester';
import type { QueryResult } from './QueryTester';

type WorkflowStep = 'hierarchy' | 'strategy' | 'compare' | 'test' | 'export';

export function WorkflowPanel() {
  const {
    selectedDocumentId,
    hierarchyTree,
    setHierarchyTree,
    setHierarchyLoading,
    selectedNodeId,
    selectNode,
    selectedStrategy,
    setStrategy,
    chunkingParameters,
    setParameters,
    resetParameters,
    comparisonResults,
    setComparisonResults,
    setComparing,
    recommendation,
    setRecommendation,
  } = useStore();

  const [currentStep, setCurrentStep] = useState<WorkflowStep>('hierarchy');
  const [error, setError] = useState<string | null>(null);

  // Load hierarchy when document is selected
  useEffect(() => {
    if (selectedDocumentId && !hierarchyTree) {
      loadHierarchy();
    }
  }, [selectedDocumentId]);

  const loadHierarchy = async () => {
    if (!selectedDocumentId) return;

    setHierarchyLoading(true);
    setError(null);
    try {
      const tree = await hierarchyAPI.build(selectedDocumentId);
      setHierarchyTree(tree);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to build hierarchy');
    } finally {
      setHierarchyLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!selectedDocumentId) return;

    setComparing(true);
    setError(null);
    try {
      const strategies = [
        {
          name: 'hierarchical',
          config: {
            ...chunkingParameters,
            group_under_headings: true,
          },
        },
        {
          name: 'fixed',
          config: {
            target_tokens: chunkingParameters.max_tokens,
            overlap_tokens: chunkingParameters.overlap_tokens,
          },
        },
        {
          name: 'semantic',
          config: {
            target_tokens: chunkingParameters.max_tokens,
          },
        },
      ];

      const result = await comparisonAPI.compare(selectedDocumentId, strategies);
      setComparisonResults(result.strategies);
      setRecommendation(result.recommendation);
      setCurrentStep('compare');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compare strategies');
    } finally {
      setComparing(false);
    }
  };

  const handleQueryTest = async (query: string): Promise<QueryResult[]> => {
    if (!selectedDocumentId) return [];

    try {
      const result = await queryTestAPI.testQuery(
        query,
        [selectedStrategy],
        selectedDocumentId
      );

      // Transform StrategyQueryResult to QueryResult format
      const strategyResult = result.strategies[0];
      return strategyResult.results.map((r) => ({
        query,
        retrieved_chunks: strategyResult.retrieved_count,
        top_chunk_id: r.chunk_id,
        top_score: r.score,
        chunk_preview: r.content_preview,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to test query');
      return [];
    }
  };

  const handleSelectStrategy = (strategy: string) => {
    setStrategy(strategy as any);
  };

  if (!selectedDocumentId) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400">
        <div className="text-center">
          <p className="text-lg mb-2">No document selected</p>
          <p className="text-sm">Upload a document to begin</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {/* Workflow Steps */}
      <div className="flex border-b border-gray-700">
        {STEPS.map((step, index) => (
          <button
            key={step.id}
            onClick={() => setCurrentStep(step.id)}
            className={`flex-1 px-4 py-3 text-sm font-medium border-r border-gray-700 transition-colors ${
              currentStep === step.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <span className="text-xs opacity-60">{index + 1}.</span>
              <span>{step.label}</span>
            </div>
          </button>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border-b border-red-700 px-4 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

      {/* Step Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Step 1: Hierarchy */}
        {currentStep === 'hierarchy' && hierarchyTree && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">Document Structure</h2>
              <button
                onClick={() => setCurrentStep('strategy')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium"
              >
                Next: Choose Strategy →
              </button>
            </div>
            <TreeView
              tree={hierarchyTree}
              onNodeSelect={(node) => selectNode(node.section_id)}
              selectedNodeId={selectedNodeId ?? undefined}
            />
          </div>
        )}

        {/* Step 2: Strategy */}
        {currentStep === 'strategy' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">Choose Chunking Strategy</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setCurrentStep('hierarchy')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-medium"
                >
                  ← Back
                </button>
                <button
                  onClick={handleCompare}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium"
                >
                  Next: Compare Strategies →
                </button>
              </div>
            </div>

            <StrategyPicker
              selected={selectedStrategy}
              onSelect={setStrategy}
            />

            <ParameterPanel
              strategy={selectedStrategy}
              parameters={chunkingParameters}
              onChange={(params) => setParameters(params)}
            />
          </div>
        )}

        {/* Step 3: Compare */}
        {currentStep === 'compare' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">Compare Strategies</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setCurrentStep('strategy')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-medium"
                >
                  ← Back
                </button>
                <button
                  onClick={() => setCurrentStep('test')}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium"
                  disabled={comparisonResults.length === 0}
                >
                  Next: Test Queries →
                </button>
              </div>
            </div>

            {comparisonResults.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <p>Click "Compare Strategies" to see results</p>
              </div>
            ) : (
              <>
                {recommendation && (
                  <RecommendationBox
                    results={comparisonResults}
                    recommendation={recommendation}
                  />
                )}
                <SideBySide
                  results={comparisonResults}
                  onSelectStrategy={handleSelectStrategy}
                />
              </>
            )}
          </div>
        )}

        {/* Step 4: Test */}
        {currentStep === 'test' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">Test Queries</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setCurrentStep('compare')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-medium"
                >
                  ← Back
                </button>
                <button
                  onClick={() => setCurrentStep('export')}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded font-medium"
                >
                  Next: Export →
                </button>
              </div>
            </div>

            <QueryTester
              strategies={[selectedStrategy]}
              onRunQuery={handleQueryTest}
            />
          </div>
        )}

        {/* Step 5: Export */}
        {currentStep === 'export' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">Export Chunks</h2>
              <button
                onClick={() => setCurrentStep('test')}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-medium"
              >
                ← Back
              </button>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="font-bold mb-4">Ready to Export</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Strategy:</span>
                  <span className="font-mono capitalize">{selectedStrategy}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Max Tokens:</span>
                  <span className="font-mono">{chunkingParameters.max_tokens}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Overlap:</span>
                  <span className="font-mono">{chunkingParameters.overlap_tokens}</span>
                </div>
              </div>

              <div className="mt-6 flex gap-3">
                <button className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded font-medium">
                  Export as JSONL
                </button>
                <button className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded font-medium">
                  Export as JSON
                </button>
                <button className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded font-medium">
                  Export as CSV
                </button>
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-900 to-blue-900 rounded-lg p-6 border border-green-500">
              <div className="text-4xl mb-2">✅</div>
              <h3 className="font-bold text-lg mb-2">Workflow Complete!</h3>
              <p className="text-sm text-gray-300">
                You've successfully visualized, configured, compared, and tested your chunking strategy.
                Export your chunks when ready!
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const STEPS = [
  { id: 'hierarchy' as WorkflowStep, label: 'Hierarchy' },
  { id: 'strategy' as WorkflowStep, label: 'Strategy' },
  { id: 'compare' as WorkflowStep, label: 'Compare' },
  { id: 'test' as WorkflowStep, label: 'Test' },
  { id: 'export' as WorkflowStep, label: 'Export' },
];
