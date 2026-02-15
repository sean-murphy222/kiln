import { useState, useCallback } from 'react';
import { Search, Play, Plus, Check, X, Clock } from 'lucide-react';
import { useStore } from '../store/useStore';
import { testAPI, projectAPI } from '../api/chonk';
import type { SearchResult } from '../api/chonk';

export function TestPanel() {
  const {
    project,
    searchQuery,
    searchResults,
    isSearching,
    setSearchQuery,
    setSearchResults,
    setSearching,
    setProject,
    setError,
    selectChunk,
    selectedDocumentId,
  } = useStore();

  const [testSuiteName, setTestSuiteName] = useState('');
  const [showCreateSuite, setShowCreateSuite] = useState(false);

  // Handle search
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;

    setSearching(true);
    try {
      const result = await testAPI.search(
        searchQuery,
        5,
        selectedDocumentId ? [selectedDocumentId] : undefined
      );
      setSearchResults(result.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setSearching(false);
    }
  }, [searchQuery, selectedDocumentId, setSearchResults, setSearching, setError]);

  // Handle create test suite
  const handleCreateSuite = async () => {
    if (!testSuiteName.trim()) return;

    try {
      await testAPI.createSuite(testSuiteName.trim());
      const updatedProject = await projectAPI.get();
      setProject(updatedProject);
      setTestSuiteName('');
      setShowCreateSuite(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create test suite');
    }
  };

  return (
    <div className="h-full flex flex-col bg-surface-panel">
      {/* Header */}
      <div className="panel-header">Test Retrieval</div>

      {/* Search input */}
      <div className="p-3 border-b border-chonk-slate">
        <div className="relative">
          <input
            type="text"
            className="input-pixel pr-10"
            placeholder="Type a test query..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-chonk-gray hover:text-accent-primary"
            onClick={handleSearch}
            disabled={isSearching}
          >
            {isSearching ? (
              <Clock size={18} className="animate-spin" />
            ) : (
              <Search size={18} />
            )}
          </button>
        </div>
        <p className="text-xs text-chonk-gray mt-2">
          {selectedDocumentId
            ? 'Searching current document'
            : 'Searching all documents'}
        </p>
      </div>

      {/* Search results */}
      <div className="flex-1 overflow-y-auto">
        {searchResults.length > 0 ? (
          <div className="p-2 space-y-2">
            {searchResults.map((result) => (
              <SearchResultCard
                key={result.chunk_id}
                result={result}
                onClick={() => selectChunk(result.chunk_id)}
              />
            ))}
          </div>
        ) : searchQuery && !isSearching ? (
          <div className="p-4 text-center text-chonk-gray text-sm">
            No results found
          </div>
        ) : (
          <div className="p-4 text-center text-chonk-gray text-sm">
            <p>Test your chunks before export</p>
            <p className="mt-2 text-xs">
              Type a query to see which chunks match
            </p>
          </div>
        )}
      </div>

      {/* Test suites section */}
      <div className="border-t border-chonk-slate">
        <div className="panel-header flex items-center justify-between">
          <span>Test Suites</span>
          <button
            className="text-chonk-gray hover:text-accent-primary"
            onClick={() => setShowCreateSuite(!showCreateSuite)}
          >
            <Plus size={14} />
          </button>
        </div>

        {showCreateSuite && (
          <div className="p-3 border-b border-chonk-slate">
            <input
              type="text"
              className="input-pixel text-sm mb-2"
              placeholder="Suite name..."
              value={testSuiteName}
              onChange={(e) => setTestSuiteName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreateSuite()}
              autoFocus
            />
            <div className="flex gap-2">
              <button
                className="btn-pixel text-xs flex-1"
                onClick={() => setShowCreateSuite(false)}
              >
                Cancel
              </button>
              <button
                className="btn-pixel-primary text-xs flex-1"
                onClick={handleCreateSuite}
              >
                Create
              </button>
            </div>
          </div>
        )}

        <div className="p-2 max-h-40 overflow-y-auto">
          {project?.test_suites.length === 0 ? (
            <p className="text-xs text-chonk-gray text-center py-2">
              No test suites yet
            </p>
          ) : (
            <div className="space-y-1">
              {project?.test_suites.map((suite) => (
                <TestSuiteItem key={suite.id} suite={suite} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface SearchResultCardProps {
  result: SearchResult;
  onClick: () => void;
}

function SearchResultCard({ result, onClick }: SearchResultCardProps) {
  const scoreColor =
    result.score >= 0.8
      ? 'text-accent-success'
      : result.score >= 0.65
      ? 'text-accent-warning'
      : 'text-accent-error';

  const scoreBg =
    result.score >= 0.8
      ? 'bg-accent-success/20'
      : result.score >= 0.65
      ? 'bg-accent-warning/20'
      : 'bg-accent-error/20';

  return (
    <div
      className="card-pixel p-3 cursor-pointer hover:bg-surface-card transition-colors"
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-mono font-bold ${scoreColor}`}>
            {(result.score * 100).toFixed(0)}%
          </span>
          <span className={`text-xs px-1.5 py-0.5 rounded ${scoreBg} ${scoreColor}`}>
            #{result.rank}
          </span>
        </div>
        {result.page_range && (
          <span className="text-xs text-chonk-gray">
            p.{result.page_range[0]}
            {result.page_range[0] !== result.page_range[1] &&
              `-${result.page_range[1]}`}
          </span>
        )}
      </div>

      {/* Hierarchy path */}
      {result.hierarchy_path && (
        <p className="text-xs text-chonk-light mb-1 truncate">
          {result.hierarchy_path}
        </p>
      )}

      {/* Content preview */}
      <p className="text-xs text-chonk-gray line-clamp-2">
        {result.content_preview}
      </p>

      {/* Document name */}
      {result.document_name && (
        <p className="text-xs text-chonk-slate mt-1">
          {result.document_name}
        </p>
      )}
    </div>
  );
}

interface TestSuiteItemProps {
  suite: {
    id: string;
    name: string;
    queries: Array<{ id: string }>;
  };
}

function TestSuiteItem({ suite }: TestSuiteItemProps) {
  const { setError } = useStore();
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<{
    passed: number;
    failed: number;
  } | null>(null);

  const handleRun = async () => {
    setIsRunning(true);
    setResult(null);
    try {
      const report = await testAPI.runSuite(suite.id);
      setResult({
        passed: report.passed_count,
        failed: report.failed_count,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run test suite');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-surface-card">
      <span className="flex-1 text-sm text-chonk-light truncate">
        {suite.name}
      </span>
      <span className="text-xs text-chonk-gray">
        {suite.queries.length}
      </span>

      {result && (
        <div className="flex items-center gap-1">
          {result.passed > 0 && (
            <span className="flex items-center gap-0.5 text-xs text-accent-success">
              <Check size={10} />
              {result.passed}
            </span>
          )}
          {result.failed > 0 && (
            <span className="flex items-center gap-0.5 text-xs text-accent-error">
              <X size={10} />
              {result.failed}
            </span>
          )}
        </div>
      )}

      <button
        className="p-1 text-chonk-gray hover:text-accent-primary"
        onClick={handleRun}
        disabled={isRunning || suite.queries.length === 0}
        title="Run test suite"
      >
        {isRunning ? (
          <Clock size={14} className="animate-spin" />
        ) : (
          <Play size={14} />
        )}
      </button>
    </div>
  );
}
