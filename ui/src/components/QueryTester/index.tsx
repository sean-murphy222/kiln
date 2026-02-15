/**
 * QueryTester - KILLER FEATURE
 *
 * Test retrieval queries before embedding to validate chunking strategy.
 */

import React, { useState } from 'react';

export interface QueryResult {
  query: string;
  retrieved_chunks: number;
  top_chunk_id: string | null;
  top_score: number;
  chunk_preview?: string;
}

interface QueryTesterProps {
  onRunQuery: (query: string) => Promise<QueryResult[]>;
  strategies: string[];
}

export const QueryTester: React.FC<QueryTesterProps> = ({
  onRunQuery,
  strategies,
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Record<string, QueryResult[]>>({});
  const [isLoading, setIsLoading] = useState(false);

  const handleRunQuery = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      // Run query against all strategies
      const strategyResults: Record<string, QueryResult[]> = {};
      for (const strategy of strategies) {
        strategyResults[strategy] = await onRunQuery(query);
      }
      setResults(strategyResults);
    } catch (error) {
      console.error('Query failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900 to-pink-900 rounded-lg p-4 border border-purple-500">
        <h2 className="text-xl font-bold mb-2">🔍 Test Before You Embed</h2>
        <p className="text-sm text-gray-300">
          See which strategy retrieves better results for your queries.
          No embeddings cost until you're confident!
        </p>
      </div>

      {/* Query Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleRunQuery()}
          placeholder="Enter a test query (e.g., 'What are the safety requirements?')"
          className="flex-1 px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500"
          disabled={isLoading}
        />
        <button
          onClick={handleRunQuery}
          disabled={isLoading || !query.trim()}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
        >
          {isLoading ? 'Testing...' : 'Run Query'}
        </button>
      </div>

      {/* Sample Queries */}
      <div className="flex flex-wrap gap-2">
        <span className="text-sm text-gray-400">Try:</span>
        {SAMPLE_QUERIES.map((sample) => (
          <button
            key={sample}
            onClick={() => setQuery(sample)}
            className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            {sample}
          </button>
        ))}
      </div>

      {/* Results */}
      {Object.keys(results).length > 0 && (
        <div className="space-y-3">
          <h3 className="font-bold text-lg">Results</h3>
          {strategies.map((strategy) => (
            <div
              key={strategy}
              className="bg-gray-800 rounded-lg p-4 border border-gray-700"
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-bold capitalize">{strategy}</h4>
                {results[strategy]?.[0] && (
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-400">Top Score:</span>
                    <span className="font-mono font-bold">
                      {results[strategy][0].top_score.toFixed(3)}
                    </span>
                  </div>
                )}
              </div>

              {results[strategy] && results[strategy].length > 0 ? (
                <div className="space-y-2">
                  {results[strategy].slice(0, 3).map((result, i) => (
                    <div
                      key={i}
                      className="bg-gray-900 rounded p-3 text-sm"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-400">
                          Chunk {result.top_chunk_id}
                        </span>
                        <span className="font-mono text-green-400">
                          {result.top_score.toFixed(3)}
                        </span>
                      </div>
                      {result.chunk_preview && (
                        <p className="text-gray-300 text-xs mt-2">
                          {result.chunk_preview.slice(0, 150)}...
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-sm">No results</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const SAMPLE_QUERIES = [
  'What are the requirements?',
  'How do I perform maintenance?',
  'What tools are needed?',
];
