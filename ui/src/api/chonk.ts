/**
 * CHONK API Client
 *
 * Communicates with the Python backend via REST API.
 */

const API_BASE = 'http://127.0.0.1:8420';

// Type definitions matching the Python backend
export interface ChunkMetadata {
  tags: string[];
  hierarchy_hint: string | null;
  notes: string | null;
  custom: Record<string, unknown>;
}

export interface QualityScore {
  token_range: number;
  sentence_complete: number;
  hierarchy_preserved: number;
  table_integrity: number;
  reference_complete: number;
  overall: number;
}

export interface Chunk {
  id: string;
  block_ids: string[];
  content: string;
  token_count: number;
  quality: QualityScore;
  hierarchy_path: string;
  user_metadata: ChunkMetadata;
  system_metadata: Record<string, unknown>;
  is_modified: boolean;
  is_locked: boolean;
  created_at: string;
  modified_at: string | null;
}

export interface Block {
  id: string;
  type: string;
  content: string;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    page: number;
  } | null;
  page: number;
  parent_id: string | null;
  children_ids: string[];
  metadata: Record<string, unknown>;
  confidence: number;
  heading_level: number | null;
}

export interface DocumentMetadata {
  title: string | null;
  author: string | null;
  subject: string | null;
  keywords: string[];
  created_date: string | null;
  modified_date: string | null;
  page_count: number;
  word_count: number;
  file_size_bytes: number;
  custom: Record<string, unknown>;
}

export interface Document {
  id: string;
  source_path: string;
  source_type: string;
  blocks: Block[];
  chunks: Chunk[];
  metadata: DocumentMetadata;
  loader_used: string;
  parser_used: string;
  chunker_used: string;
  chunker_config: Record<string, unknown>;
  loaded_at: string;
  last_chunked_at: string | null;
}

export interface TestQuery {
  id: string;
  query: string;
  expected_chunk_ids: string[];
  excluded_chunk_ids: string[];
  notes: string | null;
  created_at: string;
}

export interface TestSuite {
  id: string;
  name: string;
  queries: TestQuery[];
  created_at: string;
  modified_at: string | null;
}

export interface ProjectSettings {
  default_chunker: string;
  default_chunk_size: number;
  default_overlap: number;
  embedding_model: string;
  output_directory: string | null;
}

export interface Project {
  id: string;
  name: string;
  documents: Document[];
  test_suites: TestSuite[];
  settings: ProjectSettings;
  project_path: string | null;
  created_at: string;
  modified_at: string | null;
}

export interface SearchResult {
  chunk_id: string;
  score: number;
  rank: number;
  document_id: string | null;
  document_name: string | null;
  content_preview: string;
  token_count: number;
  page_range: [number, number] | null;
  hierarchy_path: string;
}

export interface QualityReport {
  total_chunks: number;
  average_score: number;
  problem_count: number;
  warning_count: number;
  problems: Array<{ chunk_id: string; score: number; details: QualityScore }>;
  warnings: Array<{ chunk_id: string; score: number; details: QualityScore }>;
  all_scores: Array<{ chunk_id: string; score: number; details: QualityScore }>;
}

// Hierarchy interfaces
export interface HierarchyNode {
  section_id: string;
  heading: string | null;
  heading_level: number;
  content: string;
  token_count: number;
  page_range: number[];
  hierarchy_path: string;
  depth: number;
  is_leaf: boolean;
  child_count: number;
  children: HierarchyNode[];
  heading_block_id: string | null;
  content_block_ids: string[];
}

export interface HierarchyTree {
  document_id: string;
  statistics: {
    total_nodes: number;
    nodes_with_content: number;
    nodes_with_children: number;
    leaf_nodes: number;
    max_depth: number;
    avg_tokens_per_node: number;
    min_tokens: number;
    max_tokens: number;
    level_distribution: Record<number, number>;
  };
  root: HierarchyNode;
}

// Strategy comparison interfaces
export interface StrategyResult {
  strategy_name: string;
  chunks_count: number;
  avg_tokens: number;
  min_tokens: number;
  max_tokens: number;
  avg_quality_score: number;
  hierarchy_preservation: number;
  chunks_with_context: number;
  processing_time_ms: number;
}

export interface ComparisonResult {
  document_id: string;
  strategies: StrategyResult[];
  recommendation: string;
  compared_at: string;
}

// Query testing interfaces
export interface QueryResult {
  chunk_id: string;
  score: number;
  content_preview: string;
  hierarchy_path: string;
  token_count: number;
}

export interface StrategyQueryResult {
  strategy_name: string;
  query: string;
  results: QueryResult[];
  top_score: number;
  retrieved_count: number;
}

// API Error class
export class APIError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

// Generic fetch wrapper
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new APIError(response.status, error.detail || response.statusText);
  }

  return response.json();
}

// Project endpoints
export const projectAPI = {
  create: (name: string, outputDirectory?: string) =>
    apiFetch<{ id: string; name: string; created_at: string }>('/api/project/new', {
      method: 'POST',
      body: JSON.stringify({ name, output_directory: outputDirectory }),
    }),

  open: (path: string) =>
    apiFetch<{ id: string; name: string; document_count: number; created_at: string }>(
      `/api/project/open?path=${encodeURIComponent(path)}`,
      { method: 'POST' }
    ),

  save: (path?: string) =>
    apiFetch<{ path: string; saved_at: string }>(
      `/api/project/save${path ? `?path=${encodeURIComponent(path)}` : ''}`,
      { method: 'POST' }
    ),

  get: () => apiFetch<Project>('/api/project'),
};

// Document endpoints
export const documentAPI = {
  upload: async (file: File, extractionTier?: string) => {
    const formData = new FormData();
    formData.append('file', file);

    const url = extractionTier
      ? `${API_BASE}/api/documents/upload?extraction_tier=${encodeURIComponent(extractionTier)}`
      : `${API_BASE}/api/documents/upload`;

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new APIError(response.status, error.detail);
    }

    return response.json() as Promise<{
      document_id: string;
      filename: string;
      page_count: number;
      chunk_count: number;
      word_count: number;
      tier_used?: string;
      extraction_info?: Record<string, unknown>;
      warnings?: string[];
    }>;
  },

  get: (documentId: string) =>
    apiFetch<Document>(`/api/documents/${documentId}`),

  delete: (documentId: string) =>
    apiFetch<{ deleted: string }>(`/api/documents/${documentId}`, {
      method: 'DELETE',
    }),

  rechunk: (
    documentId: string,
    config: {
      chunker?: string;
      target_tokens?: number;
      max_tokens?: number;
      min_tokens?: number;
      overlap_tokens?: number;
    }
  ) =>
    apiFetch<{ chunk_count: number; quality: QualityReport }>(
      `/api/documents/${documentId}/rechunk`,
      {
        method: 'POST',
        body: JSON.stringify({
          chunker: config.chunker ?? 'hierarchy',
          target_tokens: config.target_tokens ?? 400,
          max_tokens: config.max_tokens ?? 600,
          min_tokens: config.min_tokens ?? 100,
          overlap_tokens: config.overlap_tokens ?? 50,
          respect_boundaries: true,
          preserve_tables: true,
          preserve_code: true,
        }),
      }
    ),

  getQuality: (documentId: string) =>
    apiFetch<QualityReport>(`/api/documents/${documentId}/quality`),
};

// Chunk endpoints
export const chunkAPI = {
  merge: (chunkIds: string[]) =>
    apiFetch<Chunk>('/api/chunks/merge', {
      method: 'POST',
      body: JSON.stringify({ chunk_ids: chunkIds }),
    }),

  split: (chunkId: string, splitPosition: number) =>
    apiFetch<{ chunk_a: Chunk; chunk_b: Chunk }>('/api/chunks/split', {
      method: 'POST',
      body: JSON.stringify({ chunk_id: chunkId, split_position: splitPosition }),
    }),

  update: (
    chunkId: string,
    updates: {
      tags?: string[];
      hierarchy_hint?: string | null;
      notes?: string | null;
      custom?: Record<string, unknown>;
      is_locked?: boolean;
    }
  ) =>
    apiFetch<Chunk>(`/api/chunks/${chunkId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    }),

  getSuggestions: (chunkId: string) =>
    apiFetch<{ chunk_id: string; quality_score: number; suggestions: string[] }>(
      `/api/chunks/${chunkId}/suggestions`
    ),
};

// Search/Testing endpoints
export const testAPI = {
  search: (query: string, topK = 5, documentIds?: string[]) =>
    apiFetch<{ query: string; results: SearchResult[] }>('/api/test/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK, document_ids: documentIds }),
    }),

  getStatus: () =>
    apiFetch<{ indexed: boolean; chunk_count: number }>('/api/test/status'),

  reindex: () =>
    apiFetch<{ indexed: boolean; chunk_count: number }>('/api/test/reindex', {
      method: 'POST',
    }),

  createSuite: (name: string) =>
    apiFetch<TestSuite>(`/api/test-suites?name=${encodeURIComponent(name)}`, {
      method: 'POST',
    }),

  addQuery: (
    suiteId: string,
    query: string,
    expectedChunkIds?: string[],
    excludedChunkIds?: string[],
    notes?: string
  ) =>
    apiFetch<TestQuery>(`/api/test-suites/${suiteId}/queries`, {
      method: 'POST',
      body: JSON.stringify({
        query,
        expected_chunk_ids: expectedChunkIds ?? [],
        excluded_chunk_ids: excludedChunkIds ?? [],
        notes,
      }),
    }),

  runSuite: (suiteId: string, topK = 5, documentIds?: string[]) =>
    apiFetch<{
      suite_id: string;
      suite_name: string;
      passed_count: number;
      failed_count: number;
      pass_rate: number;
      total_time_ms: number;
      run_at: string;
      results: Array<{
        query_id: string;
        query_text: string;
        passed: boolean;
        results: SearchResult[];
        missing_expected: string[];
        unexpected_excluded: string[];
        execution_time_ms: number;
      }>;
    }>(`/api/test-suites/${suiteId}/run?top_k=${topK}`, {
      method: 'POST',
    }),

  getCoverage: (suiteId: string, topK = 5) =>
    apiFetch<{
      total_chunks: number;
      never_retrieved_count: number;
      never_retrieved: string[];
      frequently_retrieved: Array<{ chunk_id: string; query_count: number }>;
      coverage_rate: number;
    }>(`/api/test-suites/${suiteId}/coverage?top_k=${topK}`),
};

// Export endpoints
export const exportAPI = {
  export: (format: string, outputPath: string, documentId?: string) =>
    apiFetch<{ path: string; format: string; exported_at: string }>('/api/export', {
      method: 'POST',
      body: JSON.stringify({ format, output_path: outputPath, document_id: documentId }),
    }),

  getFormats: () => apiFetch<{ formats: string[] }>('/api/export/formats'),
};

// Extractor interface
export interface Extractor {
  id: string;
  name: string;
  description: string;
  available: boolean;
  tier: number;
  install_hint: string | null;
}

// Utility endpoints
export const utilAPI = {
  getLoaders: () => apiFetch<{ extensions: string[] }>('/api/loaders'),
  getChunkers: () => apiFetch<{ chunkers: string[] }>('/api/chunkers'),
  getExtractors: () =>
    apiFetch<{ extractors: Extractor[]; available_count: number }>('/api/extractors'),
  healthCheck: () => apiFetch<{ status: string; version: string }>('/api/health'),
};

// Settings endpoints
export const settingsAPI = {
  get: () => apiFetch<Record<string, unknown>>('/api/settings'),
  save: (settings: Record<string, unknown>) =>
    apiFetch<{ saved: boolean }>('/api/settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    }),
};

// Hierarchy endpoints
export const hierarchyAPI = {
  build: (documentId: string) =>
    apiFetch<HierarchyTree>('/api/hierarchy/build', {
      method: 'POST',
      body: JSON.stringify({ document_id: documentId }),
    }),

  get: (documentId: string) =>
    apiFetch<HierarchyTree>(`/api/hierarchy/${documentId}`),

  getStats: (documentId: string) =>
    apiFetch<{
      total_nodes: number;
      total_headings: number;
      max_depth: number;
      quality_score: number;
    }>(`/api/hierarchy/${documentId}/stats`),
};

// Comparison endpoints
export const comparisonAPI = {
  compare: (
    documentId: string,
    strategies: Array<{
      name: string;
      config: {
        target_tokens?: number;
        max_tokens?: number;
        overlap_tokens?: number;
        preserve_tables?: boolean;
        preserve_code?: boolean;
        group_under_headings?: boolean;
        heading_weight?: number;
      };
    }>
  ) =>
    apiFetch<ComparisonResult>('/api/chunk/compare', {
      method: 'POST',
      body: JSON.stringify({ document_id: documentId, strategies }),
    }),

  preview: (
    documentId: string,
    config: {
      chunker: string;
      target_tokens?: number;
      max_tokens?: number;
      overlap_tokens?: number;
      preserve_tables?: boolean;
      preserve_code?: boolean;
      group_under_headings?: boolean;
      heading_weight?: number;
    }
  ) =>
    apiFetch<{ chunks: Chunk[]; quality: QualityReport }>('/api/chunk/preview', {
      method: 'POST',
      body: JSON.stringify({ document_id: documentId, ...config }),
    }),
};

// Enhanced query testing across strategies
export const queryTestAPI = {
  testQuery: (query: string, strategies: string[], documentId?: string) =>
    apiFetch<{
      query: string;
      strategies: StrategyQueryResult[];
    }>('/api/test/query', {
      method: 'POST',
      body: JSON.stringify({ query, strategies, document_id: documentId }),
    }),

  compareStrategies: (
    queries: string[],
    strategies: string[],
    documentId?: string
  ) =>
    apiFetch<{
      queries: Array<{
        query: string;
        strategies: StrategyQueryResult[];
      }>;
      summary: {
        best_strategy: string;
        avg_scores: Record<string, number>;
      };
    }>('/api/test/compare-strategies', {
      method: 'POST',
      body: JSON.stringify({ queries, strategies, document_id: documentId }),
    }),
};

// Diagnostic interfaces
export interface ChunkProblem {
  chunk_id: string;
  problem_type: string;
  severity: 'high' | 'medium' | 'low';
  description: string;
  details: Record<string, unknown>;
  suggested_fix: string | null;
}

export interface DiagnosticStatistics {
  total_problems: number;
  unique_chunks_with_problems: number;
  by_type: Record<string, number>;
  by_severity: Record<string, number>;
  avg_problems_per_chunk: number;
}

export interface GeneratedQuestion {
  question: string;
  expected_chunk_id: string;
  question_type: string;
  context: Record<string, unknown>;
}

export interface QuestionTestResult {
  question: string;
  expected_chunk_id: string;
  question_type: string;
  retrieved_chunk_ids: string[];
  success: boolean;
  top_score: number;
  expected_rank: number | null;
}

export interface FixAction {
  action_type: 'merge' | 'split';
  chunk_ids: string[];
  description: string;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface FixPlan {
  actions: FixAction[];
  estimated_improvement: number;
  conflicts: string[];
  warnings: string[];
  total_actions: number;
}

export interface FixResult {
  success: boolean;
  chunks_before: number;
  chunks_after: number;
  actions_applied: FixAction[];
  new_chunks: Chunk[];
  errors: string[];
}

export interface DiagnosticResult {
  document_id: string;
  problems: ChunkProblem[];
  statistics: DiagnosticStatistics;
  questions_generated: number;
  questions_tested: number;
  question_pass_rate: number;
  analyzed_at: string;
}

// Diagnostic endpoints
export const diagnosticAPI = {
  analyze: (documentId: string, includeQuestions = true, testQuestions = true) =>
    apiFetch<DiagnosticResult>('/api/diagnostics/analyze', {
      method: 'POST',
      body: JSON.stringify({
        document_id: documentId,
        include_questions: includeQuestions,
        test_questions: testQuestions,
      }),
    }),

  getProblems: (documentId: string) =>
    apiFetch<{
      document_id: string;
      problems: ChunkProblem[];
      statistics: DiagnosticStatistics;
    }>(`/api/diagnostics/${documentId}/problems`),

  generateQuestions: (documentId: string) =>
    apiFetch<{
      document_id: string;
      questions: GeneratedQuestion[];
      chunk_coverage: number;
      total_questions: number;
    }>('/api/diagnostics/generate-questions', {
      method: 'POST',
      body: JSON.stringify({ document_id: documentId }),
    }),

  testQuestions: (documentId: string, questions: GeneratedQuestion[], topK = 5) =>
    apiFetch<{
      document_id: string;
      results: QuestionTestResult[];
      pass_rate: number;
      avg_expected_rank: number;
      total_tested: number;
    }>('/api/diagnostics/test-questions', {
      method: 'POST',
      body: JSON.stringify({
        document_id: documentId,
        questions,
        top_k: topK,
      }),
    }),

  previewFixes: (documentId: string, autoResolveConflicts = true) =>
    apiFetch<{
      document_id: string;
      problems_found: number;
      fix_plan: FixPlan;
    }>('/api/diagnostics/preview-fixes', {
      method: 'POST',
      body: JSON.stringify({
        document_id: documentId,
        auto_resolve_conflicts: autoResolveConflicts,
      }),
    }),

  applyFixes: (documentId: string, autoResolveConflicts = true, validate = true) =>
    apiFetch<{
      document_id: string;
      result: string;
      fix_result: FixResult;
      before: {
        problems: number;
        statistics: DiagnosticStatistics;
      };
      after: {
        problems: number;
        statistics: DiagnosticStatistics;
      };
      improvement: {
        problems_fixed: number;
        reduction_rate: number;
      };
    }>('/api/diagnostics/apply-fixes', {
      method: 'POST',
      body: JSON.stringify({
        document_id: documentId,
        auto_resolve_conflicts: autoResolveConflicts,
        validate,
      }),
    }),
};
