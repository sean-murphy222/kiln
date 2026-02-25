/**
 * Hearth API Client
 *
 * Communicates with the Hearth backend (inference & chat) via REST API.
 */

const API_BASE = "http://127.0.0.1:8420/api/hearth";

// ============================================================
// Type definitions
// ============================================================

export interface ModelSlot {
  id: string;
  name: string;
  base_model: string;
  adapter_path: string | null;
  status: "ready" | "loading" | "error" | "unloaded";
  loaded_at: string | null;
}

export interface Citation {
  document_id: string;
  document_title: string;
  section: string;
  page: number;
  relevance_score: number;
  snippet: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  timestamp: string;
  model_id: string | null;
}

export interface Conversation {
  id: string;
  title: string;
  model_id: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
}

export interface QueryRequest {
  query: string;
  conversation_id?: string;
  model_id?: string;
  include_citations?: boolean;
  max_context_chunks?: number;
}

export interface QueryResponse {
  conversation_id: string;
  message: Message;
  processing_time_ms: number;
}

export interface DocumentSummary {
  id: string;
  title: string;
  source_type: string;
  chunk_count: number;
  page_count: number;
}

export interface DocumentDetail {
  id: string;
  title: string;
  source_path: string;
  source_type: string;
  chunks: Array<{
    id: string;
    content: string;
    hierarchy_path: string;
    page: number;
    token_count: number;
  }>;
  metadata: Record<string, unknown>;
}

export interface FeedbackSubmission {
  message_id: string;
  conversation_id: string;
  signal_type: "positive" | "negative" | "flag_incorrect" | "flag_incomplete";
  comment?: string;
}

export interface FeedbackDashboard {
  total_queries: number;
  positive_count: number;
  negative_count: number;
  flagged_count: number;
  acceptance_rate: number;
  period_days: number;
}

export interface FeedbackPattern {
  id: string;
  pattern_type: string;
  description: string;
  severity: "high" | "medium" | "low";
  occurrence_count: number;
  routing_suggestion: string;
}

export interface RoutingDecision {
  signal_type: string;
  routed_to: "quarry" | "forge" | "foundry" | "none";
  reason: string;
  priority: "high" | "medium" | "low";
}

// ============================================================
// API Error
// ============================================================

export class HearthAPIError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

// ============================================================
// Generic fetch wrapper
// ============================================================

async function hearthFetch<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Unknown error" }));
    throw new HearthAPIError(
      response.status,
      error.detail || response.statusText,
    );
  }

  return response.json();
}

// ============================================================
// Endpoint groups
// ============================================================

export const hearthHealthAPI = {
  check: () => hearthFetch<{ status: string }>("/health"),
};

export const modelAPI = {
  list: () => hearthFetch<ModelSlot[]>("/models"),

  register: (data: {
    name: string;
    base_model: string;
    adapter_path?: string | null;
  }) =>
    hearthFetch<ModelSlot>("/models/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  load: (modelId: string) =>
    hearthFetch<ModelSlot>(`/models/${modelId}/load`, { method: "POST" }),

  unload: (modelId: string) =>
    hearthFetch<ModelSlot>(`/models/${modelId}/unload`, { method: "POST" }),

  getStatus: (modelId: string) =>
    hearthFetch<ModelSlot>(`/models/${modelId}/status`),
};

export const queryAPI = {
  send: (request: QueryRequest) =>
    hearthFetch<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify(request),
    }),

  multiDiscipline: (request: QueryRequest & { discipline_ids: string[] }) =>
    hearthFetch<QueryResponse>("/query/multi-discipline", {
      method: "POST",
      body: JSON.stringify(request),
    }),
};

export const conversationAPI = {
  list: () => hearthFetch<Conversation[]>("/conversations"),

  get: (id: string) => hearthFetch<Conversation>(`/conversations/${id}`),

  delete: (id: string) =>
    hearthFetch<{ deleted: boolean }>(`/conversations/${id}`, {
      method: "DELETE",
    }),
};

export const documentBrowseAPI = {
  list: () => hearthFetch<DocumentSummary[]>("/documents"),

  get: (id: string) => hearthFetch<DocumentDetail>(`/documents/${id}`),

  search: (query: string, topK: number = 5) =>
    hearthFetch<{
      query: string;
      results: Array<{
        document_id: string;
        chunk_id: string;
        score: number;
        snippet: string;
        hierarchy_path: string;
      }>;
    }>("/documents/search", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK }),
    }),
};

export const feedbackAPI = {
  submit: (feedback: FeedbackSubmission) =>
    hearthFetch<{ id: string; routed: boolean }>("/feedback", {
      method: "POST",
      body: JSON.stringify(feedback),
    }),

  getRouting: () => hearthFetch<RoutingDecision[]>("/feedback/routing"),

  getDashboard: (periodDays: number = 30) =>
    hearthFetch<FeedbackDashboard>(
      `/feedback/dashboard?period_days=${periodDays}`,
    ),

  getPatterns: () => hearthFetch<FeedbackPattern[]>("/feedback/patterns"),
};
