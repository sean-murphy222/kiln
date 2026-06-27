/**
 * Hearth API Client
 *
 * Communicates with the Hearth backend (inference & chat) via REST API.
 *
 * The Hearth FastAPI router (hearth/src/server.py) returns domain `to_dict()`
 * shapes that differ from the UI's normalized shapes:
 *   - list endpoints wrap results ({ "models": [...] }, { "conversations": [...] })
 *   - model slots use slot_id/display_name/base_model_family/lora_path
 *   - POST /query takes { query, slot_id, ... } and returns
 *     { answer, citations[], conversation_id, model_used, latency_ms, chunk_count }
 *
 * Each method below translates the real backend payload into the interfaces
 * declared in this file, which are the contract the UI components rely on.
 */

const API_BASE = "http://127.0.0.1:8420/api/hearth";

// ============================================================
// Type definitions (UI-normalized shapes)
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
  document_id?: string;
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
  /** The target model slot id (sent to the backend as `slot_id`). */
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
  query: string;
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
// Raw backend payload shapes (hearth/src/inference.py to_dict)
// ============================================================

interface RawModelSlot {
  slot_id: string;
  display_name: string;
  base_model_family: string;
  status: ModelSlot["status"];
  model_path: string | null;
  discipline_id: string | null;
  lora_path: string | null;
  loaded_at: string | null;
}

interface RawCitation {
  document_title: string;
  section: string;
  page: number;
  relevance_score: number;
  snippet: string;
}

interface RawQueryResponse {
  answer: string;
  citations: RawCitation[];
  conversation_id: string;
  model_used: string | null;
  latency_ms: number;
  chunk_count: number;
}

interface RawConversationSummary {
  conversation_id: string;
  title: string;
  created_at: string;
  model_slot_id: string;
  turn_count: number;
}

interface RawConversationTurn {
  turn_id: string;
  query: string;
  response: string;
  citations: RawCitation[];
  timestamp: string;
  model_slot_id: string;
}

interface RawConversation {
  conversation_id: string;
  title: string;
  turns: RawConversationTurn[];
  created_at: string;
  model_slot_id: string;
}

// ============================================================
// Transforms
// ============================================================

function toModelSlot(raw: RawModelSlot): ModelSlot {
  return {
    id: raw.slot_id,
    name: raw.display_name,
    base_model: raw.base_model_family,
    adapter_path: raw.lora_path,
    status: raw.status,
    loaded_at: raw.loaded_at,
  };
}

function toCitation(raw: RawCitation): Citation {
  return {
    document_title: raw.document_title,
    section: raw.section,
    page: raw.page,
    relevance_score: raw.relevance_score,
    snippet: raw.snippet,
  };
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
  list: async (): Promise<ModelSlot[]> => {
    const data = await hearthFetch<{ models: RawModelSlot[] }>("/models");
    return (data.models ?? []).map(toModelSlot);
  },

  register: async (data: {
    name: string;
    base_model: string;
    adapter_path?: string | null;
  }): Promise<ModelSlot> => {
    const raw = await hearthFetch<RawModelSlot>("/models/register", {
      method: "POST",
      body: JSON.stringify({
        slot_id: data.name,
        display_name: data.name,
        base_model_family: data.base_model,
        lora_path: data.adapter_path ?? null,
      }),
    });
    return toModelSlot(raw);
  },

  load: async (modelId: string): Promise<ModelSlot> => {
    const raw = await hearthFetch<RawModelSlot>(`/models/${modelId}/load`, {
      method: "POST",
    });
    return toModelSlot(raw);
  },

  unload: (modelId: string) =>
    hearthFetch<{ model_id: string; status: string }>(
      `/models/${modelId}/unload`,
      { method: "POST" },
    ),

  getStatus: async (modelId: string): Promise<ModelSlot> => {
    const raw = await hearthFetch<RawModelSlot>(`/models/${modelId}/status`);
    return toModelSlot(raw);
  },
};

export const queryAPI = {
  send: async (request: QueryRequest): Promise<QueryResponse> => {
    const raw = await hearthFetch<RawQueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify({
        query: request.query,
        slot_id: request.model_id,
        conversation_id: request.conversation_id,
        include_citations: request.include_citations ?? true,
        max_context_chunks: request.max_context_chunks ?? 5,
      }),
    });
    const message: Message = {
      id: `msg-${Date.now()}`,
      role: "assistant",
      content: raw.answer,
      citations: (raw.citations ?? []).map(toCitation),
      timestamp: new Date().toISOString(),
      model_id: raw.model_used ?? null,
    };
    return {
      conversation_id: raw.conversation_id,
      message,
      processing_time_ms: raw.latency_ms,
    };
  },

  multiDiscipline: async (
    request: QueryRequest & { discipline_ids: string[] },
  ): Promise<QueryResponse> => {
    const data = await hearthFetch<{ responses: RawQueryResponse[] }>(
      "/query/multi-discipline",
      {
        method: "POST",
        body: JSON.stringify({
          query: request.query,
          slot_ids: request.discipline_ids,
        }),
      },
    );
    const raw = data.responses[0];
    const message: Message = {
      id: `msg-${Date.now()}`,
      role: "assistant",
      content: raw?.answer ?? "",
      citations: (raw?.citations ?? []).map(toCitation),
      timestamp: new Date().toISOString(),
      model_id: raw?.model_used ?? null,
    };
    return {
      conversation_id: raw?.conversation_id ?? "",
      message,
      processing_time_ms: raw?.latency_ms ?? 0,
    };
  },
};

export const conversationAPI = {
  list: async (): Promise<Conversation[]> => {
    const data = await hearthFetch<{
      conversations: RawConversationSummary[];
    }>("/conversations");
    return (data.conversations ?? []).map((c) => ({
      id: c.conversation_id,
      title: c.title,
      model_id: c.model_slot_id,
      messages: [],
      created_at: c.created_at,
      updated_at: c.created_at,
    }));
  },

  get: async (id: string): Promise<Conversation> => {
    const raw = await hearthFetch<RawConversation>(`/conversations/${id}`);
    const messages: Message[] = [];
    for (const turn of raw.turns ?? []) {
      messages.push({
        id: `${turn.turn_id}-q`,
        role: "user",
        content: turn.query,
        citations: [],
        timestamp: turn.timestamp,
        model_id: turn.model_slot_id,
      });
      messages.push({
        id: `${turn.turn_id}-a`,
        role: "assistant",
        content: turn.response,
        citations: (turn.citations ?? []).map(toCitation),
        timestamp: turn.timestamp,
        model_id: turn.model_slot_id,
      });
    }
    return {
      id: raw.conversation_id,
      title: raw.title,
      model_id: raw.model_slot_id,
      messages,
      created_at: raw.created_at,
      updated_at: raw.created_at,
    };
  },

  delete: (id: string) =>
    hearthFetch<{ deleted: string }>(`/conversations/${id}`, {
      method: "DELETE",
    }),
};

export const documentBrowseAPI = {
  list: async (): Promise<DocumentSummary[]> => {
    const data = await hearthFetch<{
      documents: Array<{
        document_id: string;
        title: string;
        document_type: string;
        chunk_count: number;
        page_count: number | null;
      }>;
    }>("/documents");
    return (data.documents ?? []).map((d) => ({
      id: d.document_id,
      title: d.title,
      source_type: d.document_type,
      chunk_count: d.chunk_count,
      page_count: d.page_count ?? 0,
    }));
  },

  get: async (id: string): Promise<DocumentDetail> => {
    const raw = await hearthFetch<{
      document_id: string;
      title: string;
      document_type: string;
      chunks: DocumentDetail["chunks"];
      metadata: Record<string, unknown>;
    }>(`/documents/${id}`);
    return {
      id: raw.document_id,
      title: raw.title,
      source_path: "",
      source_type: raw.document_type,
      chunks: raw.chunks ?? [],
      metadata: raw.metadata ?? {},
    };
  },

  search: (query: string, limit: number = 5) =>
    hearthFetch<{
      results: Array<Record<string, unknown>>;
    }>("/documents/search", {
      method: "POST",
      body: JSON.stringify({ query, limit }),
    }),
};

export const feedbackAPI = {
  submit: (feedback: FeedbackSubmission) =>
    hearthFetch<Record<string, unknown>>("/feedback", {
      method: "POST",
      body: JSON.stringify({
        signal_type: feedback.signal_type,
        conversation_id: feedback.conversation_id,
        query: feedback.query,
        user_comment: feedback.comment,
      }),
    }),

  getRouting: (signalId: string) =>
    hearthFetch<RoutingDecision>(
      `/feedback/routing?signal_id=${encodeURIComponent(signalId)}`,
    ),

  getDashboard: (disciplineId: string, days: number = 30) =>
    hearthFetch<FeedbackDashboard>(
      `/feedback/dashboard?discipline_id=${encodeURIComponent(
        disciplineId,
      )}&days=${days}`,
    ),

  getPatterns: async (disciplineId?: string): Promise<FeedbackPattern[]> => {
    const qs = disciplineId
      ? `?discipline_id=${encodeURIComponent(disciplineId)}`
      : "";
    const data = await hearthFetch<{ patterns: FeedbackPattern[] }>(
      `/feedback/patterns${qs}`,
    );
    return data.patterns ?? [];
  },
};
