/**
 * Mappers from Hearth API client types to Hearth store types.
 *
 * The API client and the Zustand store intentionally use slightly different
 * shapes (the store adds a synthetic citation `id` and a `feedback` field).
 * These helpers translate between the two at the wiring boundary.
 */

import type {
  Citation as ApiCitation,
  Message as ApiMessage,
  Conversation as ApiConversation,
  ModelSlot as ApiModelSlot,
} from "@/api/hearth";
import type {
  Citation,
  Message,
  Conversation,
  ModelSlot,
} from "@/store/useHearthStore";

export function mapCitation(citation: ApiCitation, index: number): Citation {
  const docKey = citation.document_id ?? citation.document_title;
  return {
    id: `${docKey}:${citation.page}:${index}`,
    document_title: citation.document_title,
    section: citation.section,
    page: citation.page,
    relevance_score: citation.relevance_score,
    snippet: citation.snippet,
  };
}

export function mapMessage(message: ApiMessage): Message {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    citations: (message.citations ?? []).map(mapCitation),
    timestamp: message.timestamp,
    feedback: null,
  };
}

export function mapConversation(conversation: ApiConversation): Conversation {
  return {
    id: conversation.id,
    title: conversation.title,
    model_id: conversation.model_id,
    messages: (conversation.messages ?? []).map(mapMessage),
    created_at: conversation.created_at,
    updated_at: conversation.updated_at,
  };
}

export function mapModelSlot(slot: ApiModelSlot): ModelSlot {
  return {
    id: slot.id,
    name: slot.name,
    status: slot.status,
    base_model: slot.base_model,
    adapter_path: slot.adapter_path,
  };
}
