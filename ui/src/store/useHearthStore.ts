/**
 * Hearth-specific state — conversations, model slots, feedback.
 */

import { create } from 'zustand';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations: Citation[];
  timestamp: string;
  feedback?: 'positive' | 'negative' | null;
}

export interface Citation {
  id: string;
  document_title: string;
  section: string;
  page: number;
  relevance_score: number;
  snippet: string;
}

export interface Conversation {
  id: string;
  title: string;
  model_id: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
}

export interface ModelSlot {
  id: string;
  name: string;
  status: 'ready' | 'loading' | 'error' | 'unloaded';
  base_model: string;
  adapter_path: string | null;
}

interface HearthState {
  conversations: Conversation[];
  activeConversationId: string | null;
  modelSlots: ModelSlot[];
  activeModelId: string | null;
  citationPanelOpen: boolean;
  isStreaming: boolean;
  isLoading: boolean;
  error: string | null;

  setConversations: (conversations: Conversation[]) => void;
  setActiveConversation: (id: string | null) => void;
  addMessage: (conversationId: string, message: Message) => void;
  setModelSlots: (slots: ModelSlot[]) => void;
  setActiveModel: (id: string | null) => void;
  toggleCitationPanel: () => void;
  setStreaming: (streaming: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useHearthStore = create<HearthState>((set) => ({
  conversations: [],
  activeConversationId: null,
  modelSlots: [],
  activeModelId: null,
  citationPanelOpen: true,
  isStreaming: false,
  isLoading: false,
  error: null,

  setConversations: (conversations) => set({ conversations }),
  setActiveConversation: (activeConversationId) => set({ activeConversationId }),
  addMessage: (conversationId, message) =>
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === conversationId
          ? { ...c, messages: [...c.messages, message], updated_at: new Date().toISOString() }
          : c
      ),
    })),
  setModelSlots: (modelSlots) => set({ modelSlots }),
  setActiveModel: (activeModelId) => set({ activeModelId }),
  toggleCitationPanel: () => set((s) => ({ citationPanelOpen: !s.citationPanelOpen })),
  setStreaming: (isStreaming) => set({ isStreaming }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
