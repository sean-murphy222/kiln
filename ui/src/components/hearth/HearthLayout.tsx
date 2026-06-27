import { useState, useCallback, useEffect } from "react";
import {
  Flame,
  PanelLeftClose,
  PanelLeft,
  PanelRightClose,
  PanelRight,
} from "lucide-react";
import { useHearthStore } from "@/store/useHearthStore";
import type { Citation } from "@/store/useHearthStore";
import { modelAPI, conversationAPI, queryAPI } from "@/api/hearth";
import { mapConversation, mapMessage, mapModelSlot } from "@/lib/hearthMappers";
import { showToast } from "@/components/common/Toast";
import { ToolHeader } from "@/components/shell/ToolHeader";
import { ConversationList } from "./ConversationList";
import { ChatArea } from "./ChatArea";
import { CitationPanel } from "./CitationPanel";
import { ModelSwitcher } from "./ModelSwitcher";

/** Local (not-yet-persisted) conversation ids use this prefix. */
const LOCAL_PREFIX = "conv-";

export function HearthLayout() {
  const {
    conversations,
    activeConversationId,
    modelSlots,
    activeModelId,
    citationPanelOpen,
    isStreaming,
    setActiveConversation,
    setConversations,
    setModelSlots,
    setActiveModel,
    toggleCitationPanel,
    setStreaming,
  } = useHearthStore();

  const [conversationListOpen, setConversationListOpen] = useState(true);
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);

  const activeConversation = conversations.find(
    (c) => c.id === activeConversationId,
  );
  const activeMessages = activeConversation?.messages ?? [];

  // Citations from the last assistant message
  const lastAssistantMsg = [...activeMessages]
    .reverse()
    .find((m) => m.role === "assistant");
  const activeCitations = lastAssistantMsg?.citations ?? [];

  // Load models and conversations from the backend on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [slots, convos] = await Promise.all([
          modelAPI.list(),
          conversationAPI.list(),
        ]);
        if (cancelled) return;
        const mappedSlots = slots.map(mapModelSlot);
        setModelSlots(mappedSlots);
        setConversations(convos.map(mapConversation));
        const ready = mappedSlots.find((s) => s.status === "ready");
        if (ready) setActiveModel(ready.id);
        else if (mappedSlots[0]) setActiveModel(mappedSlots[0].id);
      } catch (err) {
        if (!cancelled) {
          showToast(
            "error",
            err instanceof Error ? err.message : "Failed to load Hearth data",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setModelSlots, setConversations, setActiveModel]);

  const refreshModels = useCallback(async () => {
    try {
      const slots = await modelAPI.list();
      setModelSlots(slots.map(mapModelSlot));
    } catch {
      /* non-fatal: keep existing slots */
    }
  }, [setModelSlots]);

  const handleNewChat = useCallback(() => {
    // Conversations are created server-side on first query; just reset.
    setActiveConversation(null);
  }, [setActiveConversation]);

  const handleSend = useCallback(
    async (content: string) => {
      const now = new Date().toISOString();
      const userMsg = {
        id: `msg-${Date.now()}`,
        role: "user" as const,
        content,
        citations: [],
        timestamp: now,
      };

      // Only send a conversation_id the backend already knows about.
      const realConvId =
        activeConversationId && !activeConversationId.startsWith(LOCAL_PREFIX)
          ? activeConversationId
          : undefined;

      // Optimistically show the user message.
      let workingId = activeConversationId;
      let workingConvos = conversations;
      if (!workingId) {
        workingId = `${LOCAL_PREFIX}${Date.now()}`;
        workingConvos = [
          {
            id: workingId,
            title: content.slice(0, 40) || "New conversation",
            model_id: activeModelId ?? "",
            messages: [userMsg],
            created_at: now,
            updated_at: now,
          },
          ...conversations,
        ];
        setActiveConversation(workingId);
      } else {
        workingConvos = conversations.map((c) =>
          c.id === workingId
            ? { ...c, messages: [...c.messages, userMsg], updated_at: now }
            : c,
        );
      }
      setConversations(workingConvos);

      setStreaming(true);
      try {
        const response = await queryAPI.send({
          query: content,
          conversation_id: realConvId,
          model_id: activeModelId ?? undefined,
          include_citations: true,
        });
        const assistantMsg = mapMessage(response.message);
        const finalId = response.conversation_id || workingId;
        setConversations(
          workingConvos.map((c) =>
            c.id === workingId
              ? {
                  ...c,
                  id: finalId,
                  messages: [...c.messages, assistantMsg],
                  updated_at: new Date().toISOString(),
                }
              : c,
          ),
        );
        if (finalId !== workingId) setActiveConversation(finalId);
      } catch (err) {
        showToast("error", err instanceof Error ? err.message : "Query failed");
      } finally {
        setStreaming(false);
      }
    },
    [
      activeConversationId,
      activeModelId,
      conversations,
      setConversations,
      setActiveConversation,
      setStreaming,
    ],
  );

  const handleSelectModel = useCallback(
    async (modelId: string) => {
      setActiveModel(modelId);
      const slot = modelSlots.find((m) => m.id === modelId);
      if (slot && slot.status === "unloaded") {
        try {
          await modelAPI.load(modelId);
          await refreshModels();
        } catch (err) {
          showToast(
            "error",
            err instanceof Error ? err.message : "Failed to load model",
          );
        }
      }
    },
    [modelSlots, setActiveModel, refreshModels],
  );

  const handleLoadModel = useCallback(
    async (modelId: string) => {
      try {
        await modelAPI.load(modelId);
        await refreshModels();
        setActiveModel(modelId);
      } catch (err) {
        showToast(
          "error",
          err instanceof Error ? err.message : "Failed to load model",
        );
      }
    },
    [refreshModels, setActiveModel],
  );

  const handleUnloadModel = useCallback(
    async (modelId: string) => {
      try {
        await modelAPI.unload(modelId);
        await refreshModels();
      } catch (err) {
        showToast(
          "error",
          err instanceof Error ? err.message : "Failed to unload model",
        );
      }
    },
    [refreshModels],
  );

  const handleCitationClick = useCallback(
    (citation: Citation) => {
      setActiveCitationId(citation.id);
      if (!citationPanelOpen) toggleCitationPanel();
    },
    [citationPanelOpen, toggleCitationPanel],
  );

  const handleDeleteConversation = useCallback(
    async (id: string) => {
      setConversations(conversations.filter((c) => c.id !== id));
      if (activeConversationId === id) setActiveConversation(null);
      if (!id.startsWith(LOCAL_PREFIX)) {
        try {
          await conversationAPI.delete(id);
        } catch (err) {
          showToast(
            "error",
            err instanceof Error
              ? err.message
              : "Failed to delete conversation",
          );
        }
      }
    },
    [
      conversations,
      activeConversationId,
      setConversations,
      setActiveConversation,
    ],
  );

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <ToolHeader
        icon={Flame}
        title="Hearth"
        color="#D4A058"
        breadcrumb={activeConversation ? [activeConversation.title] : undefined}
      >
        {/* Panel toggles */}
        <button
          onClick={() => setConversationListOpen(!conversationListOpen)}
          className="btn-ghost btn-icon btn-sm"
          title={
            conversationListOpen ? "Hide conversations" : "Show conversations"
          }
        >
          {conversationListOpen ? (
            <PanelLeftClose size={15} />
          ) : (
            <PanelLeft size={15} />
          )}
        </button>
        <button
          onClick={toggleCitationPanel}
          className="btn-ghost btn-icon btn-sm"
          title={citationPanelOpen ? "Hide citations" : "Show citations"}
        >
          {citationPanelOpen ? (
            <PanelRightClose size={15} />
          ) : (
            <PanelRight size={15} />
          )}
        </button>

        {/* Model selector */}
        <ModelSwitcher
          models={modelSlots}
          activeModelId={activeModelId}
          onSelect={handleSelectModel}
          onLoad={handleLoadModel}
          onUnload={handleUnloadModel}
        />
      </ToolHeader>

      {/* Three-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Conversation list */}
        {conversationListOpen && (
          <ConversationList
            conversations={conversations}
            activeId={activeConversationId}
            onSelect={setActiveConversation}
            onNew={handleNewChat}
            onDelete={handleDeleteConversation}
          />
        )}

        {/* Center: Chat area */}
        <div className="flex-1 min-w-0">
          <ChatArea
            messages={activeMessages}
            isStreaming={isStreaming}
            onSend={handleSend}
            onCitationClick={handleCitationClick}
          />
        </div>

        {/* Right: Citation panel */}
        {citationPanelOpen && (
          <CitationPanel
            citations={activeCitations}
            activeCitationId={activeCitationId}
            onClose={toggleCitationPanel}
          />
        )}
      </div>
    </div>
  );
}
