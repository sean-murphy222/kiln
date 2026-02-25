import { useState, useCallback } from 'react';
import { Flame, PanelLeftClose, PanelLeft, PanelRightClose, PanelRight } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useHearthStore } from '@/store/useHearthStore';
import type { Citation } from '@/store/useHearthStore';
import { ToolHeader } from '@/components/shell/ToolHeader';
import { ConversationList } from './ConversationList';
import { ChatArea } from './ChatArea';
import { CitationPanel } from './CitationPanel';
import { ModelSwitcher } from './ModelSwitcher';

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
    addMessage,
    setActiveModel,
    toggleCitationPanel,
    setStreaming,
  } = useHearthStore();

  const [conversationListOpen, setConversationListOpen] = useState(true);
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);

  const activeConversation = conversations.find((c) => c.id === activeConversationId);
  const activeMessages = activeConversation?.messages ?? [];

  // Citations from the last assistant message
  const lastAssistantMsg = [...activeMessages].reverse().find((m) => m.role === 'assistant');
  const activeCitations = lastAssistantMsg?.citations ?? [];

  const handleNewChat = useCallback(() => {
    const newConv = {
      id: `conv-${Date.now()}`,
      title: 'New conversation',
      model_id: activeModelId ?? '',
      messages: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    setConversations([newConv, ...conversations]);
    setActiveConversation(newConv.id);
  }, [conversations, activeModelId, setConversations, setActiveConversation]);

  const handleSend = useCallback(
    (content: string) => {
      if (!activeConversationId) {
        // Auto-create conversation
        handleNewChat();
        return;
      }

      // Add user message
      const userMsg = {
        id: `msg-${Date.now()}`,
        role: 'user' as const,
        content,
        citations: [],
        timestamp: new Date().toISOString(),
      };
      addMessage(activeConversationId, userMsg);

      // Simulate assistant response (will be replaced by real API call)
      setStreaming(true);
      setTimeout(() => {
        const assistantMsg = {
          id: `msg-${Date.now() + 1}`,
          role: 'assistant' as const,
          content:
            'This is a placeholder response. The Hearth backend API is required for real inference. ' +
            'When connected, responses will include citations like [1] and [2] that reference your processed documents.',
          citations: [
            {
              id: `cit-${Date.now()}`,
              document_title: 'TM-9-2320-280-10',
              section: '2.3 Engine Maintenance',
              page: 42,
              relevance_score: 0.92,
              snippet:
                'Preventive maintenance checks should be performed at regular intervals as specified in the maintenance allocation chart.',
            },
            {
              id: `cit-${Date.now() + 1}`,
              document_title: 'TM-9-2320-280-10',
              section: '4.1 Troubleshooting',
              page: 87,
              relevance_score: 0.76,
              snippet:
                'If the engine fails to start after three attempts, check the fuel supply, battery connections, and starter motor relay.',
            },
          ],
          timestamp: new Date().toISOString(),
        };
        addMessage(activeConversationId, assistantMsg);
        setStreaming(false);
      }, 1500);
    },
    [activeConversationId, addMessage, setStreaming, handleNewChat],
  );

  const handleCitationClick = useCallback((citation: Citation) => {
    setActiveCitationId(citation.id);
    if (!citationPanelOpen) toggleCitationPanel();
  }, [citationPanelOpen, toggleCitationPanel]);

  const handleDeleteConversation = useCallback(
    (id: string) => {
      setConversations(conversations.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        setActiveConversation(null);
      }
    },
    [conversations, activeConversationId, setConversations, setActiveConversation],
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
          title={conversationListOpen ? 'Hide conversations' : 'Show conversations'}
        >
          {conversationListOpen ? <PanelLeftClose size={15} /> : <PanelLeft size={15} />}
        </button>
        <button
          onClick={toggleCitationPanel}
          className="btn-ghost btn-icon btn-sm"
          title={citationPanelOpen ? 'Hide citations' : 'Show citations'}
        >
          {citationPanelOpen ? <PanelRightClose size={15} /> : <PanelRight size={15} />}
        </button>

        {/* Model selector */}
        <ModelSwitcher
          models={modelSlots}
          activeModelId={activeModelId}
          onSelect={setActiveModel}
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
