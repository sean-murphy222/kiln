import { useRef, useEffect } from 'react';
import { Flame, MessageSquare } from 'lucide-react';
import { cn } from '@/lib/cn';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import type { Message, Citation } from '@/store/useHearthStore';

interface ChatAreaProps {
  messages: Message[];
  isStreaming: boolean;
  onSend: (message: string) => void;
  onCitationClick?: (citation: Citation) => void;
  onFeedback?: (messageId: string, feedback: 'positive' | 'negative') => void;
}

/** Ember breathing indicator — the hearth is thinking */
function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 px-1">
      <div className="bg-kiln-800 border border-kiln-600/50 rounded-lg px-4 py-3 max-w-[85%]">
        <div className="flex items-center gap-3">
          {/* Breathing ember bar */}
          <div className="relative w-24 h-1.5 rounded-full overflow-hidden bg-kiln-700">
            <div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{
                background: 'linear-gradient(90deg, #D4A058, #E8734A, #D4A058)',
                animation: 'ember-breathe 2s ease-in-out infinite',
              }}
            />
          </div>
          <span className="text-2xs text-kiln-500">thinking</span>
        </div>
      </div>

      <style>{`
        @keyframes ember-breathe {
          0%, 100% { width: 20%; opacity: 0.4; }
          50% { width: 100%; opacity: 1; }
        }
      `}</style>
    </div>
  );
}

/** Empty state when no messages */
function EmptyChat() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center max-w-sm animate-fade-in">
        <div
          className="w-14 h-14 rounded-xl mx-auto mb-5 flex items-center justify-center"
          style={{ background: 'rgba(212, 160, 88, 0.06)' }}
        >
          <Flame size={24} className="text-hearth-glow/60" strokeWidth={1.5} />
        </div>
        <h3 className="font-display text-base font-semibold text-kiln-300 mb-2">
          Start a conversation
        </h3>
        <p className="text-sm text-kiln-500 leading-relaxed">
          Ask questions about your processed documents.
          Responses include citations to source material.
        </p>
      </div>
    </div>
  );
}

export function ChatArea({
  messages,
  isStreaming,
  onSend,
  onCitationClick,
  onFeedback,
}: ChatAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length, isStreaming]);

  return (
    <div className="flex flex-col h-full min-w-0">
      {/* Messages area */}
      {messages.length === 0 ? (
        <EmptyChat />
      ) : (
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-6 py-6 space-y-4">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onCitationClick={onCitationClick}
                onFeedback={onFeedback}
              />
            ))}

            {isStreaming && <TypingIndicator />}

            <div ref={bottomRef} />
          </div>
        </div>
      )}

      {/* Input area */}
      <ChatInput
        onSend={onSend}
        disabled={isStreaming}
      />
    </div>
  );
}
