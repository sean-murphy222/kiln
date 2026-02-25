import { useState } from 'react';
import { ThumbsUp, ThumbsDown, Clock } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { Message, Citation } from '@/store/useHearthStore';

interface MessageBubbleProps {
  message: Message;
  onCitationClick?: (citation: Citation) => void;
  onFeedback?: (messageId: string, feedback: 'positive' | 'negative') => void;
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Replace [1], [2] etc. with styled citation badges. */
function renderContentWithCitations(
  content: string,
  citations: Citation[],
  onCitationClick?: (citation: Citation) => void,
) {
  const parts = content.split(/(\[\d+\])/g);

  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const index = parseInt(match[1], 10) - 1;
      const citation = citations[index];
      return (
        <button
          key={i}
          className={cn(
            'inline-flex items-center justify-center',
            'w-5 h-5 mx-0.5 rounded text-[10px] font-mono font-bold',
            'bg-hearth-glow/20 text-hearth-glow border border-hearth-glow/30',
            'hover:bg-hearth-glow/30 hover:border-hearth-glow/50',
            'transition-all duration-150 cursor-pointer',
            'align-super -translate-y-0.5',
          )}
          onClick={() => citation && onCitationClick?.(citation)}
          title={citation ? `${citation.document_title} — p.${citation.page}` : `Reference ${match[1]}`}
        >
          {match[1]}
        </button>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function MessageBubble({ message, onCitationClick, onFeedback }: MessageBubbleProps) {
  const [hovering, setHovering] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div
      className={cn(
        'group relative flex w-full',
        isUser ? 'justify-end' : 'justify-start',
      )}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      <div
        className={cn(
          'relative max-w-[85%] rounded-lg px-4 py-3',
          isUser
            ? 'bg-ember/10 border border-ember/15 ml-12'
            : [
                'bg-kiln-800 border border-kiln-600/50 mr-12',
                'shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]',
              ],
        )}
      >
        {/* Content */}
        <div
          className={cn(
            'text-sm leading-relaxed whitespace-pre-wrap',
            isUser ? 'text-kiln-200' : 'text-kiln-300',
          )}
        >
          {isUser
            ? message.content
            : renderContentWithCitations(message.content, message.citations, onCitationClick)}
        </div>

        {/* Timestamp row */}
        <div className={cn(
          'flex items-center gap-2 mt-2 text-2xs text-kiln-500',
          isUser ? 'justify-end' : 'justify-between',
        )}>
          <span className="flex items-center gap-1">
            <Clock size={10} />
            {formatTime(message.timestamp)}
          </span>

          {/* Feedback buttons — assistant only, on hover */}
          {!isUser && (
            <div
              className={cn(
                'flex items-center gap-1 transition-opacity duration-200',
                hovering ? 'opacity-100' : 'opacity-0',
              )}
            >
              <button
                onClick={() => onFeedback?.(message.id, 'positive')}
                className={cn(
                  'p-1 rounded transition-colors',
                  message.feedback === 'positive'
                    ? 'text-success bg-success/10'
                    : 'text-kiln-500 hover:text-success hover:bg-success/10',
                )}
                title="Helpful"
              >
                <ThumbsUp size={12} />
              </button>
              <button
                onClick={() => onFeedback?.(message.id, 'negative')}
                className={cn(
                  'p-1 rounded transition-colors',
                  message.feedback === 'negative'
                    ? 'text-error bg-error/10'
                    : 'text-kiln-500 hover:text-error hover:bg-error/10',
                )}
                title="Not helpful"
              >
                <ThumbsDown size={12} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
