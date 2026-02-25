import { useState } from 'react';
import { Plus, MessageSquare, Search, Trash2 } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { Conversation } from '@/store/useHearthStore';

interface ConversationListProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete?: (id: string) => void;
}

function formatRelativeTime(dateStr: string): string {
  const now = Date.now();
  const date = new Date(dateStr).getTime();
  const diff = now - date;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'now';
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d`;
  return new Date(dateStr).toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export function ConversationList({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
}: ConversationListProps) {
  const [filter, setFilter] = useState('');

  const filtered = filter
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(filter.toLowerCase()),
      )
    : conversations;

  return (
    <div className="w-60 flex flex-col h-full bg-kiln-800 border-r border-kiln-600/50">
      {/* New chat button */}
      <div className="p-3">
        <button
          onClick={onNew}
          className={cn(
            'w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-md',
            'bg-hearth-glow/10 border border-hearth-glow/20 text-hearth-glow',
            'hover:bg-hearth-glow/15 hover:border-hearth-glow/30',
            'transition-all duration-150 text-sm font-medium',
          )}
        >
          <Plus size={15} />
          New Chat
        </button>
      </div>

      {/* Search filter */}
      <div className="px-3 pb-2">
        <div className="relative">
          <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-kiln-500" />
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Search..."
            className="input-field-sm pl-8"
          />
        </div>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-2">
        {filtered.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <MessageSquare size={20} className="mx-auto mb-2 text-kiln-600" />
            <p className="text-xs text-kiln-500">
              {filter ? 'No matching conversations' : 'No conversations yet'}
            </p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {filtered.map((conv) => {
              const isActive = conv.id === activeId;
              return (
                <button
                  key={conv.id}
                  onClick={() => onSelect(conv.id)}
                  className={cn(
                    'group w-full text-left px-3 py-2.5 rounded-md',
                    'transition-all duration-150',
                    isActive
                      ? 'bg-hearth-glow/8 border border-hearth-glow/15'
                      : 'hover:bg-kiln-700/60 border border-transparent',
                  )}
                >
                  <div className="flex items-center justify-between mb-0.5">
                    <span
                      className={cn(
                        'text-xs font-medium truncate flex-1 mr-2',
                        isActive ? 'text-hearth-glow' : 'text-kiln-300',
                      )}
                    >
                      {conv.title}
                    </span>
                    <span className="text-2xs text-kiln-500 flex-shrink-0">
                      {formatRelativeTime(conv.updated_at)}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-2xs text-kiln-500">
                      {conv.messages.length} message{conv.messages.length !== 1 ? 's' : ''}
                    </span>
                    {onDelete && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDelete(conv.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-0.5 text-kiln-500 hover:text-error transition-all"
                        title="Delete conversation"
                      >
                        <Trash2 size={11} />
                      </button>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
