import { useState, useRef, useCallback, useEffect } from 'react';
import { Send } from 'lucide-react';
import { cn } from '@/lib/cn';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

const MAX_ROWS = 6;
const LINE_HEIGHT = 22;

export function ChatInput({ onSend, disabled = false, placeholder }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const maxHeight = MAX_ROWS * LINE_HEIGHT;
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="border-t border-kiln-600/50 bg-kiln-800/50 p-4">
      <div
        className={cn(
          'flex items-end gap-3 rounded-lg border transition-colors duration-150',
          'bg-kiln-900 px-4 py-3',
          disabled
            ? 'border-kiln-600/30 opacity-60'
            : 'border-kiln-600 focus-within:border-hearth-glow/40',
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder ?? 'Ask a question about your documents...'}
          disabled={disabled}
          rows={1}
          className={cn(
            'flex-1 bg-transparent resize-none text-sm text-kiln-200',
            'placeholder:text-kiln-500 outline-none',
            'leading-[22px]',
          )}
          style={{ maxHeight: MAX_ROWS * LINE_HEIGHT }}
        />

        <div className="flex items-center gap-2 flex-shrink-0 pb-0.5">
          {/* Character count */}
          {value.length > 0 && (
            <span className="text-2xs text-kiln-600 tabular-nums">
              {value.length}
            </span>
          )}

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!value.trim() || disabled}
            className={cn(
              'w-8 h-8 flex items-center justify-center rounded-md transition-all duration-150',
              value.trim() && !disabled
                ? 'bg-hearth-glow text-kiln-900 hover:bg-hearth-glow-light active:scale-95'
                : 'bg-kiln-700 text-kiln-500 cursor-not-allowed',
            )}
            title="Send message (Enter)"
          >
            <Send size={14} />
          </button>
        </div>
      </div>

      <div className="flex items-center justify-between mt-2 px-1">
        <span className="text-2xs text-kiln-600">
          Enter to send · Shift+Enter for new line
        </span>
        {disabled && (
          <span className="text-2xs text-hearth-glow animate-pulse-soft">
            Generating response...
          </span>
        )}
      </div>
    </div>
  );
}
