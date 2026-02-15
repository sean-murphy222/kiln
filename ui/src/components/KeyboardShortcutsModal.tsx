import { X, Keyboard } from 'lucide-react';

interface KeyboardShortcutsModalProps {
  onClose: () => void;
}

interface Shortcut {
  keys: string[];
  description: string;
}

const shortcuts: Shortcut[] = [
  { keys: ['Ctrl', 'O'], description: 'Open file' },
  { keys: ['Ctrl', 'S'], description: 'Save project' },
  { keys: ['Ctrl', 'E'], description: 'Export chunks' },
  { keys: ['Ctrl', ','], description: 'Open settings' },
  { keys: ['Ctrl', 'B'], description: 'Toggle sidebar' },
  { keys: ['Ctrl', 'T'], description: 'Toggle test panel' },
  { keys: ['Ctrl', 'A'], description: 'Select all chunks' },
  { keys: ['Esc'], description: 'Clear selection' },
  { keys: ['↑', '↓'], description: 'Navigate chunks' },
  { keys: ['Shift', '↑/↓'], description: 'Extend selection' },
  { keys: ['Ctrl', 'Click'], description: 'Multi-select chunks' },
  { keys: ['1-9'], description: 'Quick select document' },
];

export function KeyboardShortcutsModal({ onClose }: KeyboardShortcutsModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="card-pixel w-full max-w-md animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-chonk-slate">
          <div className="flex items-center gap-2">
            <Keyboard size={18} className="text-accent-primary" />
            <h2 className="text-sm font-medium text-chonk-white">
              Keyboard Shortcuts
            </h2>
          </div>
          <button
            className="p-1 text-chonk-gray hover:text-chonk-white"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 max-h-[60vh] overflow-y-auto">
          <div className="space-y-2">
            {shortcuts.map((shortcut, index) => (
              <div
                key={index}
                className="flex items-center justify-between py-2 px-3 rounded bg-surface-card"
              >
                <span className="text-sm text-chonk-light">
                  {shortcut.description}
                </span>
                <div className="flex items-center gap-1">
                  {shortcut.keys.map((key, keyIndex) => (
                    <span key={keyIndex}>
                      <kbd className="px-2 py-1 text-xs font-mono bg-surface-bg border border-chonk-slate rounded text-chonk-white">
                        {key}
                      </kbd>
                      {keyIndex < shortcut.keys.length - 1 && (
                        <span className="text-chonk-gray mx-1">+</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 p-3 rounded bg-surface-bg border border-chonk-slate">
            <p className="text-xs text-chonk-gray">
              <span className="text-accent-primary">Tip:</span> On macOS, use{' '}
              <kbd className="px-1 py-0.5 text-[10px] font-mono bg-surface-card border border-chonk-slate rounded">
                Cmd
              </kbd>{' '}
              instead of{' '}
              <kbd className="px-1 py-0.5 text-[10px] font-mono bg-surface-card border border-chonk-slate rounded">
                Ctrl
              </kbd>
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end px-4 py-3 border-t border-chonk-slate">
          <button className="btn-pixel" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
