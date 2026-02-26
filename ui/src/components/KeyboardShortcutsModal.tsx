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
      <div className="card w-full max-w-md animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-kiln-600">
          <div className="flex items-center gap-2">
            <Keyboard size={18} className="text-ember" />
            <h2 className="text-sm font-medium text-kiln-100">
              Keyboard Shortcuts
            </h2>
          </div>
          <button
            className="p-1 text-kiln-500 hover:text-kiln-100"
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
                className="flex items-center justify-between py-2 px-3 rounded bg-kiln-700"
              >
                <span className="text-sm text-kiln-300">
                  {shortcut.description}
                </span>
                <div className="flex items-center gap-1">
                  {shortcut.keys.map((key, keyIndex) => (
                    <span key={keyIndex}>
                      <kbd className="px-2 py-1 text-xs font-mono bg-kiln-900 border border-kiln-600 rounded text-kiln-100">
                        {key}
                      </kbd>
                      {keyIndex < shortcut.keys.length - 1 && (
                        <span className="text-kiln-500 mx-1">+</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 p-3 rounded bg-kiln-900 border border-kiln-600">
            <p className="text-xs text-kiln-500">
              <span className="text-ember">Tip:</span> On macOS, use{' '}
              <kbd className="px-1 py-0.5 text-[10px] font-mono bg-kiln-700 border border-kiln-600 rounded">
                Cmd
              </kbd>{' '}
              instead of{' '}
              <kbd className="px-1 py-0.5 text-[10px] font-mono bg-kiln-700 border border-kiln-600 rounded">
                Ctrl
              </kbd>
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end px-4 py-3 border-t border-kiln-600">
          <button className="btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
