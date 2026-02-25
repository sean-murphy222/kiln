import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Cpu, Loader2, AlertCircle, Power, PowerOff } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { ModelSlot } from '@/store/useHearthStore';

interface ModelSwitcherProps {
  models: ModelSlot[];
  activeModelId: string | null;
  onSelect: (modelId: string) => void;
  onLoad?: (modelId: string) => void;
  onUnload?: (modelId: string) => void;
}

function statusDot(status: ModelSlot['status']) {
  switch (status) {
    case 'ready':
      return 'bg-success shadow-[0_0_6px_rgba(92,184,122,0.4)]';
    case 'loading':
      return 'bg-warning animate-pulse';
    case 'error':
      return 'bg-error';
    case 'unloaded':
      return 'bg-kiln-500';
  }
}

function statusIcon(status: ModelSlot['status']) {
  switch (status) {
    case 'ready':
      return <Cpu size={12} className="text-success" />;
    case 'loading':
      return <Loader2 size={12} className="text-warning animate-spin" />;
    case 'error':
      return <AlertCircle size={12} className="text-error" />;
    case 'unloaded':
      return <Cpu size={12} className="text-kiln-500" />;
  }
}

export function ModelSwitcher({
  models,
  activeModelId,
  onSelect,
  onLoad,
  onUnload,
}: ModelSwitcherProps) {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const activeModel = models.find((m) => m.id === activeModelId);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div ref={dropdownRef} className="relative">
      {/* Trigger */}
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-md text-xs',
          'border transition-all duration-150',
          open
            ? 'bg-kiln-700 border-kiln-500 text-kiln-200'
            : 'bg-kiln-800 border-kiln-600 text-kiln-400 hover:text-kiln-300 hover:border-kiln-500',
        )}
      >
        {activeModel ? (
          <>
            <span className={cn('w-2 h-2 rounded-full', statusDot(activeModel.status))} />
            <span className="font-medium">{activeModel.name}</span>
          </>
        ) : (
          <span className="text-kiln-500">No model</span>
        )}
        <ChevronDown size={12} className={cn('transition-transform', open && 'rotate-180')} />
      </button>

      {/* Dropdown */}
      {open && (
        <div
          className={cn(
            'absolute right-0 top-full mt-1 w-72 z-50',
            'bg-kiln-800 border border-kiln-600 rounded-lg shadow-kiln-lg',
            'animate-fade-in',
          )}
        >
          <div className="p-2">
            <div className="px-2 py-1.5 text-2xs font-semibold text-kiln-500 uppercase tracking-wider">
              Model Slots
            </div>

            {models.length === 0 ? (
              <div className="px-2 py-4 text-center text-xs text-kiln-500">
                No models registered
              </div>
            ) : (
              <div className="space-y-0.5">
                {models.map((model) => {
                  const isActive = model.id === activeModelId;
                  return (
                    <div
                      key={model.id}
                      className={cn(
                        'flex items-center gap-2 px-2 py-2 rounded-md',
                        'transition-colors duration-100',
                        isActive
                          ? 'bg-hearth-glow/8'
                          : 'hover:bg-kiln-700',
                      )}
                    >
                      {/* Select model */}
                      <button
                        onClick={() => {
                          onSelect(model.id);
                          setOpen(false);
                        }}
                        className="flex-1 flex items-center gap-2 text-left min-w-0"
                      >
                        {statusIcon(model.status)}
                        <div className="min-w-0">
                          <div className={cn(
                            'text-xs font-medium truncate',
                            isActive ? 'text-hearth-glow' : 'text-kiln-200',
                          )}>
                            {model.name}
                          </div>
                          <div className="text-2xs text-kiln-500 truncate">
                            {model.base_model}
                          </div>
                        </div>
                      </button>

                      {/* Load/Unload action */}
                      {model.status === 'unloaded' && onLoad && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onLoad(model.id);
                          }}
                          className="p-1.5 text-kiln-500 hover:text-success rounded transition-colors"
                          title="Load model"
                        >
                          <Power size={13} />
                        </button>
                      )}
                      {model.status === 'ready' && onUnload && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onUnload(model.id);
                          }}
                          className="p-1.5 text-kiln-500 hover:text-warning rounded transition-colors"
                          title="Unload model"
                        >
                          <PowerOff size={13} />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
