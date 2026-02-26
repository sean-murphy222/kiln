import { useState, useEffect, useCallback } from "react";
import { CheckCircle2, XCircle, AlertTriangle, Info, X } from "lucide-react";
import { cn } from "@/lib/cn";

type ToastType = "success" | "error" | "warning" | "info";

interface ToastData {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

const TOAST_ICONS = {
  success: CheckCircle2,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
} as const;

const TOAST_STYLES = {
  success: "border-success/30 text-success",
  error: "border-error/30 text-error",
  warning: "border-warning/30 text-warning",
  info: "border-info/30 text-info",
} as const;

// Simple global toast state
let toastListeners: Array<(toasts: ToastData[]) => void> = [];
let toastQueue: ToastData[] = [];

export function showToast(type: ToastType, message: string, duration = 4000) {
  const toast: ToastData = {
    id: `toast-${Date.now()}`,
    type,
    message,
    duration,
  };
  toastQueue = [...toastQueue, toast];
  toastListeners.forEach((fn) => fn(toastQueue));

  if (duration > 0) {
    setTimeout(() => {
      toastQueue = toastQueue.filter((t) => t.id !== toast.id);
      toastListeners.forEach((fn) => fn(toastQueue));
    }, duration);
  }
}

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: ToastData;
  onDismiss: () => void;
}) {
  const Icon = TOAST_ICONS[toast.type];

  return (
    <div
      className={cn(
        "card-elevated px-4 py-3 flex items-center gap-3 text-sm animate-slide-up border",
        TOAST_STYLES[toast.type],
      )}
    >
      <Icon size={16} className="shrink-0" />
      <span className="flex-1 text-kiln-200">{toast.message}</span>
      <button
        onClick={onDismiss}
        className="p-0.5 rounded hover:bg-kiln-600 text-kiln-500 transition-colors shrink-0"
      >
        <X size={14} />
      </button>
    </div>
  );
}

export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastData[]>([]);

  useEffect(() => {
    toastListeners.push(setToasts);
    return () => {
      toastListeners = toastListeners.filter((fn) => fn !== setToasts);
    };
  }, []);

  const dismiss = useCallback((id: string) => {
    toastQueue = toastQueue.filter((t) => t.id !== id);
    toastListeners.forEach((fn) => fn(toastQueue));
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <ToastItem
          key={toast.id}
          toast={toast}
          onDismiss={() => dismiss(toast.id)}
        />
      ))}
    </div>
  );
}
