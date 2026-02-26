import { useEffect, useState } from 'react';
import { X, AlertCircle, AlertTriangle, Info, CheckCircle } from 'lucide-react';
import { useStore } from '../store/useStore';

type ToastType = 'error' | 'warning' | 'info' | 'success';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

export function ErrorToast() {
  const { error, setError } = useStore();
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Add toast when error changes
  useEffect(() => {
    if (error) {
      const newToast: Toast = {
        id: Date.now().toString(),
        message: error,
        type: 'error',
        duration: 5000,
      };

      setToasts((prev) => [...prev, newToast]);

      // Clear store error after showing
      setTimeout(() => setError(null), 100);
    }
  }, [error, setError]);

  // Remove toast after duration
  useEffect(() => {
    if (toasts.length === 0) return;

    const timer = setTimeout(() => {
      setToasts((prev) => prev.slice(1));
    }, toasts[0]?.duration || 5000);

    return () => clearTimeout(timer);
  }, [toasts]);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const getIcon = (type: ToastType) => {
    switch (type) {
      case 'error':
        return <AlertCircle size={18} className="text-error" />;
      case 'warning':
        return <AlertTriangle size={18} className="text-warning" />;
      case 'info':
        return <Info size={18} className="text-ember" />;
      case 'success':
        return <CheckCircle size={18} className="text-success" />;
    }
  };

  const getBorderColor = (type: ToastType) => {
    switch (type) {
      case 'error':
        return 'border-error';
      case 'warning':
        return 'border-warning';
      case 'info':
        return 'border-ember';
      case 'success':
        return 'border-success';
    }
  };

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 left-4 z-50 flex flex-col gap-2 max-w-md">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`
            flex items-start gap-3 p-3 rounded-lg
            bg-kiln-800 border-l-4 shadow-lg
            animate-slide-up
            ${getBorderColor(toast.type)}
          `}
        >
          {getIcon(toast.type)}
          <p className="flex-1 text-sm text-kiln-300">{toast.message}</p>
          <button
            className="text-kiln-500 hover:text-kiln-100"
            onClick={() => removeToast(toast.id)}
          >
            <X size={16} />
          </button>
        </div>
      ))}
    </div>
  );
}

// Utility function to show custom toasts
export function showToast(message: string, type: ToastType = 'info') {
  // This could be expanded to use a global toast manager
  console.log(`[${type.toUpperCase()}] ${message}`);
}
