import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div className={cn("flex items-center justify-center py-16", className)}>
      <div className="text-center max-w-sm animate-fade-in">
        <div className="w-14 h-14 rounded-2xl bg-kiln-700/50 flex items-center justify-center mx-auto mb-4">
          <Icon size={24} className="text-kiln-500" strokeWidth={1.5} />
        </div>
        <h3 className="font-display text-sm font-semibold text-kiln-300 mb-1">
          {title}
        </h3>
        {description && (
          <p className="text-2xs text-kiln-500 leading-relaxed mb-4">
            {description}
          </p>
        )}
        {action && (
          <button onClick={action.onClick} className="btn-primary btn-sm">
            {action.label}
          </button>
        )}
      </div>
    </div>
  );
}
