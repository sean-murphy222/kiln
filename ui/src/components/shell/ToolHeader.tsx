import type { ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import { cn } from '@/lib/cn';

interface ToolHeaderProps {
  icon: React.ElementType;
  title: string;
  color: string;
  breadcrumb?: string[];
  children?: ReactNode;
  className?: string;
}

export function ToolHeader({
  icon: Icon,
  title,
  color,
  breadcrumb,
  children,
  className,
}: ToolHeaderProps) {
  return (
    <header
      className={cn(
        'h-12 flex items-center justify-between px-5 border-b border-kiln-600/70',
        'bg-kiln-800/80 backdrop-blur-sm',
        className,
      )}
    >
      {/* Left: Tool identity + breadcrumb */}
      <div className="flex items-center gap-3 min-w-0">
        {/* Tool icon with color identity */}
        <div
          className="flex items-center justify-center w-7 h-7 rounded-md"
          style={{ background: `${color}12` }}
        >
          <Icon size={15} style={{ color }} strokeWidth={2} />
        </div>

        {/* Title */}
        <h1
          className="font-display text-sm font-semibold tracking-tight"
          style={{ color }}
        >
          {title}
        </h1>

        {/* Breadcrumb trail */}
        {breadcrumb && breadcrumb.length > 0 && (
          <div className="flex items-center gap-1 text-xs text-kiln-500 min-w-0">
            {breadcrumb.map((segment, i) => (
              <span key={i} className="flex items-center gap-1 min-w-0">
                <ChevronRight size={12} className="flex-shrink-0 text-kiln-600" />
                <span className={cn(
                  'truncate',
                  i === breadcrumb.length - 1 ? 'text-kiln-300' : 'text-kiln-500',
                )}>
                  {segment}
                </span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Right: Action buttons */}
      {children && (
        <div className="flex items-center gap-2 flex-shrink-0">
          {children}
        </div>
      )}
    </header>
  );
}
