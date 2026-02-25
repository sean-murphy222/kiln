import { useEffect, useCallback } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Mountain, Hammer, Factory, Flame, Settings } from 'lucide-react';
import { cn } from '@/lib/cn';

interface ToolDef {
  id: string;
  label: string;
  path: string;
  icon: React.ElementType;
  color: string;
  glowColor: string;
  bgTint: string;
  shortcut: string;
}

const TOOLS: ToolDef[] = [
  {
    id: 'quarry',
    label: 'Quarry',
    path: '/quarry',
    icon: Mountain,
    color: '#7C92A8',
    glowColor: 'rgba(124, 146, 168, 0.35)',
    bgTint: 'rgba(124, 146, 168, 0.08)',
    shortcut: '1',
  },
  {
    id: 'forge',
    label: 'Forge',
    path: '/forge',
    icon: Hammer,
    color: '#D4915C',
    glowColor: 'rgba(212, 145, 92, 0.35)',
    bgTint: 'rgba(212, 145, 92, 0.08)',
    shortcut: '2',
  },
  {
    id: 'foundry',
    label: 'Foundry',
    path: '/foundry',
    icon: Factory,
    color: '#6BA089',
    glowColor: 'rgba(107, 160, 137, 0.35)',
    bgTint: 'rgba(107, 160, 137, 0.08)',
    shortcut: '3',
  },
  {
    id: 'hearth',
    label: 'Hearth',
    path: '/hearth',
    icon: Flame,
    color: '#D4A058',
    glowColor: 'rgba(212, 160, 88, 0.35)',
    bgTint: 'rgba(212, 160, 88, 0.08)',
    shortcut: '4',
  },
];

export function NavRail() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      const tool = TOOLS.find((t) => t.shortcut === e.key);
      if (tool) {
        e.preventDefault();
        navigate(tool.path);
      }
      if (e.key === ',') {
        e.preventDefault();
        navigate('/settings');
      }
    },
    [navigate],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return (
    <nav
      className="nav-rail h-full flex-shrink-0 relative z-10"
      aria-label="Tool navigation"
    >
      {/* Subtle top highlight edge */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-kiln-500/20 to-transparent" />

      {/* Tool items */}
      <div className="flex flex-col items-center gap-1 flex-1">
        {TOOLS.map((tool) => {
          const Icon = tool.icon;
          const isActive = location.pathname.startsWith(tool.path);

          return (
            <NavLink
              key={tool.id}
              to={tool.path}
              className="group relative w-12 flex flex-col items-center gap-1 py-2.5 rounded-md
                         transition-all duration-200 outline-none
                         focus-visible:ring-2 focus-visible:ring-ember/50"
              style={{
                background: isActive ? tool.bgTint : undefined,
                color: isActive ? tool.color : undefined,
              }}
              title={`${tool.label} (Ctrl+${tool.shortcut})`}
              aria-label={`${tool.label} tool`}
            >
              {/* Active indicator — glowing left bar */}
              {isActive && (
                <span
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-6 rounded-r-full
                             transition-all duration-300"
                  style={{
                    background: tool.color,
                    boxShadow: `0 0 8px ${tool.glowColor}, 0 0 20px ${tool.glowColor}`,
                  }}
                />
              )}

              <Icon
                size={20}
                strokeWidth={isActive ? 2 : 1.5}
                className={cn(
                  'transition-all duration-200',
                  isActive
                    ? 'drop-shadow-sm'
                    : 'text-kiln-500 group-hover:text-kiln-300',
                )}
              />

              <span
                className={cn(
                  'text-[9px] font-medium leading-none tracking-wide transition-colors duration-200',
                  isActive ? 'opacity-100' : 'text-kiln-500 opacity-70 group-hover:opacity-100 group-hover:text-kiln-400',
                )}
              >
                {tool.label}
              </span>
            </NavLink>
          );
        })}
      </div>

      {/* Bottom section — Settings */}
      <div className="flex flex-col items-center gap-2 pb-2">
        <div className="w-8 h-px bg-kiln-600/50 mb-1" />
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            cn(
              'w-10 h-10 flex items-center justify-center rounded-md transition-all duration-200',
              'outline-none focus-visible:ring-2 focus-visible:ring-ember/50',
              isActive
                ? 'bg-kiln-700 text-kiln-200'
                : 'text-kiln-500 hover:text-kiln-400 hover:bg-kiln-700/50',
            )
          }
          title="Settings (Ctrl+,)"
          aria-label="Settings"
        >
          <Settings size={18} strokeWidth={1.5} />
        </NavLink>
      </div>
    </nav>
  );
}

export { TOOLS };
export type { ToolDef };
