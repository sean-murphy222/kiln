import { Outlet } from 'react-router-dom';
import { NavRail } from './NavRail';

export function AppShell() {
  return (
    <div className="h-screen flex bg-kiln-900 overflow-hidden">
      {/* Navigation rail */}
      <NavRail />

      {/* Content area — each tool route renders here */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {/* Faint left-edge shadow for depth against nav rail */}
        <div className="absolute left-0 top-0 bottom-0 w-px bg-gradient-to-b from-kiln-600/30 via-kiln-600/10 to-kiln-600/30 z-10 pointer-events-none" />
        <Outlet />
      </main>
    </div>
  );
}
