import { useEffect, useState, useCallback } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useStore } from "./store/useStore";
import { projectAPI, utilAPI } from "./api/chonk";
import { AppShell } from "./components/shell/AppShell";
import { Layout } from "./components/Layout";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ForgeLayout } from "./components/forge/ForgeLayout";
import { FoundryLayout } from "./components/foundry/FoundryLayout";
import { HearthLayout } from "./components/hearth/HearthLayout";
import { SettingsPage } from "./components/settings/SettingsPage";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ErrorToast } from "./components/ErrorToast";
import { ToastContainer } from "./components/common/Toast";

declare global {
  interface Window {
    electronAPI?: {
      openFile: () => Promise<{ canceled: boolean; filePaths: string[] }>;
      openProject: () => Promise<{ canceled: boolean; filePaths: string[] }>;
      saveFile: (
        defaultPath?: string,
      ) => Promise<{ canceled: boolean; filePath?: string }>;
      saveProject: (
        defaultPath?: string,
      ) => Promise<{ canceled: boolean; filePath?: string }>;
      platform: string;
    };
  }
}

function QuarryRoute() {
  const { project, setProject, setLoading, setError, isLoading } = useStore();
  const [backendReady, setBackendReady] = useState(false);
  const [checkingBackend, setCheckingBackend] = useState(true);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await utilAPI.healthCheck();
        setBackendReady(true);
      } catch {
        setBackendReady(false);
      } finally {
        setCheckingBackend(false);
      }
    };

    checkBackend();
    const interval = setInterval(() => {
      if (!backendReady) checkBackend();
    }, 2000);
    return () => clearInterval(interval);
  }, [backendReady]);

  const handleNewProject = useCallback(
    async (name: string) => {
      setLoading(true);
      setError(null);
      try {
        await projectAPI.create(name);
        const proj = await projectAPI.get();
        setProject(proj);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to create project",
        );
      } finally {
        setLoading(false);
      }
    },
    [setProject, setLoading, setError],
  );

  const handleOpenProject = useCallback(async () => {
    if (!window.electronAPI) {
      setError("File dialogs only available in Electron");
      return;
    }
    const result = await window.electronAPI.openProject();
    if (result.canceled || !result.filePaths[0]) return;

    setLoading(true);
    setError(null);
    try {
      await projectAPI.open(result.filePaths[0]);
      const proj = await projectAPI.get();
      setProject(proj);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to open project");
    } finally {
      setLoading(false);
    }
  }, [setProject, setLoading, setError]);

  if (checkingBackend) {
    return (
      <div className="h-full flex flex-col items-center justify-center">
        <div className="text-sm text-kiln-400 animate-pulse-soft">
          Connecting to backend...
        </div>
      </div>
    );
  }

  if (!backendReady) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-6">
        <div className="font-display text-2xl font-bold text-ember">Kiln</div>
        <div className="text-kiln-400 text-center max-w-md">
          <p className="mb-4 text-sm">Backend server is not running.</p>
          <code className="block mt-2 p-3 bg-kiln-800 border border-kiln-600 rounded-kiln font-mono text-xs text-kiln-300">
            uvicorn kiln_server:app --port 8420
          </code>
        </div>
        <div className="text-2xs text-kiln-500 animate-pulse-soft">
          Retrying connection...
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <WelcomeScreen
        onNewProject={handleNewProject}
        onOpenProject={handleOpenProject}
        isLoading={isLoading}
      />
    );
  }

  return <Layout />;
}

function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/quarry" element={<QuarryRoute />} />
          <Route path="/forge" element={<ForgeLayout />} />
          <Route path="/foundry" element={<FoundryLayout />} />
          <Route path="/hearth" element={<HearthLayout />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/" element={<Navigate to="/quarry" replace />} />
          <Route path="*" element={<Navigate to="/quarry" replace />} />
        </Route>
      </Routes>
      <ErrorToast />
      <ToastContainer />
    </ErrorBoundary>
  );
}

export default App;
