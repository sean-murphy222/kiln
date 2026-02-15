import { useEffect, useState, useCallback } from 'react';
import { useStore } from './store/useStore';
import { projectAPI, utilAPI } from './api/chonk';
import { Layout } from './components/Layout';
import { WelcomeScreen } from './components/WelcomeScreen';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ErrorToast } from './components/ErrorToast';

declare global {
  interface Window {
    electronAPI?: {
      openFile: () => Promise<{ canceled: boolean; filePaths: string[] }>;
      openProject: () => Promise<{ canceled: boolean; filePaths: string[] }>;
      saveFile: (defaultPath?: string) => Promise<{ canceled: boolean; filePath?: string }>;
      saveProject: (defaultPath?: string) => Promise<{ canceled: boolean; filePath?: string }>;
      platform: string;
    };
  }
}

function App() {
  const { project, setProject, setLoading, setError, isLoading } = useStore();
  const [backendReady, setBackendReady] = useState(false);
  const [checkingBackend, setCheckingBackend] = useState(true);

  // Check if backend is running
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

    // Retry every 2 seconds if not ready
    const interval = setInterval(() => {
      if (!backendReady) {
        checkBackend();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [backendReady]);

  // Create new project
  const handleNewProject = useCallback(async (name: string) => {
    setLoading(true);
    setError(null);
    try {
      await projectAPI.create(name);
      const project = await projectAPI.get();
      setProject(project);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
    } finally {
      setLoading(false);
    }
  }, [setProject, setLoading, setError]);

  // Open existing project
  const handleOpenProject = useCallback(async () => {
    if (!window.electronAPI) {
      setError('File dialogs only available in Electron');
      return;
    }

    const result = await window.electronAPI.openProject();
    if (result.canceled || !result.filePaths[0]) return;

    setLoading(true);
    setError(null);
    try {
      await projectAPI.open(result.filePaths[0]);
      const project = await projectAPI.get();
      setProject(project);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open project');
    } finally {
      setLoading(false);
    }
  }, [setProject, setLoading, setError]);

  // Show loading while checking backend
  if (checkingBackend) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-surface-bg">
        <div className="text-pixel text-accent-primary animate-pulse-pixel">
          LOADING...
        </div>
      </div>
    );
  }

  // Show backend connection error
  if (!backendReady) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-surface-bg gap-6">
        <div className="text-pixel text-4xl text-accent-error">
          CHONK
        </div>
        <div className="text-chonk-light text-center max-w-md">
          <p className="mb-4">Backend server is not running.</p>
          <p className="text-sm text-chonk-gray">
            Start the server with:
          </p>
          <code className="block mt-2 p-3 bg-surface-panel rounded font-mono text-sm">
            cd src && uvicorn chonk.server:app --port 8420
          </code>
        </div>
        <div className="text-xs text-chonk-gray animate-pulse">
          Retrying connection...
        </div>
      </div>
    );
  }

  // Show welcome screen if no project
  if (!project) {
    return (
      <ErrorBoundary>
        <WelcomeScreen
          onNewProject={handleNewProject}
          onOpenProject={handleOpenProject}
          isLoading={isLoading}
        />
        <ErrorToast />
      </ErrorBoundary>
    );
  }

  // Show main layout
  return (
    <ErrorBoundary>
      <Layout />
      <ErrorToast />
    </ErrorBoundary>
  );
}

export default App;
