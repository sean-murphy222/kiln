import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Bug } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('CHONK Error Boundary caught error:', error, errorInfo);
    this.setState({ errorInfo });

    // In production, you might send this to an error reporting service
    // logErrorToService(error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="h-screen flex items-center justify-center bg-kiln-900 p-8">
          <div className="card max-w-lg w-full">
            {/* Header */}
            <div className="flex items-center gap-3 p-4 border-b border-kiln-600">
              <div className="w-10 h-10 rounded-full bg-error/20 flex items-center justify-center">
                <AlertTriangle className="text-error" size={24} />
              </div>
              <div>
                <h1 className="text-lg font-bold text-kiln-100">
                  Something went wrong
                </h1>
                <p className="text-sm text-kiln-500">
                  CHONK encountered an unexpected error
                </p>
              </div>
            </div>

            {/* Error details */}
            <div className="p-4 space-y-4">
              {this.state.error && (
                <div className="p-3 rounded bg-kiln-900 border border-error/30">
                  <p className="text-xs font-mono text-error break-all">
                    {this.state.error.message}
                  </p>
                </div>
              )}

              {this.state.errorInfo && (
                <details className="group">
                  <summary className="flex items-center gap-2 text-xs text-kiln-500 cursor-pointer hover:text-kiln-300">
                    <Bug size={14} />
                    <span>Technical details</span>
                  </summary>
                  <pre className="mt-2 p-2 rounded bg-kiln-900 text-[10px] font-mono text-kiln-500 overflow-auto max-h-48">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}

              <div className="p-3 rounded bg-kiln-700">
                <p className="text-sm text-kiln-300">
                  Your work should be safe. Try refreshing the page or restarting the app.
                  If the problem persists, please report it.
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between p-4 border-t border-kiln-600">
              <a
                href="https://github.com/anthropics/claude-code/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-kiln-500 hover:text-ember"
              >
                Report issue
              </a>
              <div className="flex gap-2">
                <button
                  className="btn-secondary flex items-center gap-2"
                  onClick={this.handleReset}
                >
                  Try again
                </button>
                <button
                  className="btn-primary flex items-center gap-2"
                  onClick={this.handleReload}
                >
                  <RefreshCw size={14} />
                  Reload
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
