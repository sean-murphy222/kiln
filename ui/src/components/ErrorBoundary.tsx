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
        <div className="h-screen flex items-center justify-center bg-surface-bg p-8">
          <div className="card-pixel max-w-lg w-full">
            {/* Header */}
            <div className="flex items-center gap-3 p-4 border-b border-chonk-slate">
              <div className="w-10 h-10 rounded-full bg-accent-error/20 flex items-center justify-center">
                <AlertTriangle className="text-accent-error" size={24} />
              </div>
              <div>
                <h1 className="text-lg font-bold text-chonk-white">
                  Something went wrong
                </h1>
                <p className="text-sm text-chonk-gray">
                  CHONK encountered an unexpected error
                </p>
              </div>
            </div>

            {/* Error details */}
            <div className="p-4 space-y-4">
              {this.state.error && (
                <div className="p-3 rounded bg-surface-bg border border-accent-error/30">
                  <p className="text-xs font-mono text-accent-error break-all">
                    {this.state.error.message}
                  </p>
                </div>
              )}

              {this.state.errorInfo && (
                <details className="group">
                  <summary className="flex items-center gap-2 text-xs text-chonk-gray cursor-pointer hover:text-chonk-light">
                    <Bug size={14} />
                    <span>Technical details</span>
                  </summary>
                  <pre className="mt-2 p-2 rounded bg-surface-bg text-[10px] font-mono text-chonk-gray overflow-auto max-h-48">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}

              <div className="p-3 rounded bg-surface-card">
                <p className="text-sm text-chonk-light">
                  Your work should be safe. Try refreshing the page or restarting the app.
                  If the problem persists, please report it.
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between p-4 border-t border-chonk-slate">
              <a
                href="https://github.com/anthropics/claude-code/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-chonk-gray hover:text-accent-primary"
              >
                Report issue
              </a>
              <div className="flex gap-2">
                <button
                  className="btn-pixel flex items-center gap-2"
                  onClick={this.handleReset}
                >
                  Try again
                </button>
                <button
                  className="btn-pixel-primary flex items-center gap-2"
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
