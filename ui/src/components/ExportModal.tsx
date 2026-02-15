import { useState } from 'react';
import { X, Download, FileJson, FileSpreadsheet, FileText } from 'lucide-react';
import { useStore } from '../store/useStore';
import { exportAPI } from '../api/chonk';

interface ExportModalProps {
  documentId?: string;
  documentName?: string;
  onClose: () => void;
}

export function ExportModal({ documentId, documentName, onClose }: ExportModalProps) {
  const { setError } = useStore();

  const [format, setFormat] = useState('jsonl');
  const [isExporting, setIsExporting] = useState(false);

  const formats = [
    {
      id: 'jsonl',
      name: 'JSONL',
      description: 'Newline-delimited JSON (LangChain, LlamaIndex compatible)',
      icon: FileJson,
      extension: '.jsonl',
    },
    {
      id: 'json',
      name: 'JSON',
      description: 'Full JSON with metadata and structure',
      icon: FileJson,
      extension: '.json',
    },
    {
      id: 'csv',
      name: 'CSV',
      description: 'Spreadsheet-friendly format',
      icon: FileSpreadsheet,
      extension: '.csv',
    },
  ];

  const handleExport = async () => {
    // Check for Electron API
    if (!window.electronAPI) {
      setError('Export requires desktop app');
      return;
    }

    const selectedFormat = formats.find((f) => f.id === format);
    const defaultName = documentName
      ? `${documentName.replace(/\.[^/.]+$/, '')}_chunks${selectedFormat?.extension}`
      : `chunks${selectedFormat?.extension}`;

    const result = await window.electronAPI.saveFile(defaultName);
    if (result.canceled || !result.filePath) return;

    setIsExporting(true);
    try {
      await exportAPI.export(format, result.filePath, documentId);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="card-pixel w-full max-w-md animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-chonk-slate">
          <div className="flex items-center gap-2">
            <Download size={18} className="text-accent-primary" />
            <h2 className="text-sm font-medium text-chonk-white">
              Export Chunks
            </h2>
          </div>
          <button
            className="p-1 text-chonk-gray hover:text-chonk-white"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          <p className="text-sm text-chonk-gray">
            {documentId ? (
              <>
                Export chunks from{' '}
                <span className="text-chonk-white">{documentName}</span>
              </>
            ) : (
              'Export all chunks from the project'
            )}
          </p>

          {/* Format selection */}
          <div className="space-y-2">
            {formats.map((f) => {
              const Icon = f.icon;
              return (
                <label
                  key={f.id}
                  className={`
                    flex items-center gap-3 p-3 rounded cursor-pointer transition-colors
                    ${format === f.id
                      ? 'bg-accent-primary/20 border border-accent-primary'
                      : 'bg-surface-card border border-transparent hover:border-chonk-slate'
                    }
                  `}
                >
                  <input
                    type="radio"
                    name="format"
                    value={f.id}
                    checked={format === f.id}
                    onChange={(e) => setFormat(e.target.value)}
                    className="sr-only"
                  />
                  <Icon
                    size={20}
                    className={format === f.id ? 'text-accent-primary' : 'text-chonk-gray'}
                  />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-chonk-white">
                      {f.name}
                    </div>
                    <div className="text-xs text-chonk-gray">{f.description}</div>
                  </div>
                  <span className="text-xs text-chonk-slate">{f.extension}</span>
                </label>
              );
            })}
          </div>

          {/* Export info */}
          <div className="p-3 rounded bg-surface-card">
            <div className="text-xs text-chonk-gray">
              <p className="mb-1">Export includes:</p>
              <ul className="list-disc list-inside space-y-0.5 text-chonk-light">
                <li>Chunk content and metadata</li>
                <li>Quality scores</li>
                <li>User tags and notes</li>
                <li>Hierarchy paths</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-4 py-3 border-t border-chonk-slate">
          <button className="btn-pixel" onClick={onClose} disabled={isExporting}>
            Cancel
          </button>
          <button
            className="btn-pixel-primary flex items-center gap-2"
            onClick={handleExport}
            disabled={isExporting}
          >
            <Download size={14} />
            {isExporting ? 'Exporting...' : 'Export'}
          </button>
        </div>
      </div>
    </div>
  );
}
