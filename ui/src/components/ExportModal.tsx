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
      <div className="card w-full max-w-md animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-kiln-600">
          <div className="flex items-center gap-2">
            <Download size={18} className="text-ember" />
            <h2 className="text-sm font-medium text-kiln-100">
              Export Chunks
            </h2>
          </div>
          <button
            className="p-1 text-kiln-500 hover:text-kiln-100"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          <p className="text-sm text-kiln-500">
            {documentId ? (
              <>
                Export chunks from{' '}
                <span className="text-kiln-100">{documentName}</span>
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
                      ? 'bg-ember/20 border border-ember'
                      : 'bg-kiln-700 border border-transparent hover:border-kiln-600'
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
                    className={format === f.id ? 'text-ember' : 'text-kiln-500'}
                  />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-kiln-100">
                      {f.name}
                    </div>
                    <div className="text-xs text-kiln-500">{f.description}</div>
                  </div>
                  <span className="text-xs text-kiln-600">{f.extension}</span>
                </label>
              );
            })}
          </div>

          {/* Export info */}
          <div className="p-3 rounded bg-kiln-700">
            <div className="text-xs text-kiln-500">
              <p className="mb-1">Export includes:</p>
              <ul className="list-disc list-inside space-y-0.5 text-kiln-300">
                <li>Chunk content and metadata</li>
                <li>Quality scores</li>
                <li>User tags and notes</li>
                <li>Hierarchy paths</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-4 py-3 border-t border-kiln-600">
          <button className="btn-secondary" onClick={onClose} disabled={isExporting}>
            Cancel
          </button>
          <button
            className="btn-primary flex items-center gap-2"
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
