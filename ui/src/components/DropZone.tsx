import { useState, useCallback, useRef } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { useStore } from '../store/useStore';
import { documentAPI, projectAPI, settingsAPI } from '../api/chonk';

interface DropZoneProps {
  children: React.ReactNode;
}

export function DropZone({ children }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const dragCounter = useRef(0);

  const { setProject, selectDocument, setError } = useStore();

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      dragCounter.current = 0;

      const files = Array.from(e.dataTransfer.files);
      const validExtensions = ['.pdf', '.docx', '.md', '.txt', '.markdown'];

      const validFiles = files.filter((file) => {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        return validExtensions.includes(ext);
      });

      if (validFiles.length === 0) {
        setError('No valid files. Supported: PDF, DOCX, MD, TXT');
        return;
      }

      setIsUploading(true);

      try {
        // Get extraction tier setting
        const settings = await settingsAPI.get().catch(() => ({}));
        const extractionTier = (settings as Record<string, unknown>).extraction_tier as string | undefined;

        for (let i = 0; i < validFiles.length; i++) {
          const file = validFiles[i];
          setUploadProgress(`Uploading ${file.name} (${i + 1}/${validFiles.length})...`);

          const result = await documentAPI.upload(file, extractionTier);

          // Log extraction info if available
          if (result.tier_used) {
            console.log(`[CHONK] Extracted ${file.name} using tier: ${result.tier_used}`);
          }
          if (result.warnings && result.warnings.length > 0) {
            console.warn(`[CHONK] Extraction warnings for ${file.name}:`, result.warnings);
          }

          // Select the first uploaded document
          if (i === 0) {
            selectDocument(result.document_id);
          }
        }

        // Refresh project
        const updatedProject = await projectAPI.get();
        setProject(updatedProject);
        setUploadProgress(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setIsUploading(false);
        setUploadProgress(null);
      }
    },
    [setProject, selectDocument, setError]
  );

  return (
    <div
      className="relative h-full"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {children}

      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-50 bg-surface-bg/90 flex items-center justify-center">
          <div className="card-pixel p-8 text-center animate-pulse-pixel">
            <Upload size={48} className="mx-auto mb-4 text-accent-primary" />
            <p className="text-lg text-chonk-white mb-2">Drop files to upload</p>
            <p className="text-sm text-chonk-gray">PDF, DOCX, MD, TXT</p>
          </div>
        </div>
      )}

      {/* Upload progress overlay */}
      {isUploading && (
        <div className="absolute inset-0 z-50 bg-surface-bg/90 flex items-center justify-center">
          <div className="card-pixel p-8 text-center">
            <div className="w-12 h-12 mx-auto mb-4 border-4 border-accent-primary border-t-transparent rounded-full animate-spin" />
            <p className="text-chonk-white">{uploadProgress || 'Processing...'}</p>
          </div>
        </div>
      )}
    </div>
  );
}
