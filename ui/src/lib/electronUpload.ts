/**
 * Helpers for uploading a file chosen via the Electron native file dialog.
 *
 * In Electron the dialog returns a file path (not a browser File), and the
 * main process reads its bytes over IPC as base64. This converts that payload
 * into a File so the same `documentAPI.upload(file)` path used by the web UI
 * works unchanged.
 */

export interface ElectronFileRead {
  name: string;
  /** Base64-encoded file contents. */
  data: string;
  mimeType: string;
}

/** Build a browser `File` from an Electron `file:read` IPC result. */
export function fileFromElectronRead(read: ElectronFileRead): File {
  const binary = atob(read.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new File([bytes], read.name, { type: read.mimeType });
}
