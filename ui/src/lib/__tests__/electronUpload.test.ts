import { describe, expect, it } from "vitest";

import { fileFromElectronRead } from "../electronUpload";

describe("fileFromElectronRead", () => {
  it("builds a File with the right name, type, and bytes from base64", async () => {
    const text = "hello world";
    const data = btoa(text);
    const file = fileFromElectronRead({
      name: "doc.pdf",
      data,
      mimeType: "application/pdf",
    });

    expect(file).toBeInstanceOf(File);
    expect(file.name).toBe("doc.pdf");
    expect(file.type).toBe("application/pdf");
    expect(await file.text()).toBe(text);
  });

  it("round-trips binary (non-text) bytes intact", async () => {
    const bytes = new Uint8Array([0, 255, 16, 128, 1]);
    const data = btoa(String.fromCharCode(...bytes));
    const file = fileFromElectronRead({
      name: "b.bin",
      data,
      mimeType: "application/octet-stream",
    });

    const out = new Uint8Array(await file.arrayBuffer());
    expect(Array.from(out)).toEqual(Array.from(bytes));
  });
});
