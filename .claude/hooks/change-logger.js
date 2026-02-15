#!/usr/bin/env node
/**
 * H-12: Change Audit Logger (async)
 * Fires on: PostToolUse [Edit|Write|MultiEdit]
 * Logs every file modification to session-log.jsonl
 */
const { readStdin, git, logSession, normPath } = require("./hook-utils");
const fs = require("fs");

const data = readStdin();
const filePath = normPath(
  (data.tool_input && (data.tool_input.file_path || data.tool_input.filePath)) || ""
);
const tool = data.tool_name || "unknown";

if (!filePath) process.exit(0);

let lines = 0;
try {
  if (fs.existsSync(filePath)) {
    lines = fs.readFileSync(filePath, "utf-8").split("\n").length;
  }
} catch { /* non-critical */ }

logSession({
  tool,
  file: filePath,
  lines,
  branch: git("branch --show-current"),
});
