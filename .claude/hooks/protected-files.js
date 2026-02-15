#!/usr/bin/env node
/**
 * H-05: Protected File Write Guard
 * Fires on: PreToolUse [Write|Edit|MultiEdit]
 * Blocks writes to .env, secrets, .git/, hook config
 */
const { readStdin, normPath, block } = require("./hook-utils");

const data = readStdin();
const filePath = normPath(
  (data.tool_input && (data.tool_input.file_path || data.tool_input.filePath)) || ""
);

if (!filePath) process.exit(0);

// Sensitive files
if (/\.(env$|env\.)|secrets|credentials|\.pem$|\.key$|\.p12$|\.pfx$/i.test(filePath)) {
  block(
    `BLOCKED: Cannot write to sensitive file: ${filePath}\n` +
      "Sensitive files (.env, secrets, credentials, key files) must be edited manually."
  );
}

// Git internals
if (/^\.git\//i.test(filePath) || /[/\\]\.git[/\\]/i.test(filePath)) {
  block("BLOCKED: Cannot write to .git/ directory.");
}

// Hook configuration (prevent self-modification)
if (/\.claude[/\\]settings\.json$/i.test(filePath)) {
  block(
    "BLOCKED: Cannot modify hook configuration (.claude/settings.json).\n" +
      "Hook config changes must be made manually by the project owner."
  );
}
