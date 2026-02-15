#!/usr/bin/env node
/**
 * H-04: Secrets File Firewall
 * Fires on: PreToolUse [Bash]
 * Blocks commands that modify .env, secrets, credentials, key files
 * Allows read-only operations (grep, cat .env.example, git diff)
 */
const { readStdin, block } = require("./hook-utils");

const data = readStdin();
const cmd = (data.tool_input && data.tool_input.command) || "";

if (!cmd) process.exit(0);

// Pattern: command touches sensitive files
const sensitivePattern =
  /\.(env[^.]|env$)|secrets|credentials|api[_-]?key|\.pem|\.key|\.p12|\.pfx/i;

if (sensitivePattern.test(cmd)) {
  // Allow read-only operations
  const readOnly =
    /^(grep|cat.*\.example|echo|test\s|git diff|git log|git show|head|tail|wc|ls|find.*-name|\[.*-[fedr])/i;
  if (readOnly.test(cmd)) {
    process.exit(0);
  }

  block(
    "BLOCKED: Command touches sensitive files (.env, secrets, credentials, keys).\n" +
      "Read-only operations (grep, cat .env.example) are allowed.\n" +
      "If you need to modify secrets, do it manually outside of Claude Code."
  );
}
