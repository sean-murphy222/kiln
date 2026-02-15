#!/usr/bin/env node
/**
 * H-08: System Destructive Command Guard
 * Fires on: PreToolUse [Bash]
 * Blocks rm -rf /, chmod 777, mkfs, dd, curl|bash, etc.
 */
const { readStdin, block } = require("./hook-utils");

const data = readStdin();
const cmd = (data.tool_input && data.tool_input.command) || "";

if (!cmd) process.exit(0);

const dangerous = [
  { pattern: /rm\s+(-\w*r\w*f|-\w*f\w*r)\s+[/~$]/, msg: "Destructive rm command targeting critical path" },
  { pattern: /rm\s+(-\w*r\w*f|-\w*f\w*r)\s+\.git/, msg: "Destructive rm targeting .git directory" },
  { pattern: /chmod\s+(777|a\+w)/, msg: "World-writable permissions (777/a+w) are never appropriate" },
  { pattern: /^mkfs/i, msg: "Filesystem-destructive command" },
  { pattern: /^dd\s+if=/i, msg: "Disk-destructive command (dd)" },
  { pattern: /(curl|wget).*\|\s*(bash|sh|zsh)/, msg: "Piping remote content to shell. Download first, review, then execute." },
  { pattern: /:\(\)\s*\{.*\|.*&\s*\}\s*;/, msg: "Fork bomb detected" },
];

for (const { pattern, msg } of dangerous) {
  if (pattern.test(cmd)) {
    block(`BLOCKED: ${msg}\nCommand: ${cmd}`);
  }
}
