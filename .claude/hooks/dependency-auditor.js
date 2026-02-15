#!/usr/bin/env node
/**
 * H-09: Dependency Install Auditor
 * Fires on: PreToolUse [Bash]
 * Logs and audits new dependency installations
 */
const { readStdin, feedback, run, logSession } = require("./hook-utils");

const data = readStdin();
const cmd = (data.tool_input && data.tool_input.command) || "";

if (!cmd) process.exit(0);

// npm install <package>
const npmMatch = cmd.match(/^npm (install|i)\s+([^-]\S+)/);
if (npmMatch) {
  const pkg = npmMatch[2];
  feedback(`DEPENDENCY AUDIT: Installing npm package '${pkg}'`);
  logSession({ event: "dep-install", type: "npm", package: pkg });

  const { stdout } = run("npm audit --json 2>&1");
  try {
    const audit = JSON.parse(stdout);
    const critical = audit?.metadata?.vulnerabilities?.critical || 0;
    if (critical > 0) {
      feedback(`WARNING: Project has ${critical} critical vulnerabilities. Review with npm audit.`);
    }
  } catch { /* audit parsing failed, non-blocking */ }
}

// pip install <package>
const pipMatch = cmd.match(/^pip3? install\s+([^-]\S+)/);
if (pipMatch) {
  const pkg = pipMatch[1];
  feedback(`DEPENDENCY AUDIT: Installing pip package '${pkg}'`);
  logSession({ event: "dep-install", type: "pip", package: pkg });
}

// Always allow — warns but doesn't block
process.exit(0);
