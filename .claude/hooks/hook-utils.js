/**
 * hook-utils.js — Shared utilities for all Claude Code hooks
 * Cross-platform (Windows native, Git Bash, macOS, Linux)
 */
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

/**
 * Read JSON from stdin (how Claude Code passes context to hooks)
 */
function readStdin() {
  try {
    const input = fs.readFileSync(0, "utf-8");
    return JSON.parse(input);
  } catch {
    return {};
  }
}

/**
 * Get the project directory (works across platforms)
 */
function getProjectDir() {
  return (
    process.env.CLAUDE_PROJECT_DIR ||
    process.env.INIT_CWD ||
    process.cwd()
  );
}

/**
 * Normalize a file path for cross-platform comparison
 */
function normPath(p) {
  return p ? p.replace(/\\/g, "/") : "";
}

/**
 * Run a git command and return stdout, or empty string on failure
 */
function git(args, cwd) {
  try {
    return execSync(`git ${args}`, {
      cwd: cwd || getProjectDir(),
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
      timeout: 30000,
    }).trim();
  } catch {
    return "";
  }
}

/**
 * Run any command and return { stdout, exitCode }
 */
function run(cmd, cwd) {
  try {
    const stdout = execSync(cmd, {
      cwd: cwd || getProjectDir(),
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
      timeout: 60000,
    });
    return { stdout: stdout.trim(), exitCode: 0 };
  } catch (e) {
    return { stdout: (e.stdout || "").trim(), exitCode: e.status || 1 };
  }
}

/**
 * Check if a command exists on the system
 */
function commandExists(cmd) {
  try {
    const check =
      process.platform === "win32" ? `where ${cmd}` : `which ${cmd}`;
    execSync(check, { stdio: ["pipe", "pipe", "pipe"] });
    return true;
  } catch {
    return false;
  }
}

/**
 * Write to stderr (how hooks send feedback to Claude)
 */
function feedback(msg) {
  process.stderr.write(msg + "\n");
}

/**
 * Block the operation (exit code 2 = hook rejection in Claude Code)
 */
function block(msg) {
  feedback(msg);
  process.exit(2);
}

/**
 * Get the BACKLOG.md content
 */
function readBacklog() {
  const backlogPath = path.join(getProjectDir(), "BACKLOG.md");
  try {
    return fs.readFileSync(backlogPath, "utf-8");
  } catch {
    return "";
  }
}

/**
 * Append to session log (JSONL format)
 */
function logSession(entry) {
  const logPath = path.join(getProjectDir(), ".claude", "session-log.jsonl");
  try {
    const dir = path.dirname(logPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(
      logPath,
      JSON.stringify({ ts: new Date().toISOString(), ...entry }) + "\n"
    );
  } catch {
    // Non-critical, don't fail the hook
  }
}

module.exports = {
  readStdin,
  getProjectDir,
  normPath,
  git,
  run,
  commandExists,
  feedback,
  block,
  readBacklog,
  logSession,
};
