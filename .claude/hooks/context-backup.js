#!/usr/bin/env node
/**
 * H-18: Context Backup (PreCompact)
 * Saves critical state before context compaction for H-02 recovery
 */
const { git, feedback, getProjectDir } = require("./hook-utils");
const path = require("path");
const fs = require("fs");

const projectDir = getProjectDir();
const backupDir = path.join(projectDir, ".claude", "backups");
const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const backupFile = path.join(backupDir, `backup-${ts}.md`);

// Ensure backup directory exists
if (!fs.existsSync(backupDir)) fs.mkdirSync(backupDir, { recursive: true });

const branch = git("branch --show-current") || "unknown";
const taskMatch = branch.match(/T-(\d+)/);
const taskId = taskMatch ? `T-${taskMatch[1]}` : "unknown";

const recentCommits = git("log --oneline -5") || "none";
const status = git("status --short") || "clean";

const logPath = path.join(projectDir, ".claude", "session-log.jsonl");
let recentLog = "no log";
try {
  if (fs.existsSync(logPath)) {
    recentLog = fs
      .readFileSync(logPath, "utf-8")
      .trim()
      .split("\n")
      .slice(-10)
      .join("\n");
  }
} catch { /* non-critical */ }

// Read in-progress section from backlog
let backlogContext = "check BACKLOG.md";
try {
  const backlog = fs.readFileSync(
    path.join(projectDir, "BACKLOG.md"),
    "utf-8"
  );
  const inProgress = backlog.match(/## In Progress[\s\S]*?(?=\n## )/);
  if (inProgress) backlogContext = inProgress[0].trim();
} catch { /* non-critical */ }

const content = `# Context Backup — ${ts}

## Active Task
- Task: ${taskId}
- Branch: ${branch}

## Recent Commits on This Branch
${recentCommits}

## Modified Files (uncommitted)
${status}

## Last 10 Session Log Entries
${recentLog}

## Backlog Status
${backlogContext}
`;

fs.writeFileSync(backupFile, content);
feedback(`Context backed up to: ${backupFile}`);

// Clean old backups (keep last 10)
try {
  const backups = fs
    .readdirSync(backupDir)
    .filter((f) => f.endsWith(".md"))
    .sort()
    .reverse();
  for (const old of backups.slice(10)) {
    fs.unlinkSync(path.join(backupDir, old));
  }
} catch { /* non-critical */ }
