#!/usr/bin/env node
/**
 * H-01 & H-02: Sprint Context Loader / Resume Context Recovery
 * Fires on: SessionStart (startup | resume | compact)
 */
const { git, feedback, readBacklog, getProjectDir } = require("./hook-utils");
const fs = require("fs");
const path = require("path");

const mode = process.argv[2] || "startup";
const projectDir = getProjectDir();

feedback("=== AUTONOMOUS SCRUM SESSION ===");
feedback(`Mode: ${mode} | Date: ${new Date().toLocaleString()}`);
feedback("");

// Current git state
const branch = git("branch --show-current");
feedback(`Branch: ${branch || "not in git repo"}`);

const latestMain = git("log --oneline -1 origin/main");
if (latestMain) feedback(`Latest main: ${latestMain}`);

const featureBranches = git("branch --list feature/* --sort=-committerdate");
if (featureBranches) {
  feedback("Active feature branches:");
  featureBranches
    .split("\n")
    .slice(0, 5)
    .forEach((b) => feedback(`  ${b.trim()}`));
  feedback("");
}

// Backlog status
const backlog = readBacklog();
if (backlog) {
  const ready = (backlog.match(/^- \[ \]/gm) || []).length;
  const inProgress = (backlog.match(/In Progress/g) || []).length;
  const done = (backlog.match(/^- \[x\]/gm) || []).length;
  feedback(`Sprint: ${done} done | ${inProgress} in progress | ${ready} ready`);
}

// Resume-specific: recover from context backup
if (mode === "resume" || mode === "compact") {
  feedback("");
  feedback("--- Context Recovery ---");
  const backupDir = path.join(projectDir, ".claude", "backups");
  try {
    const backups = fs
      .readdirSync(backupDir)
      .filter((f) => f.endsWith(".md"))
      .sort()
      .reverse();
    if (backups.length > 0) {
      const latest = path.join(backupDir, backups[0]);
      feedback(`Recovering from: ${latest}`);
      feedback(fs.readFileSync(latest, "utf-8"));
    } else {
      feedback("No backup found. Check BACKLOG.md for current task.");
    }
  } catch {
    feedback("No backup directory. Check BACKLOG.md for current task.");
  }
}

feedback("");
feedback(
  "ENFORCEMENT ACTIVE: 19 hooks enforcing quality, security, and conventions."
);
feedback("Read BACKLOG.md for current sprint tasks.");
