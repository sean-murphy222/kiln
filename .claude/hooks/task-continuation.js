#!/usr/bin/env node
/**
 * H-15: Task Continuation Nudge
 * Fires on: Stop [*] (after quality gate passes)
 * Keeps the sprint loop going when tasks remain
 */
const { readBacklog, feedback } = require("./hook-utils");

// Only nudge in sprint mode
if (process.env.SPRINT_MODE !== "1") process.exit(0);

const backlog = readBacklog();
if (!backlog) process.exit(0);

const ready = (backlog.match(/^- \[ \]/gm) || []).length;

if (ready > 0) {
  feedback("");
  feedback(`Quality gate passed. ${ready} task(s) remaining in backlog.`);
  feedback("Run /pick-task to start the next highest-priority unblocked task.");
  process.exit(2);
}
