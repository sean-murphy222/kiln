#!/usr/bin/env node
/**
 * H-03: Sprint State Anchor
 * Fires on: UserPromptSubmit (every prompt)
 * Micro-injects current task to prevent drift during long sessions
 */
const { git, feedback } = require("./hook-utils");

const branch = git("branch --show-current");

if (branch && branch.startsWith("feature/")) {
  const match = branch.match(/T-(\d+)/);
  const taskId = match ? `T-${match[1]}` : "";
  if (taskId) {
    feedback(`CURRENT TASK: ${taskId} | BRANCH: ${branch}`);
  }
}
