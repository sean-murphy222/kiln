#!/usr/bin/env node
/**
 * H-06 & H-07: Branch Protector + Force Push Guard
 * Fires on: PreToolUse [Bash]
 * Blocks commits/pushes to main, blocks all force pushes
 */
const { readStdin, git, block } = require("./hook-utils");

const data = readStdin();
const cmd = (data.tool_input && data.tool_input.command) || "";

if (!cmd) process.exit(0);

// H-07: Block force pushes everywhere
if (/git push.*(--force|-f\b)/i.test(cmd)) {
  block(
    "BLOCKED: Force push is never allowed. It destroys shared history.\n" +
      "If you need to update a remote branch, use: git push --force-with-lease"
  );
}

// H-06: Block commits and pushes on main/master
if (/^git (commit|push|merge|cherry-pick)/i.test(cmd)) {
  const branch = git("branch --show-current");
  if (branch === "main" || branch === "master") {
    block(
      `BLOCKED: Cannot ${cmd.split(" ").slice(0, 2).join(" ")} on ${branch} branch.\n` +
        "All work must happen on feature branches or in worktrees.\n" +
        "Create a branch: git checkout -b feature/T-{id}-{description}"
    );
  }
}

// Block merging INTO main from main
if (/git merge.*(main|master)/i.test(cmd)) {
  const branch = git("branch --show-current");
  if (branch === "main" || branch === "master") {
    block("BLOCKED: Merging into main must be done by a human after review.");
  }
}
