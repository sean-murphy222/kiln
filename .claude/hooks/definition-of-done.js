#!/usr/bin/env node
/**
 * H-16: Definition of Done Gate
 * Fires on: TaskCompleted [*]
 * A task cannot be marked complete unless all criteria are met
 */
const { run, git, feedback, getProjectDir } = require("./hook-utils");
const path = require("path");
const fs = require("fs");

const projectDir = getProjectDir();
let failures = 0;
const report = [];

feedback("=== DEFINITION OF DONE CHECK ===");

// 1. Tests exist for changed source files
feedback("[1/5] Test coverage check...");
const changedSrc = git("diff --name-only origin/main")
  .split("\n")
  .filter((f) => /\.(py|js|ts|jsx|tsx|go|rs)$/.test(f) && !/test/i.test(f));

for (const src of changedSrc) {
  const base = path.basename(src).replace(/\.[^.]+$/, "");
  const patterns = [
    `test_${base}`,
    `${base}.test.`,
    `${base}_test.`,
    `${base}.spec.`,
  ];
  const allFiles = git("ls-files").split("\n");
  const hasTest = allFiles.some((f) =>
    patterns.some((p) => path.basename(f).startsWith(p))
  );
  if (!hasTest) {
    failures++;
    report.push(`MISSING TEST: No test file found for ${src}`);
  }
}
if (failures === 0) feedback("  Tests exist for all source changes.");

// 2. All tests pass
feedback("[2/5] Test execution...");
let testResult;
if (fs.existsSync(path.join(projectDir, "package.json"))) {
  testResult = run("npm test", projectDir);
} else if (fs.existsSync(path.join(projectDir, "tests"))) {
  testResult = run("python -m pytest --tb=no -q", projectDir);
} else {
  testResult = { exitCode: 0 };
}
if (testResult.exitCode !== 0) {
  failures++;
  report.push("FAILED TESTS: Test suite failed");
}
feedback("  Tests executed.");

// 3. Conventional commit format
feedback("[3/5] Commit format...");
const lastCommit = git("log -1 --pretty=%s");
if (
  lastCommit &&
  !/^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?!?:/.test(
    lastCommit
  )
) {
  failures++;
  report.push(
    `COMMIT FORMAT: Last commit doesn't follow Conventional Commits: '${lastCommit}'`
  );
}

// 4. No merge conflicts
feedback("[4/5] Conflict check...");
git("fetch origin main --quiet");
const mergeBase = git("merge-base HEAD origin/main");
if (mergeBase) {
  const mergeTree = git(`merge-tree ${mergeBase} origin/main HEAD`);
  const conflicts = (mergeTree.match(/^<<<<<<</gm) || []).length;
  if (conflicts > 0) {
    failures++;
    report.push(`CONFLICT: ${conflicts} conflict(s) with origin/main`);
  }
}

// 5. Clean working tree
feedback("[5/5] Clean working tree...");
const dirty = git("status --porcelain");
if (dirty) {
  failures++;
  report.push(
    `UNCOMMITTED: Working tree has uncommitted changes:\n${dirty.split("\n").slice(0, 10).join("\n")}`
  );
}

// Verdict
feedback("");
if (failures > 0) {
  feedback(`=== DEFINITION OF DONE: FAILED (${failures} issue(s)) ===`);
  report.forEach((r) => feedback(r));
  feedback("");
  feedback("Task cannot be marked complete until all criteria are met.");
  process.exit(2);
} else {
  feedback("=== DEFINITION OF DONE: PASSED ===");
}
