#!/usr/bin/env node
/**
 * H-14: End-of-Turn Quality Gate  *** THE MOST CRITICAL HOOK ***
 * Fires on: Stop [*]
 * Claude CANNOT end a turn with broken tests, lint errors,
 * security issues, or merge conflicts. Exit 2 = keep working.
 */
const { run, git, feedback, getProjectDir } = require("./hook-utils");
const path = require("path");
const fs = require("fs");

const projectDir = getProjectDir();
const logPath = path.join(projectDir, ".claude", "session-log.jsonl");

// Skip if no files modified this session
if (!fs.existsSync(logPath)) process.exit(0);

let modified;
try {
  modified = [
    ...new Set(
      fs
        .readFileSync(logPath, "utf-8")
        .trim()
        .split("\n")
        .map((line) => {
          try {
            return JSON.parse(line).file;
          } catch {
            return null;
          }
        })
        .filter(Boolean),
    ),
  ];
} catch {
  process.exit(0);
}

if (modified.length === 0) process.exit(0);

let failures = 0;
const report = [];

feedback("=== QUALITY GATE ===");

// 1. Run test suite
feedback("[1/5] Tests...");
let testResult;
if (fs.existsSync(path.join(projectDir, "package.json"))) {
  testResult = run("npm test", projectDir);
} else if (
  fs.existsSync(path.join(projectDir, "pytest.ini")) ||
  fs.existsSync(path.join(projectDir, "pyproject.toml")) ||
  fs.existsSync(path.join(projectDir, "setup.py")) ||
  fs.existsSync(path.join(projectDir, "tests"))
) {
  testResult = run("python -m pytest --tb=short", projectDir, {
    timeout: 600000,
  });
} else {
  testResult = { stdout: "", exitCode: 0 };
}

if (testResult.exitCode !== 0) {
  failures++;
  const tail = testResult.stdout.split("\n").slice(-20).join("\n");
  report.push(`FAIL: Tests failed:\n${tail}`);
} else {
  feedback("  Tests passed.");
}

// 2. Lint changed files
feedback("[2/5] Linting...");
let lintFail = false;
for (const file of modified) {
  if (!fs.existsSync(file)) continue;
  const ext = path.extname(file).toLowerCase();

  if (ext === ".py") {
    const { exitCode, stdout } = run(`ruff check "${file}"`, projectDir);
    if (exitCode !== 0 && stdout) {
      lintFail = true;
      report.push(`LINT: ${file}\n${stdout}`);
    }
  } else if ([".js", ".jsx", ".ts", ".tsx"].includes(ext)) {
    if (
      fs.existsSync(path.join(projectDir, "node_modules", ".bin", "eslint"))
    ) {
      const { exitCode, stdout } = run(`npx eslint "${file}"`, projectDir);
      if (exitCode !== 0 && stdout) {
        lintFail = true;
        report.push(`LINT: ${file}\n${stdout}`);
      }
    }
  }
}
if (lintFail) failures++;
else feedback("  Lint passed.");

// 3. Security scan
feedback("[3/5] Security scan...");
let secFail = false;
for (const file of modified) {
  if (!fs.existsSync(file)) continue;
  if (path.extname(file) === ".py") {
    const { exitCode, stdout } = run(`bandit -q "${file}"`, projectDir);
    if (exitCode !== 0 && stdout) {
      secFail = true;
      report.push(`SECURITY: ${file}\n${stdout}`);
    }
  }
}
if (secFail) failures++;
else feedback("  Security scan passed.");

// 4. Secrets scan on git diff
feedback("[4/5] Secrets scan...");
const diff = git("diff HEAD");
if (diff) {
  const secretPattern =
    /(?:api[_-]?key|secret|password|token|credential)\s*[:=]/i;
  const secretLines = diff
    .split("\n")
    .filter(
      (line) =>
        line.startsWith("+") &&
        !line.startsWith("+++") &&
        secretPattern.test(line) &&
        !/#.*TODO|#.*FIXME|\.example/.test(line),
    );
  if (secretLines.length > 0) {
    failures++;
    report.push(
      `SECRETS DETECTED in diff:\n${secretLines.slice(0, 5).join("\n")}`,
    );
  } else {
    feedback("  No secrets detected.");
  }
} else {
  feedback("  No diff to scan.");
}

// 5. Conflict check
feedback("[5/5] Conflict check...");
const branch = git("branch --show-current");
if (branch && branch !== "main" && branch !== "master") {
  git("fetch origin main --quiet");
  const mergeBase = git("merge-base HEAD origin/main");
  if (mergeBase) {
    const mergeTree = git(`merge-tree ${mergeBase} origin/main HEAD`);
    const conflicts = (mergeTree.match(/^<<<<<<</gm) || []).length;
    if (conflicts > 0) {
      failures++;
      report.push(
        `CONFLICT: ${conflicts} merge conflict(s) with origin/main. Rebase needed.`,
      );
    } else {
      feedback("  No conflicts with main.");
    }
  }
} else {
  feedback("  Skipped (on main or no branch).");
}

// Verdict
feedback("");
if (failures > 0) {
  feedback(`=== QUALITY GATE FAILED (${failures} issue(s)) ===`);
  report.forEach((r) => feedback(r));
  feedback("");
  feedback("Fix all issues before completing this turn.");
  process.exit(2);
} else {
  feedback("=== QUALITY GATE PASSED ===");
  process.exit(0);
}
