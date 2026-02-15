#!/usr/bin/env node
/**
 * H-10 & H-11: Auto-Formatter + Auto-Linter
 * Fires on: PostToolUse [Edit|Write|MultiEdit]
 * Formats and lints every modified file, feeds errors back as context
 */
const { readStdin, run, commandExists, feedback, normPath } = require("./hook-utils");
const path = require("path");
const fs = require("fs");

const data = readStdin();
const filePath = normPath(
  (data.tool_input && (data.tool_input.file_path || data.tool_input.filePath)) || ""
);

if (!filePath || !fs.existsSync(filePath)) process.exit(0);

const ext = path.extname(filePath).toLowerCase();

switch (ext) {
  case ".py": {
    // Format
    if (commandExists("black")) {
      run(`black --quiet "${filePath}"`);
    } else if (commandExists("ruff")) {
      run(`ruff format "${filePath}"`);
    }
    // Lint
    if (commandExists("ruff")) {
      const { stdout, exitCode } = run(`ruff check "${filePath}"`);
      if (exitCode !== 0 && stdout) {
        feedback(`LINT [${filePath}]:\n${stdout.split("\n").slice(-10).join("\n")}`);
      }
    }
    // Type check
    if (commandExists("mypy")) {
      const { stdout, exitCode } = run(`mypy "${filePath}" --no-error-summary`);
      if (exitCode !== 0 && stdout && !stdout.includes("Success")) {
        feedback(`TYPE [${filePath}]:\n${stdout.split("\n").slice(-5).join("\n")}`);
      }
    }
    break;
  }

  case ".js":
  case ".jsx": {
    if (commandExists("npx")) {
      run(`npx prettier --write "${filePath}"`);
      if (fs.existsSync("node_modules/.bin/eslint")) {
        const { stdout, exitCode } = run(`npx eslint "${filePath}"`);
        if (exitCode !== 0 && stdout) {
          feedback(`LINT [${filePath}]:\n${stdout.split("\n").slice(-10).join("\n")}`);
        }
      }
    }
    break;
  }

  case ".ts":
  case ".tsx": {
    if (commandExists("npx")) {
      run(`npx prettier --write "${filePath}"`);
      if (fs.existsSync("node_modules/.bin/eslint")) {
        const { stdout, exitCode } = run(`npx eslint "${filePath}"`);
        if (exitCode !== 0 && stdout) {
          feedback(`LINT [${filePath}]:\n${stdout.split("\n").slice(-10).join("\n")}`);
        }
      }
      if (fs.existsSync("tsconfig.json")) {
        const { stdout, exitCode } = run(`npx tsc --noEmit --skipLibCheck`);
        if (exitCode !== 0 && stdout) {
          feedback(`TYPE [${filePath}]:\n${stdout.split("\n").slice(-5).join("\n")}`);
        }
      }
    }
    break;
  }

  case ".go": {
    if (commandExists("gofmt")) run(`gofmt -w "${filePath}"`);
    break;
  }

  case ".rs": {
    if (commandExists("rustfmt")) run(`rustfmt "${filePath}"`);
    break;
  }
}
