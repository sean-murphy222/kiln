---
name: conflict-analysis
description: >
  Git conflict prevention and resolution. Apply before starting any new
  task or when preparing to merge/rebase. Analyzes active branches and
  worktrees to predict and prevent merge conflicts.
---

# Conflict Analysis Skill

## Pre-Task Conflict Check

Run this BEFORE starting any new task:

```bash
# 1. Fetch latest main
git fetch origin main

# 2. Map files touched by each active feature branch
echo "=== Active Branch File Map ==="
for branch in $(git branch -r --list 'origin/feature/*' 2>/dev/null); do
    echo "--- $branch ---"
    git diff --name-only origin/main...$branch 2>/dev/null
done

# 3. Map files touched by each worktree
echo "=== Worktree File Map ==="
for wt in $(git worktree list --porcelain | grep "^worktree" | awk '{print $2}'); do
    BRANCH=$(git -C "$wt" branch --show-current 2>/dev/null)
    echo "--- $wt ($BRANCH) ---"
    git -C "$wt" diff --name-only origin/main 2>/dev/null
done

# 4. Compare against your task's expected files
# If ANY overlap → pick a different task or wait
```

## Pre-Push Conflict Check

Run this BEFORE pushing:

```bash
# Fetch and test merge
git fetch origin main
MERGE_BASE=$(git merge-base HEAD origin/main)
MERGE_RESULT=$(git merge-tree $MERGE_BASE origin/main HEAD)

# Check for conflict markers
if echo "$MERGE_RESULT" | grep -q "^<<<<<<<"; then
    echo "CONFLICTS DETECTED — rebase needed"
    echo "$MERGE_RESULT" | grep "^<<<<<<<" -A 5
else
    echo "Clean merge — safe to push"
fi
```

## Conflict Resolution Workflow

If conflicts are detected:
1. `git rebase origin/main` (preferred over merge)
2. Fix each conflict file
3. `git add {resolved files}`
4. `git rebase --continue`
5. Run full test suite
6. Re-run conflict check to confirm clean

## Conflict Map Format (for BACKLOG.md)

```markdown
## Conflict Map
| Branch | Files Modified | Overlaps With |
|--------|---------------|---------------|
| feature/T-001-parser | src/parser.py, tests/test_parser.py | None |
| feature/T-002-chunker | src/chunker.py, src/utils.py | T-003 (utils.py) |
| feature/T-003-embedder | src/embedder.py, src/utils.py | T-002 (utils.py) |
```

Tasks that overlap must be sequenced, not parallelized.
