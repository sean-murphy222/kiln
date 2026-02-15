# /conflicts — Full Conflict Analysis

Run a comprehensive conflict analysis across all active work:

```bash
echo "=== Fetching latest main ==="
git fetch origin main

echo ""
echo "=== Active Feature Branches ==="
for branch in $(git branch --list 'feature/*' --sort=-committerdate); do
    echo "--- $branch ---"
    git diff --name-only origin/main...$branch 2>/dev/null
    echo ""
done

echo "=== Active Worktrees ==="
git worktree list

echo ""
echo "=== Merge Check (current branch vs main) ==="
MERGE_BASE=$(git merge-base HEAD origin/main 2>/dev/null)
if [ -n "$MERGE_BASE" ]; then
    RESULT=$(git merge-tree $MERGE_BASE origin/main HEAD 2>/dev/null)
    if echo "$RESULT" | grep -q "^<<<<<<<"; then
        echo "CONFLICTS DETECTED with origin/main!"
        echo "$RESULT" | grep "^<<<<<<< " | sed 's/^<<<<<<< /  Conflict in: /'
    else
        echo "Clean merge with origin/main."
    fi
fi
```

After running the analysis:
1. Identify any file overlaps between active branches
2. Update the Conflict Map section in BACKLOG.md
3. Flag any tasks that should be blocked due to overlaps
4. Report findings in a clear summary
