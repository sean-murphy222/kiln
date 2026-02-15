# /ship — Validate and Push

This is the final gate before code leaves the local machine.
Run ALL checks, then push only if everything passes.

```bash
echo "=== PRE-SHIP VALIDATION ==="
BRANCH=$(git branch --show-current)
FAILURES=0

# 1. Not on main
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
    echo "FAIL: Cannot ship from $BRANCH"
    exit 1
fi

# 2. Clean working tree
if [ -n "$(git status --porcelain)" ]; then
    echo "FAIL: Uncommitted changes"
    FAILURES=$((FAILURES + 1))
fi

# 3. Tests pass
echo "[1/6] Running tests..."
npm test 2>/dev/null || python -m pytest 2>/dev/null
[ $? -ne 0 ] && FAILURES=$((FAILURES + 1))

# 4. Lint clean
echo "[2/6] Linting..."
ruff check . 2>/dev/null || npx eslint . 2>/dev/null
[ $? -ne 0 ] && FAILURES=$((FAILURES + 1))

# 5. No secrets in diff
echo "[3/6] Secrets scan..."
SECRETS=$(git diff origin/main | grep -iE '(api.key|secret|password|token|credential)\s*[:=]' | grep -v '\.example\|#.*TODO')
[ -n "$SECRETS" ] && echo "FAIL: Secrets detected" && FAILURES=$((FAILURES + 1))

# 6. Conventional commits
echo "[4/6] Commit format..."
BAD_COMMITS=$(git log origin/main..HEAD --pretty=%s | grep -cvE '^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)')
[ "$BAD_COMMITS" -gt 0 ] && echo "FAIL: Non-conventional commits" && FAILURES=$((FAILURES + 1))

# 7. No conflicts with main
echo "[5/6] Conflict check..."
git fetch origin main
MERGE_BASE=$(git merge-base HEAD origin/main)
CONFLICTS=$(git merge-tree $MERGE_BASE origin/main HEAD 2>/dev/null | grep -c "^<<<<<<<")
[ "$CONFLICTS" -gt 0 ] && echo "FAIL: Merge conflicts" && FAILURES=$((FAILURES + 1))

# 8. Coverage check
echo "[6/6] Coverage..."
# Add your coverage check command here

if [ $FAILURES -gt 0 ]; then
    echo ""
    echo "=== SHIP BLOCKED: $FAILURES failure(s) ==="
    echo "Fix all issues before shipping."
else
    echo ""
    echo "=== ALL CHECKS PASSED ==="
    git push origin $BRANCH
    echo "Pushed $BRANCH to origin."
fi
```

After a successful push, suggest the developer create a PR for human review.
