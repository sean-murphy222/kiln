# /status — Sprint Progress Dashboard

Display a comprehensive sprint status:

```bash
echo "=== SPRINT STATUS ==="
echo "Date: $(date '+%Y-%m-%d %H:%M')"
echo "Branch: $(git branch --show-current)"
echo ""

echo "--- Git State ---"
git log --oneline -5
echo ""
git status --short
echo ""

echo "--- Active Branches ---"
git branch --list 'feature/*' --sort=-committerdate -v
echo ""

echo "--- Worktrees ---"
git worktree list 2>/dev/null
echo ""

echo "--- Session Log (last 5 changes) ---"
tail -5 .claude/session-log.jsonl 2>/dev/null || echo "No session log yet"
```

Then read BACKLOG.md and summarize:
- Tasks Done / In Progress / Ready / Blocked
- Current sprint velocity (story points completed)
- Estimated remaining work
- Any blockers or risks to flag
