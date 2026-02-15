# /sprint-start — Initialize Sprint Session

Read BACKLOG.md to understand the current sprint state.

Run conflict analysis:
```bash
git fetch origin main
for branch in $(git branch --list 'feature/*' --sort=-committerdate); do
    echo "=== $branch ==="
    git diff --name-only origin/main...$branch 2>/dev/null
done
```

Update the conflict map in BACKLOG.md.

Print sprint status:
- Tasks completed / in progress / ready
- Active branches and what they're working on
- Any blockers or conflicts detected

Then enter plan mode and select the first task to work on.
Use /pick-task to begin implementation.
