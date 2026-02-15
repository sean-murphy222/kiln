# /pick-task — Select and Begin Next Task

1. Read BACKLOG.md and find the highest-priority unblocked task in Ready state.

2. Run conflict analysis against active branches:
```bash
git fetch origin main
# Check if task's expected files overlap with active work
for branch in $(git branch --list 'feature/*'); do
    git diff --name-only origin/main...$branch 2>/dev/null
done
```

3. If the top task has file conflicts with active work, skip it and pick the next one.

4. Create a feature branch:
```bash
git checkout -b feature/T-{id}-{short-description} origin/main
```

5. Update BACKLOG.md: move the task to "In Progress", note the branch name.

6. Enter plan mode. Create a detailed implementation plan covering:
   - What files to create/modify
   - Test strategy (what to test first in the RED phase)
   - Expected acceptance criteria validation
   - Risks and dependencies

7. After the plan is solid, begin TDD implementation.
