# /done — Complete Current Task

Pre-completion checklist:

1. Verify all acceptance criteria from the task description are met.

2. Ensure all changes are committed:
```bash
git status
git add -A
git commit -m "feat(scope): complete T-{id} - {description}"
```

3. Run the /review cycle (spawns security-reviewer and qa-engineer).

4. Update BACKLOG.md:
   - Move task to "Done" state
   - Record completion time
   - Note any tech debt discovered

5. Update CHANGELOG.md with what was added/changed/fixed.

6. The Definition of Done hook (H-16) will automatically validate:
   - Tests exist for all changed source files
   - All tests pass
   - Commit messages follow Conventional Commits
   - No merge conflicts with origin/main
   - No uncommitted changes

If H-16 fails, fix the issues before proceeding.
