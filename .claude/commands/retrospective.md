# /retrospective — Sprint Retrospective & Compounding Loop

This is the compounding loop — the single most important habit for improving
Claude Code results over time.

## Step 1: Analyze Session Log

```bash
echo "=== SESSION RETROSPECTIVE ==="
echo ""

echo "--- Files Modified This Session ---"
cat .claude/session-log.jsonl 2>/dev/null | jq -r '.file' | sort | uniq -c | sort -rn | head -20

echo ""
echo "--- Timeline ---"
cat .claude/session-log.jsonl 2>/dev/null | jq -r '"\(.ts) \(.tool) \(.file)"' | tail -30
```

## Step 2: Analyze Quality Gate Failures

Look at the git log for patterns in what went wrong:
- Which files triggered lint errors most often?
- Were there repeated test failures?
- Did any security patterns keep appearing?
- Were there conflict near-misses?

## Step 3: Identify CLAUDE.md Improvements

Based on the patterns you found, suggest specific additions to CLAUDE.md.
Format each suggestion as:

```markdown
### Common Mistakes
- DON'T: {what Claude did wrong}
  DO: {what Claude should do instead}
  WHY: {what went wrong when this happened}
```

## Step 4: Update Code Standards Skill

If you identified new patterns that aren't in
`.claude/skills/code-standards/SKILL.md`, add them to the
"Common Mistakes Catalog" or "Project-Specific Patterns" sections.

## Step 5: Sprint Metrics Summary

Report:
- Story points completed
- Tasks completed vs. planned
- Average cycle time per task
- Quality gate failure rate
- Top 3 most-edited files
- Lessons learned

Ask the human: "Should I apply these CLAUDE.md updates? Any corrections?"
The human reviews and approves before changes are committed.
