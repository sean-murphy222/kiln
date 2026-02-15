# /review — Security + QA Review Cycle

Spawn two review subagents in parallel to keep the main context clean:

1. **Security Review**: Use the @security-reviewer agent to analyze the current
   branch diff against origin/main. It will produce a structured security report
   with findings categorized by severity.

2. **QA Review**: Use the @qa-engineer agent to validate test coverage, test
   quality, and run the full regression suite. It will produce a structured
   QA report with coverage metrics.

Both agents must produce a PASS verdict for the task to proceed to /done.

If either agent returns FAIL:
- Read their specific findings
- Fix the issues they identified
- Re-run /review

Do NOT proceed to /done until both reviews pass.
