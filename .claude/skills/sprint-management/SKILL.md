---
name: sprint-management
description: >
  Sprint backlog management procedures. Apply when starting a sprint,
  picking tasks, updating task status, or completing sprint ceremonies.
disable-model-invocation: true
---

# Sprint Management Skill

## BACKLOG.md Format

Tasks follow this format:
```markdown
- [ ] **T-{NNN}** | {priority: P0/P1/P2} | {story points: 1/2/3/5/8}
  **Title:** {short description}
  **Description:** {what needs to be done}
  **Acceptance Criteria:**
  - {criterion 1}
  - {criterion 2}
  **Files:** {expected files to modify}
  **Depends On:** {task IDs or "None"}
  **Blocked By:** {active task IDs that touch same files, or "None"}
```

## Task State Machine

```
Ready → In Progress → In Review → Done
  ↑         |              |
  └─────────┘              │
  (blocked/failed)         │
  ↑                        │
  └────────────────────────┘
  (review rejected)
```

- **Ready**: All dependencies met, no file conflicts with active tasks
- **In Progress**: Developer actively working, branch created
- **In Review**: Implementation done, security + QA review in progress
- **Done**: All reviews passed, definition of done met

## Sprint Metrics

Track at the bottom of BACKLOG.md:
```markdown
## Sprint Metrics
- **Velocity:** {story points completed this sprint}
- **Tasks Completed:** {N}
- **Average Cycle Time:** {hours per task}
- **Quality Gate Failures:** {how many times H-14 blocked}
- **Conflicts Prevented:** {overlaps detected and avoided}
```

## Priority Levels

- **P0 — Critical:** Must complete this sprint. Blocks deployment.
- **P1 — High:** Should complete this sprint. Important feature/fix.
- **P2 — Normal:** Complete if time allows. Enhancement/improvement.
- **Icebox:** Not scheduled. Tech debt, nice-to-haves, future ideas.

## Picking the Next Task

1. Sort by priority (P0 first)
2. Filter out blocked tasks (check "Depends On" and conflict map)
3. Among unblocked same-priority tasks, pick smallest story points first
4. Run conflict analysis before starting
