# Changelog

All notable changes to Kiln will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### MVP Roadmap
- Phase 1 (Sprints 1-3): Quarry Completion
- Phase 2 (Sprints 4-7): Forge Core Framework
- Phase 3 (Sprints 8-10): Foundry + Integration
- Phase 4 (Sprint 11): Hearth + MVP Package

---

## [0.1.0] - 2026-02-15

### Added - Project Foundation

**Infrastructure:**
- Initialized git repository and connected to GitHub (https://github.com/sean-murphy222/kiln.git)
- Set up monorepo structure with four main modules (quarry, forge, foundry, hearth)
- Installed Autonomous Scrum Orchestrator framework with 19 hooks across 9 lifecycle events
- Configured dual-stack development environment (Python + Node.js)

**Documentation:**
- Created comprehensive README.md with project overview and quick start guide
- Created CLAUDE.md operating manual with development workflow and code conventions
- Created ARCHITECTURE.md with complete system design and technical decisions
- Created BACKLOG.md with full 11-sprint roadmap (32 tasks across 4 phases)
- Added module-specific README files for quarry/, forge/, foundry/, hearth/, shared/

**Scrum Orchestrator:**
- Installed all 19 hooks (H-01 through H-19)
  - Session context management (H-01, H-02, H-03)
  - Security firewalls (H-04, H-05, H-06, H-07, H-08)
  - Quality gates (H-14, H-16)
  - Auto-formatter and linter (H-10, H-11)
  - Dependency auditor (H-09)
  - Change logger (H-12)
  - Task continuation (H-15)
  - Context backup (H-18)
- Installed 6 specialized agents (scrum-master, developer, security-reviewer, qa-engineer, architect, plan-reviewer)
- Installed 5 skills (tdd, security-review, conflict-analysis, code-standards, sprint-management)
- Installed 9 slash commands (/sprint-start, /pick-task, /done, /review, /conflicts, /status, /ship, /techdebt, /retrospective)

**Project Structure:**
```
kiln/
├── quarry/          # Document processing (Python + FastAPI) - ~70% complete
├── forge/           # Curriculum builder (Python + React) - not started
├── foundry/         # Training & evaluation (Python) - not started
├── hearth/          # Interaction layer (Python) - not started
├── shared/          # Common utilities
├── ui/              # Kiln unified interface (Electron + React + TypeScript)
├── docs/            # Documentation
├── scripts/         # Utility scripts
└── .claude/         # Scrum orchestrator configuration
```

**Quarry (Existing Work - ~70% Complete):**
- Docling integration functional
- Basic hierarchy construction implemented
- Existing extraction pipeline operational
- TODO: Tier 1 structural fingerprinting, metadata enrichment, QA pass, retrieval integration

### Security

**Hooks Enforcing Security:**
- H-04: Secrets Firewall (blocks .env modifications)
- H-05: Protected File Write Guard (blocks writes to sensitive paths)
- H-06: Main Branch Protector (prevents commits to main)
- H-07: Force Push & Destructive Git Guard
- H-08: System Destructive Command Guard
- H-09: Dependency Install Auditor (CVE checking)

**Best Practices:**
- All secrets in environment variables (never hardcoded)
- .gitignore configured for .env, credentials, keys
- PDF parsing sandboxed with resource limits
- Input validation on all file operations
- Path traversal prevention

### Notes

This release establishes the project foundation. No functional features shipped yet - this is infrastructure setup.

**Next Sprint (Sprint 1):** T-001, T-002, T-003 - Quarry Tier 1 structural fingerprinting and ML classifier

---

## Version Numbering

**Current Component Versions:**
- Quarry: v0.7.0 (~70% complete at project start)
- Forge: v0.0.0 (not started)
- Foundry: v0.0.0 (not started)
- Hearth: v0.0.0 (not started)
- Kiln (overall): v0.1.0

**MVP Target:** v1.0.0 (Sprint 11 completion)

---

## Links

- [Repository](https://github.com/sean-murphy222/kiln)
- [Issue Tracker](https://github.com/sean-murphy222/kiln/issues)
- [Sprint Backlog](BACKLOG.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Development Guide](CLAUDE.md)
