# CLAUDE.md — Kiln Operating Manual

> This file is your primary reference. Follow it precisely. When you make a
> mistake, the human will correct you and tell you to update this file.
> Every correction makes you better. This file compounds over time.

## About This Project

**Project:** Kiln — A Complete Pipeline for Trustworthy Domain-Specific AI
**Stack:** Python (FastAPI, Unsloth/Axolotl), React (TypeScript, Tailwind), Electron
**Purpose:** Enable organizations to build accurate, auditable, locally-owned domain-specific AI systems

### The Four Tools

**Quarry** (70% complete) — Document processing pipeline
- Transforms complex documents into metadata-enriched retrieval-ready chunks
- Structure-aware processing preserves semantic boundaries
- ML classifier (not LLM) for document type detection
- Metadata-filtered retrieval (3-stage: filter → search → validate)

**Forge** (not started, Phase 2) — Curriculum builder
- Guides domain experts through creating human-validated training data
- Discipline-level training methodology (not document-level)
- Multi-contributor support with consistency checking
- Target: 300-500 examples per discipline

**Foundry** (not started, Phase 3) — Training & evaluation
- LoRA training pipeline with sensible defaults
- Automated competency-based evaluation (plain language, not ML metrics)
- Regression testing on every change
- Model merging support (optional)

**Hearth** (not started, Phase 4) — Interaction layer
- Built-in chat interface within Kiln (not standalone deployment)
- Model switching, document browsing, citation display
- Feedback capture routed to Quarry/Forge improvements
- Human authority maintained (NO synthetic training data from feedback)

## Key Directories

```
kiln/
├── quarry/          # Document processing (Python + FastAPI) - PRIMARY FOCUS
│   ├── src/
│   │   ├── tier1/        # Structural fingerprinting + ML classifier
│   │   ├── tier2/        # Docling content extraction
│   │   ├── tier3/        # Hierarchy construction + QA + metadata
│   │   └── retrieval/    # 3-stage metadata-filtered retrieval
│   └── tests/
├── forge/           # Curriculum builder (Python + React) - PHASE 2
│   ├── src/
│   │   ├── discovery.py      # Step 1: Discipline discovery interview
│   │   ├── competency.py     # Step 2: Competency mapping
│   │   ├── examples.py       # Step 3: Example elicitation
│   │   └── consistency.py    # Step 4: Quality scaffolding
│   └── tests/
├── foundry/         # Training & evaluation (Python) - PHASE 3
│   ├── src/
│   │   ├── training.py       # LoRA training pipeline
│   │   ├── evaluation.py     # Competency-based evaluation
│   │   ├── rag_integration.py # Connect LoRA + Quarry
│   │   └── regression.py     # Version management + regression testing
│   └── tests/
├── hearth/          # Interaction layer (Python inference) - PHASE 4
│   ├── src/
│   │   ├── inference.py      # Model loading + query handling
│   │   └── feedback.py       # Signal capture + routing
│   └── tests/
├── shared/          # Common utilities and data models
├── ui/              # Kiln unified interface (Electron + React + TypeScript)
├── docs/            # Documentation
├── scripts/         # Development and deployment scripts
└── .claude/         # Scrum orchestrator (hooks, agents, skills, commands)
```

## Commands

```bash
# Quarry development (PRIMARY FOCUS - Sprints 1-3)
cd quarry && uvicorn src.server:app --reload --port 8420
pytest tests/ --cov=src --cov-report=term-missing
ruff check src/
black src/
mypy src/
bandit -r src/

# Forge development (PHASE 2 - Sprints 4-7)
cd forge && pytest tests/
ruff check src/
black src/

# Foundry development (PHASE 3 - Sprints 8-10)
cd foundry && pytest tests/
ruff check src/
python src/training.py  # LoRA training

# Hearth development (PHASE 4 - Sprint 11)
cd hearth && pytest tests/

# UI development (all phases)
cd ui && npm install
npm run dev              # Vite dev server
npm run electron:dev     # Electron + Vite together
npm test                 # Jest tests
npm run lint             # ESLint
npm run typecheck        # TypeScript checking
```

## Workflow — TDD is Non-Negotiable

1. **Plan first** — Enter plan mode (Shift+Tab x2) for any task over 3 story points
2. **Tests first** — Write failing tests before any source code (red-green-refactor)
3. **Small commits** — Commit after each red-green-refactor cycle
4. **Conventional Commits** — `feat(scope): description`, `fix(scope): description`
5. **Branch per task** — `feature/T-{id}-{short-description}` (e.g., `feature/T-001-tier1-fingerprinter`)
6. **Never commit to main** — The branch-protector hook will block you anyway

### Sprint Workflow (Use the /commands)

```bash
# Morning: Start sprint session
/sprint-start          # Loads context, runs conflict analysis, enters plan mode

# During work: Pick and complete tasks
/pick-task             # Select highest-priority unblocked Ready task
# ... implement with TDD ...
/done                  # Mark complete (triggers H-16 definition of done gate)

# Before push
/review                # Spawn security + QA review subagents
/ship                  # Final validation, then push

# Evening: Cleanup and learning
/techdebt              # Scan for TODOs, duplicated code, tech debt
/retrospective         # Analyze session, suggest CLAUDE.md updates
/status                # Show sprint progress
```

## Code Conventions

### Python

- All functions have docstrings (Google style)
- All function signatures have type hints
- No `eval()`, `exec()`, or dynamic code execution
- No hardcoded secrets (use environment variables)
- No SQL string concatenation (parameterized queries only)
- Maximum 50 lines per function (extract helpers if longer)
- Error messages don't leak internals
- Imports: stdlib → third-party → project (blank line between groups)

**Example:**
```python
from pathlib import Path
from typing import List, Dict

import numpy as np
from docling.document_converter import DocumentConverter

from quarry.src.tier1.classifier import DocumentClassifier

def extract_features(doc_path: Path) -> Dict[str, float]:
    """Extract statistical features from document for classification.

    Args:
        doc_path: Path to PDF document

    Returns:
        Dictionary of feature names to values

    Raises:
        ValueError: If document cannot be read
    """
    # Implementation...
```

### TypeScript/React

- Functional components only (no class components)
- Props interfaces defined explicitly
- All state changes through hooks (useState, useReducer, Zustand)
- No inline styles (use Tailwind classes)
- Components max 200 lines (extract subcomponents)
- Use absolute imports (`@/components`, not `../../components`)

**Example:**
```typescript
interface EvaluationResultsProps {
  disciplineId: string;
  results: CompetencyResults[];
  onReview: (competencyId: string) => void;
}

export const EvaluationResults: React.FC<EvaluationResultsProps> = ({
  disciplineId,
  results,
  onReview
}) => {
  // Implementation...
};
```

## Core Architectural Principles

### 1. Human Validation Over Automation
**Critical:** Training data is ALWAYS human-validated. Never generate synthetic training data from model outputs or user interactions. This is the core value proposition and cannot be compromised.

When capturing feedback in Hearth:
- ✓ DO: Surface opportunities for improvement to discipline owners
- ✓ DO: Let humans decide whether to add examples to Forge
- ✗ DON'T: Auto-generate training data from feedback
- ✗ DON'T: Use model outputs as training data

### 2. ML Over LLM for Classification
Quarry Tier 1 uses traditional ML (random forest, gradient boost) for document classification, NOT LLMs.

**Why:**
- Format detection is a classification problem with well-defined features
- ML is faster, more deterministic, more interpretable
- Runs without GPU, produces feature importances
- LLM would be influenced by content when only structure matters

### 3. Metadata-Filtered Retrieval is Key
Don't just embed and search. Always:
1. **Stage 1:** Deterministic pre-filter on metadata (reduces search space 80-90%)
2. **Stage 2:** Semantic search on filtered subset
3. **Stage 3:** Validation pass against expected patterns

This approach is computationally efficient (critical for local deployment) and dramatically reduces false positives.

### 4. Discipline-Level Training
Forge trains at the **discipline level**, not document level.

**What this means:**
- Train the model on how the discipline works (vocabulary, conventions, reasoning)
- Specific factual content comes from RAG at inference time
- 300-500 examples cover full competency range for a discipline
- Adding new documents = Quarry task (process + index), NOT training task

### 5. Competency-Based Evaluation
Foundry reports results in SME language, not ML metrics.

**SME-friendly:**
- "Procedural comprehension: 9/10 correct"
- "Fault isolation reasoning: 7/10 correct"
- "Add more examples for parts interpretation (weak area)"

**NOT SME-friendly:**
- "Validation loss: 0.42"
- "Perplexity: 8.3"
- "F1 score: 0.87"

### 6. Local-First, Always
Every design decision considers local deployment:
- Models run on laptops (quantized 7-8B) or workstations (20B)
- No cloud connectivity required after setup
- Metadata pre-filter reduces GPU load
- Efficient data structures for resource constraints

## Security Checklist (Run Before Every Commit)

- [ ] No hardcoded API keys, tokens, or passwords
- [ ] No eval/exec with external input
- [ ] All user input is validated/sanitized
- [ ] File paths validated against traversal attacks
- [ ] PDF parsing sandboxed with resource limits (Tier 2)
- [ ] Error responses don't leak internal details
- [ ] New dependencies checked for CVEs (H-09 hook does this)

### Quarry-Specific Security

- [ ] PDF extraction runs in isolated process with timeouts
- [ ] File size limits enforced (default 100MB, configurable)
- [ ] Path traversal prevention in file operations
- [ ] No shell commands with user-controlled input

### Forge/Foundry-Specific Security

- [ ] Training data stored with access controls
- [ ] PII redaction before any data export
- [ ] Contributor attribution tracked
- [ ] Curriculum versioning and audit trail
- [ ] No sensitive metadata in model weights

## Common Mistakes

<!-- THIS SECTION IS THE MOST IMPORTANT. IT COMPOUNDS OVER TIME. -->
<!-- After every correction, add entries here: -->
<!-- - DON'T: {what you did wrong} → DO: {what to do instead} -->

### Document Processing
- DON'T: Use LLM for document type classification → DO: Use traditional ML classifier (faster, deterministic, inspectable)
- DON'T: Rely solely on semantic chunking → DO: Use structural hierarchy from document formatting
- DON'T: Skip metadata enrichment → DO: Extract metadata from formatting cues (essential for filtered retrieval)

### Training Data
- DON'T: Generate synthetic training data from models → DO: Always use human-validated examples
- DON'T: Train on document content → DO: Train on discipline patterns (content comes from RAG)
- DON'T: Present ML metrics to SMEs → DO: Report in competency language they understand

### Retrieval
- DON'T: Embed everything and hope → DO: Pre-filter with metadata first (80-90% search space reduction)
- DON'T: Return partial semantic units → DO: Ensure chunks are semantically complete with proper boundaries

## Enforcement (19 Hooks Active)

### Critical Hooks You'll Interact With

**H-14: End-of-Turn Quality Gate** ⭐ THE MOST IMPORTANT
- Runs: tests, lint, security scan, secrets check, conflict detection
- You CANNOT end a turn with broken code
- Timeout: 120 seconds
- Exit 2 = keep working until all checks pass

**H-06: Branch Protector**
- You CANNOT commit or push on main
- All work on feature branches: `feature/T-{id}-{description}`
- Force push blocked unconditionally

**H-04: Secrets Firewall**
- You CANNOT modify .env, secrets, credentials, or key files
- Read-only operations allowed (cat .env.example, git diff)

**H-05: Protected File Write Guard**
- Blocks writes to: .env*, secrets*, .claude/settings.json, *.pem, *.key
- Prevents modifying governance config

**H-10/H-11: Auto-Formatter + Linter**
- After every file write/edit: black/ruff (Python), prettier/eslint (JS/TS)
- Feeds violations back so you can fix immediately
- Catches the 10% where your formatting isn't perfect

**H-16: Definition of Done Gate**
- Tasks CANNOT be marked complete without:
  - Test files for every changed source file
  - All tests passing
  - Conventional Commit messages
  - No merge conflicts
  - Changed files committed
  - BACKLOG.md updated

**H-15: Task Continuation Nudge**
- After quality gate passes, if tasks remain in backlog
- Keeps sprint loop going until backlog exhausted

**H-01/H-02: Session Context Management**
- H-01: Injects current branch, recent commits, active task at startup
- H-02: Recovers context after compaction (reads BACKLOG.md, session log)
- Prevents you from forgetting which task you're working on

**H-18: Context Backup (before compaction)**
- Saves task state, progress notes to `.claude/backups/`
- H-02 recovers this after compaction
- Safety net for long autonomous sessions

These hooks are deterministic. They run even if you forget these instructions.

## Test Coverage Requirements

**Minimum coverage:** 80% line coverage
**Critical paths:** 100% coverage required for:
- Quarry Tier 1 classifier (document type detection)
- Forge consistency checking
- Foundry evaluation pipeline
- Hearth feedback routing logic

**Test categories:**
- Unit tests: Every module, function, class
- Integration tests: Component boundaries (Quarry→Forge, Forge→Foundry, etc.)
- End-to-end tests: Full pipeline (Sprint 10)
- Regression tests: Prevent known bugs from returning

## Performance Targets

**Quarry:**
- Tier 1 fingerprinting: < 5 seconds per document
- Tier 1 classification: < 1 second per document
- Tier 2 extraction: < 30 seconds per 100-page document (with Docling GPU acceleration)
- Metadata-filtered retrieval Stage 1: < 100ms (deterministic filter)
- Metadata-filtered retrieval Stage 2: < 500ms (semantic search on filtered set)

**Forge:**
- Discipline discovery: 45-60 minutes total (user perception, not system)
- Competency mapping: 15-20 minutes total
- Example elicitation: < 2 seconds per example save

**Foundry:**
- LoRA training: < 2 hours for 300-500 examples on modest GPU
- Evaluation: < 5 minutes for full test suite (100+ examples)
- Model merging: < 5 minutes (no retraining)

**Hearth:**
- Query latency: < 3 seconds end-to-end (retrieval + generation + citation)
- Model switching: < 10 seconds (LoRA load time)

## MVP Scope Boundaries

**IN SCOPE:**
- PDF documents with embedded text layers
- Military technical manuals (primary test domain)
- Single discipline demonstration (military maintenance)
- Local deployment on laptop/workstation

**OUT OF SCOPE (Post-MVP):**
- OCR for scanned/image-based PDFs
- Multi-discipline simultaneous deployment
- Cloud deployment
- Subscription infrastructure
- Second discipline validation (non-military)

Stay focused. The MVP is proof of concept, not production product.

## When to Use Which Agent

The `.claude/agents/` directory contains 6 specialized agents. Here's when to spawn them:

**scrum-master.md** (Opus, Lead)
- Use: As team lead in Agent Teams mode or autonomous sprint sessions
- Can: Read, git operations, task management — NO Edit/Write
- Maintains conflict map, assigns tasks, reviews output

**developer.md** (Sonnet, Teammate)
- Use: Implementing tasks with TDD workflow
- Can: All standard tools
- Follows red-green-refactor, works in isolated worktree

**security-reviewer.md** (Sonnet, Subagent)
- Use: Spawned by /review command after implementation
- Can: Read, Grep, security scans — NO Edit/Write
- Reviews diffs against OWASP patterns, scans for secrets, produces structured report

**qa-engineer.md** (Sonnet, Subagent)
- Use: Spawned by /review command for test validation
- Can: Read, coverage tools, can Write test files only
- Validates test quality, checks coverage, writes additional edge case tests

**architect.md** (Opus, Subagent)
- Use: For tasks touching multiple modules or creating new ones
- Can: Read only — NO Edit/Write/Bash
- Reviews architectural compliance, validates module boundaries

**plan-reviewer.md** (Opus, Subagent)
- Use: After plan mode produces a plan, before implementation
- Can: Read only — NO Edit/Write
- Challenges assumptions, identifies edge cases, must approve plan

## File Naming Conventions

**Python:**
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_{module_name}.py`

**TypeScript:**
- Components: `PascalCase.tsx`
- Utilities: `camelCase.ts`
- Types/interfaces: `PascalCase`
- Hooks: `use{CapitalizedName}.ts`
- Test files: `{component}.test.tsx`

## Git Commit Message Format

Follow Conventional Commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Types:** feat, fix, docs, style, refactor, test, chore
**Scopes:** quarry, forge, foundry, hearth, ui, shared, docs

**Examples:**
```
feat(quarry): implement tier1 structural fingerprinting

Add statistical document analysis for feature extraction.
Covers byte patterns, formatting markers, whitespace dist.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

```
fix(forge): resolve competency coverage calculation bug

Coverage percentage was double-counting held-out test examples.
Now correctly excludes test set from coverage metrics.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Documentation Requirements

Every module must have:
- README.md with overview, status, key concepts
- Docstrings on all public functions/classes
- Type hints (Python) or TypeScript interfaces
- Example usage in docstrings
- Edge cases and limitations documented

API endpoints require:
- Request/response schemas (Pydantic models)
- Example requests with curl
- Error responses documented
- Rate limits noted (if applicable)

## Known Limitations (To Document, Not Fix in MVP)

**Quarry:**
- Works best with structurally consistent, well-formatted documents
- Degrades gracefully on poorly formatted docs (expected)
- No OCR support (scanned PDFs out of scope)

**Forge:**
- Facilitator model not included in MVP (framework-only)
- Single-discipline focus (multi-discipline post-MVP)

**Foundry:**
- Local training only (no distributed training)
- Small models only (3-20B parameters)
- Merging limited to linear/TIES (no more advanced techniques)

**Hearth:**
- Sequential inference for multi-discipline queries (slower)
- Local deployment only (no cloud inference)

These are deliberate MVP scope boundaries, not bugs.

## Emergency Contacts / Escalation

If you encounter:
- **Docling GPU errors:** Check CUDA installation, fall back to CPU mode
- **Classifier accuracy < 70%:** Flag for manual review, more training data needed
- **LoRA training not converging:** Check Forge curriculum quality, may need more examples
- **Retrieval precision < 80%:** Review Quarry metadata enrichment, may need better structural profiles

## For Developers

When working on Kiln, remember the priority order:

**Priority 1:** Quarry completion (Sprints 1-3)
**Priority 2:** Forge framework (Sprints 4-7)
**Priority 3:** Foundry integration (Sprints 8-10)
**Priority 4:** Hearth + MVP packaging (Sprint 11)

**The MVP is not the product.** It is proof the product works. Everything after MVP builds on validated foundations rather than assumptions.

Stay focused on the current sprint. Resist scope creep. Ship incrementally.

---

## Changelog

<!-- Track major updates to this file -->

### 2026-02-15
- Initial CLAUDE.md created for Kiln project
- Integrated scrum orchestrator template with Kiln-specific context
- Documented four-tool architecture, MVP roadmap, core principles
