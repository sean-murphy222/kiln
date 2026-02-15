# Kiln

**A Complete Pipeline for Trustworthy Domain-Specific AI**

Quarry · Forge · Foundry · Hearth

---

## Overview

Kiln is a unified platform for building trustworthy, locally-owned, domain-specific AI systems. It addresses the two most critical failure points in current domain AI approaches:

1. **Quality of data for retrieval** (chunking problem)
2. **Quality of data for training** (synthetic data problem)

## The Four Tools

### 🪨 Quarry — Document Processing
Transforms complex domain documents into high-quality, metadata-enriched, retrieval-ready chunks. Structure-aware processing that preserves semantic boundaries and enables metadata-filtered retrieval.

**Status:** ~70% complete

### 🔨 Forge — Curriculum Builder
Guides domain experts through creating human-validated training data for fine-tuning small language models. Discipline-level curriculum methodology requiring no ML expertise.

**Status:** Not started (Phase 2)

### ⚙️ Foundry — Training & Evaluation
Manages model training, automated competency-based evaluation, version management, and regression testing. Translates ML complexity into SME-friendly workflows.

**Status:** Not started (Phase 3)

### 🔥 Hearth — Interaction Layer
Built-in chat interface where practitioners select trained models, connect to Quarry knowledge bases, and interact with the system. Feedback loop closes back to Forge and Quarry.

**Status:** Not started (Phase 4)

## Why Kiln?

**The Closed Loop:** Quarry + Forge work together. Chunking quality improves training data generation. Training data quality improves the model. Model quality improves RAG responses. Feedback improves both chunking and training.

**Human-Validated Provenance:** Every piece of training data authored and validated by identified domain experts. Fully auditable for regulated industries.

**Local Ownership:** Runs entirely on your hardware. Owns trained model, processed knowledge base, validated curriculum. No cloud dependency, no per-token costs.

**Compounding Assets:** Quarry's structural profile library and Forge's facilitator model improve with each use. Platform becomes more capable over time.

## MVP Timeline

22 weeks (5 months) part-time development across 11 two-week sprints:

- **Phase 1 (Sprints 1-3):** Quarry Completion
- **Phase 2 (Sprints 4-7):** Forge Core Framework
- **Phase 3 (Sprints 8-10):** Foundry + Integration
- **Phase 4 (Sprint 11):** Hearth + MVP Package

See `BACKLOG.md` for detailed sprint plan.

## Architecture

```
kiln/
├── quarry/          # Document processing pipeline (Python + FastAPI)
├── forge/           # Curriculum builder (Python + React)
├── foundry/         # Training & evaluation (Python)
├── hearth/          # Interaction layer (integrated into Kiln UI)
├── shared/          # Common utilities and data models
├── ui/              # Kiln unified interface (Electron + React + TypeScript)
├── docs/            # Documentation
└── scripts/         # Development and deployment scripts
```

See `ARCHITECTURE.md` for detailed system design.

## Quick Start

```bash
# Clone repository
git clone https://github.com/sean-murphy222/kiln.git
cd kiln

# Install Python dependencies
pip install -e .

# Optional: Install enhanced extraction (Docling - RECOMMENDED)
pip install kiln[enhanced]

# Run backend
cd quarry && uvicorn src.server:app --reload --port 8420

# Run UI (separate terminal)
cd ui && npm install && npm run electron:dev
```

## Development Workflow

This project uses the **Autonomous Scrum Orchestrator** framework. See `.claude/` for hooks, agents, skills, and slash commands.

Key commands:
- `/sprint-start` — Initialize sprint session
- `/pick-task` — Select next task from backlog
- `/done` — Mark task complete (triggers quality gates)
- `/review` — Trigger security + QA review
- `/status` — Show sprint progress

See `CLAUDE.md` for complete development guide.

## License

[License TBD]

## Contact

[Contact information TBD]
