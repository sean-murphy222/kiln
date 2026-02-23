# Kiln Architecture Overview

Kiln is a four-tool pipeline for building trustworthy, domain-specific AI systems. Each tool handles a distinct phase of the pipeline, and all tools are designed for local deployment without cloud dependencies.

---

## The Four Tools

### Quarry -- Document Processing

Transforms complex documents (primarily PDFs with embedded text) into metadata-enriched, retrieval-ready chunks.

**Key design decisions:**
- Uses traditional ML (GradientBoostingClassifier) for document type classification, not LLMs. Format detection is a classification problem with well-defined features; ML is faster, deterministic, and inspectable.
- Structure-aware processing preserves semantic boundaries rather than using naive fixed-length chunking.
- 3-stage metadata-filtered retrieval reduces search space by 80-90% before semantic search.

**Status:** 70% complete (Sprints 1-3)

### Forge -- Curriculum Builder

Guides domain experts through creating human-validated training data at the discipline level (not document level).

**Key design decisions:**
- Training data is always human-validated. Kiln never generates synthetic training data from model outputs.
- Trains at the discipline level (vocabulary, conventions, reasoning patterns). Specific factual content comes from RAG at inference time.
- Target: 300-500 examples per discipline covering the full competency range.

**Status:** Phase 2 (Sprints 4-7)

### Foundry -- Training and Evaluation

Handles LoRA fine-tuning, competency-based evaluation, regression testing, and model merging.

**Key design decisions:**
- Reports evaluation results in SME-friendly language ("Procedural comprehension: 9/10 correct"), not ML metrics (loss, perplexity, F1).
- Regression testing on every model change to prevent capability degradation.
- Dry-run backend for MVP testing; production backends (Unsloth, Axolotl) plug in without interface changes.

**Status:** Phase 3 (Sprints 8-10)

### Hearth -- Interaction Layer (Planned)

Built-in chat interface within Kiln for model interaction, citation display, and feedback capture.

**Key design decisions:**
- Feedback is surfaced to discipline owners for review; it is never auto-converted to training data.
- Human authority maintained at all times.

**Status:** Phase 4 (Sprint 11)

---

## Data Flow

```
                            QUARRY
  +-----------+    +---------------------+    +------------------+
  |           |    |  Tier 1: Structure  |    | Tier 3: Hierarchy|
  |  PDF/DOCX +--->+  Fingerprinting +   +--->+ QA Filters +     |
  |  Documents|    |  ML Classifier      |    | Metadata Enrichmt|
  +-----------+    +---------------------+    +--------+---------+
                                                       |
                                                       v
                                              +-----------------+
                                              | Export Pipeline  |
                                              | JSONL / ChonkRec |
                                              | Vector DB Adaptr |
                                              +--------+--------+
                                                       |
                           +---------------------------+
                           |                           |
                           v                           v
                   +---------------+          +-----------------+
                   |  Retrieval    |          |  FORGE          |
                   |  (3-Stage)    |          |                 |
                   |  Filter ->    |          | Discovery       |
                   |  Search ->    |          | Competency Map  |
                   |  Validate     |          | Example Elicit. |
                   +-------+-------+          +--------+--------+
                           |                           |
                           |                  Alpaca JSONL Export
                           |                           |
                           |                           v
                           |                  +-----------------+
                           |                  |  FOUNDRY        |
                           |                  |                 |
                           +----------------->+ LoRA Training   |
                                              | Evaluation      |
                                              | Regression Test |
                                              | Model Merging   |
                                              +--------+--------+
                                                       |
                                                       v
                                              +-----------------+
                                              |  HEARTH         |
                                              |  (Planned)      |
                                              |  Chat + RAG +   |
                                              |  Citations +    |
                                              |  Feedback       |
                                              +-----------------+
```

---

## Quarry Architecture

### Tier 1: Structural Fingerprinting + ML Classification

Analyzes raw PDF structure statistically without content parsing to produce a 49-feature vector.

**Modules:**
- `chonk.tier1.fingerprinter` -- Extracts 6 feature groups: byte-level, font, layout, character, repetition, structural rhythm
- `chonk.tier1.classifier` -- GradientBoostingClassifier (200 estimators, max depth 4, confidence threshold 0.45)
- `chonk.tier1.taxonomy` -- 14 document types + UNKNOWN sentinel, with structural profiles for training data generation
- `chonk.tier1.training_data` -- Synthetic training corpus generation from document type profiles
- `chonk.tier1.manual_store` -- Manual type override storage for classifier corrections
- `chonk.tier1.fallback` -- Fallback chain: classifier -> manual store -> UNKNOWN

**Performance:** Fingerprinting < 5 seconds, classification < 1 second per document.

### Tier 2: Content Extraction (Docling)

Uses IBM's Docling library for deep content extraction with layout awareness.

**Performance:** < 30 seconds per 100-page document with GPU acceleration.

### Tier 3: Hierarchy + QA + Metadata

Builds document hierarchy from structural cues, applies quality filters, and enriches metadata.

**Modules:**
- `chonk.hierarchy` -- Stack-based nesting from flat block sequences using numbering patterns
- `chonk.qa` -- Stamp-based filter pipeline with audit trail (patterns, rules, filter log)
- `chonk.cleaning` -- 5-operation normalizer: characters, hyphenation, continuation, artifacts, whitespace
- `chonk.enrichment` -- Metadata extraction from formatting cues (essential for filtered retrieval)

### Export Pipeline

Converts processed chunks to multiple output formats.

**Modules:**
- `chonk.exporters.schema` -- ChonkRecord canonical format with schema versioning (currently v1.1)
- `chonk.exporters.jsonl` -- JSONL export compatible with LangChain, LlamaIndex
- `chonk.exporters.json_export` -- JSON export
- `chonk.exporters.csv_export` -- CSV export
- `chonk.exporters.base` -- BaseExporter ABC and ExporterRegistry

### 3-Stage Retrieval

1. **Stage 1 (< 100ms):** Deterministic pre-filter on metadata -- reduces search space 80-90%
2. **Stage 2 (< 500ms):** Semantic search on the filtered subset
3. **Stage 3:** Validation pass against expected patterns

---

## Forge Architecture

Forge implements a 4-step curriculum building workflow.

### Step 1: Discovery Interview

`forge.src.discovery.DiscoveryEngine` runs a structured questionnaire (15 questions across 4 phases: Orientation, Documents, Competencies, Vocabulary) to surface discipline characteristics. Framework-only, no LLM.

### Step 2: Competency Mapping

`forge.src.competency.CompetencyMapper` translates discovery outputs into a hierarchical competency map. Experts refine and validate competencies, set coverage targets (default 25 examples per competency).

### Step 3: Example Elicitation

`forge.src.examples.ExampleElicitor` guides experts through creating training examples with competency tagging, reasoning pattern classification (procedural, diagnostic, factual, analytical, comparative, safety), and draft management.

### Step 4: Quality Scaffolding

Consistency checking and multi-contributor support. Coverage reports track progress toward the 300-500 example target.

### Storage

`forge.src.storage.ForgeStorage` uses SQLite for all persistence. Tables: contributors, disciplines, competencies, examples, discipline_contributors, curriculum_versions, discovery_sessions. Supports JSONL export in Alpaca format for Foundry consumption.

---

## Foundry Architecture

### Training Pipeline

`foundry.src.training.TrainingPipeline` handles the complete LoRA training workflow:
- CurriculumLoader validates and splits data (train/validation)
- HyperparameterAutoConfig computes sensible defaults from data statistics
- TrainingRegistry tracks all runs with JSON persistence
- Supports 4 base model families: Phi, LLaMA, Mistral, Qwen

### Evaluation System

`foundry.src.evaluation.EvaluationRunner` tests models against held-out test sets. Reports competency scores in plain language with ratings (Strong, Adequate, Needs Improvement, Weak). The ModelInference protocol abstracts inference for testability.

### RAG Integration

`foundry.src.rag_integration.RAGPipeline` connects LoRA models with Quarry retrieval. Query -> retrieval -> context building -> generation -> citation extraction. The RetrievalAdapter protocol allows swapping retrieval backends.

### Regression Testing

`foundry.src.regression.RegressionChecker` compares evaluation reports across versions. Detects regressions by severity (Minor > 10%, Major > 20%, Critical > 30%). VersionManager supports rollback when regressions are detected.

### Model Merging

`foundry.src.merging.MergePipeline` combines multiple LoRA adapters via Linear interpolation or TIES (TrIm, Elect Sign, merge). CompatibilityChecker validates adapter compatibility before merge.

### Diagnostics

`foundry.src.diagnostics.TrainingDiagnostics` monitors training metrics and detects issues (convergence failure, overfitting, instability, data quality). All guidance is in plain language linking back to Forge when curriculum changes could help.

---

## Core Architectural Principles

### 1. Human Validation Over Automation

Training data is always human-validated. Kiln never generates synthetic training data from model outputs or user feedback. This is the core value proposition.

### 2. ML Over LLM for Classification

Quarry Tier 1 uses traditional ML (GradientBoosting) for document type detection. ML is faster (< 1ms inference), deterministic, and produces inspectable feature importances. LLMs would be influenced by content when only structure matters.

### 3. Metadata-Filtered Retrieval

Always pre-filter with metadata before semantic search. This reduces GPU load (critical for local deployment) and dramatically reduces false positives.

### 4. Discipline-Level Training

Forge trains on discipline patterns (vocabulary, conventions, reasoning), not document content. Specific facts come from RAG at inference time. 300-500 examples cover a full discipline.

### 5. Competency-Based Evaluation

Results are reported in SME language, not ML metrics. "Procedural comprehension: 9/10 correct" instead of "F1 score: 0.87."

### 6. Local-First Design

Every decision considers local deployment: quantized 7-8B models on laptops, metadata pre-filtering for efficiency, no cloud connectivity required after setup.

---

## Storage Patterns

| Tool | Storage | Pattern |
|---|---|---|
| Quarry | File system + export files | Processes documents in memory, exports to JSONL/JSON/CSV |
| Forge | SQLite (ForgeStorage) | Single database with CRUD, curriculum versioning, JSONL export |
| Foundry | JSON files (registries) | TrainingRegistry, MergeRegistry, VersionManager all use JSON file persistence |
| Hearth | Planned | TBD |

---

## Security Boundaries

- PDF parsing uses file size limits (default 100 MB, configurable via `DocumentFingerprinter.max_file_size`)
- Path traversal prevention in file operations
- No `eval()`, `exec()`, or dynamic code execution
- No hardcoded secrets; environment variables for configuration
- SQL parameterized queries only (ForgeStorage uses `?` placeholders)
- Classifier model files use joblib; do not load from untrusted sources (documented in `DocumentClassifier.save()` and `DocumentClassifier.load()`)
