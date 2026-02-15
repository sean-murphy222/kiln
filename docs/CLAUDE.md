# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CHONK diagnoses why your RAG system fails and shows you how to fix it.**

### What CHONK Actually Is

CHONK is **not** just another chunking tool. It's a **diagnostic system** that makes retrieval failures visible and fixable.

**The Core Problem:**
Document chunking for RAG is a **black box**. When retrieval fails, you can't diagnose whether the problem is chunking, embedding, or search. Most tools chunk naively (fixed token windows) and the resulting semantic fragmentation causes retrieval failures that are **invisible and impossible to debug**.

**Two-Product Strategy:**

1. **Diagnostic Tool** (ship first, build fast)
   - Analyze existing chunks and identify why retrieval fails
   - This is the wedge product that creates the "aha moment"
   - Users see their chunking problems visualized
   - Generates training data while users pay you

2. **Chunking Engine** (ship second)
   - Structure-aware document processing that chunks correctly from the start
   - Users graduate to this after seeing their problems diagnosed
   - Trained on real-world correction data from the diagnostic tool

**The Diagnostic Flywheel:** Every time a user identifies a chunking problem through your tool, you capture training data for better chunking. You get paid to collect the data that improves your core technology.

**Core Value Proposition:**
- NOT: "Better chunking for all documents" (too broad, commoditizing)
- YES: "Understand why your RAG system fails and fix it" (immediate pain, clear value)

## Core Philosophy

### Diagnosis Before Optimization

**The Problem is Invisible:** When RAG retrieval fails, users don't know if it's their chunks, embeddings, or search algorithm. They can't see what's broken, so they can't fix it.

**CHONK makes failures visible:**
- Show exactly which chunks should have been retrieved
- Visualize how chunking fragmented the relevant content
- Identify the specific type of chunking failure (semantic, structural, etc.)
- Provide actionable fixes

### Four Categories of Chunk Problems

Every chunking failure falls into one of four programmatically detectable categories:

| Problem Type | Signal | Detection Method |
|--------------|--------|------------------|
| **Semantic Incompleteness** | Chunk contains partial idea | Sentence boundary detection, dangling connectives, embedding shift when adding neighbors |
| **Semantic Contamination** | Chunk contains multiple unrelated ideas | Sentence-level embedding clustering, topic coherence scoring |
| **Structural Breakage** | Chunk splits logical unit | List/table/procedure pattern matching, orphaned headers |
| **Reference Orphaning** | Chunk contains broken references | Regex for "see above," "as follows," cross-reference patterns |

### Query-Aware Diagnostics (Highest Value)

The killer feature is **query-to-failure tracing**:
1. User provides query that returned bad results
2. Tool identifies what **should** have been retrieved
3. Tool shows **how** chunking fragmented the relevant content
4. Visual overlay of chunk boundaries on source document

This creates the "aha moment" that converts free users to paid customers.

### Document Taxonomy

Different document types require different diagnostic approaches:

| Category | Complexity | Approach |
|----------|-----------|----------|
| Linear prose | Low | Existing tools work, light validation |
| Structured reference | Medium | Table/list-aware parsing |
| Procedural | Medium | Step sequence preservation |
| Multi-column technical | High | Layout analysis, reading order inference |
| Drawing packages | High | Field extraction, template matching |
| Volume-organized | Very High | Cross-document relationship mapping |

**Build a document classifier first** to route documents to appropriate diagnostic handlers.

### Structure-First (But Structure Isn't Enough)

Documents aren't flat text - they have hierarchy:
```
Document
├─ FOREWORD
├─ Section 1: Introduction
│  ├─ 1.1 Purpose
│  └─ 1.2 Scope
├─ Section 2: Requirements
│  ├─ 2.1 General Requirements
│  │  ├─ 2.1.1 Safety
│  │  └─ 2.1.2 Performance
│  └─ 2.2 Technical Requirements
└─ Appendix A
```

CHONK **preserves and visualizes** this structure, but more importantly, it **diagnoses when structure-aware chunking still fails** (e.g., when a section is too large, or when related content spans multiple sections).

## Architecture

### Backend (Python + FastAPI)

**Priority Order (What Matters Most):**

```
src/chonk/
├── diagnostics/             # 🎯 MVP FOCUS - Chunk problem detection
│   ├── analyzer.py          # Core diagnostic engine
│   ├── semantic_incomplete.py  # Detect partial ideas (dangling connectives, embedding shifts)
│   ├── semantic_contamination.py  # Detect mixed topics (embedding clustering, coherence)
│   ├── structural_breakage.py    # Detect split logical units (lists, tables, procedures)
│   ├── reference_orphaning.py    # Detect broken references (regex patterns)
│   ├── query_tracer.py      # 🌟 KILLER FEATURE - Query-to-failure tracing
│   ├── visualizer.py        # Generate chunk boundary overlays on source doc
│   └── correction_capture.py  # Log user corrections for training data
├── classification/          # 🎯 MVP FOCUS - Document type detection
│   ├── classifier.py        # Route documents to appropriate handlers
│   ├── features.py          # Extract document type features
│   └── taxonomy.py          # Document type definitions (linear, procedural, technical, etc.)
├── training/                # 🚀 FUTURE - Fine-tuning pipeline
│   ├── annotation.py        # Lightweight correction UI (<60s per correction)
│   ├── dataset.py           # Training data management
│   ├── synthetic.py         # Generate synthetic variations from corrections
│   └── layoutlm_finetune.py # Fine-tune LayoutLMv3 on correction data
├── hierarchy/               # 🌟 CORE - Document structure analysis
│   ├── tree.py              # HierarchyTree, HierarchyNode classes
│   ├── builder.py           # Build tree from blocks (any source)
│   └── analyzer.py          # Analyze structure quality, detect issues
├── chunking/                # 🌟 CORE - Intelligent chunk strategies
│   ├── base.py              # BaseChunker + ChunkerRegistry
│   ├── hierarchical.py      # Section-based chunking (RECOMMENDED)
│   ├── semantic.py          # Embedding-based semantic chunking
│   ├── fixed.py             # Token-based (baseline for comparison)
│   └── custom.py            # User-defined rules
├── comparison/              # 🌟 CORE - Compare chunking strategies
│   ├── comparer.py          # Side-by-side strategy comparison
│   └── metrics.py           # Quality metrics for chunks
├── testing/                 # 🌟 KILLER FEATURE - Test retrieval
│   ├── embedder.py          # sentence-transformers wrapper
│   ├── searcher.py          # RetrievalTester for query testing
│   └── evaluator.py         # Evaluate retrieval quality
├── core/
│   └── document.py          # Block, Chunk, HierarchyNode dataclasses
├── extraction/              # 📦 COMMODITY - Gets blocks from files
│   ├── strategy.py          # Extraction tier selection (fast/enhanced/ai)
│   ├── fast_extractor.py    # PyMuPDF + pdfplumber (Tier 1)
│   ├── docling_extractor.py # IBM Docling (Tier 2) - BEST FOR STRUCTURE
│   └── layoutparser_extractor.py  # LayoutParser (Tier 3)
├── loaders/                 # 📦 COMMODITY - Legacy loader system
│   ├── pdf.py               # pdfplumber-based PDF loader
│   ├── docx.py              # python-docx loader
│   ├── markdown.py          # markdown-it-py loader
│   └── text.py              # Plain text loader
├── exporters/
│   ├── jsonl.py             # JSONL export (LangChain/LlamaIndex)
│   ├── json_export.py       # Full JSON with metadata
│   ├── nested_json.py       # Nested hierarchy JSON
│   └── csv_export.py        # CSV export
├── utils/
│   ├── tokens.py            # tiktoken-based token counting
│   └── quality.py           # QualityAnalyzer for chunk scoring
└── server.py                # FastAPI server (port 8420)
```

**Key Insight:** Extraction is **input**. Diagnostics is **the product**. Chunking engine is the **graduation path**.

### Frontend (Electron + React + TypeScript + Tailwind)

**Priority Components (Diagnostic-First):**

```
ui/src/components/
├── DiagnosticDashboard/     # 🎯 MVP FOCUS - Main diagnostic interface
│   ├── ProblemList.tsx      # List of detected chunk problems
│   ├── ProblemCard.tsx      # Individual problem with severity, type, fix suggestions
│   ├── ChunkOverlay.tsx     # Visual overlay of chunk boundaries on source PDF
│   └── FailureTimeline.tsx  # Show query → failed retrieval → problem diagnosis
├── QueryTracer/             # 🎯 MVP KILLER FEATURE - Query-to-failure analysis
│   ├── QueryInput.tsx       # User enters failed query
│   ├── ExpectedResults.tsx  # What should have been retrieved
│   ├── ActualResults.tsx    # What was actually retrieved
│   ├── FragmentationView.tsx # Show how chunking broke the relevant content
│   └── FixSuggestions.tsx   # Actionable fixes for the problem
├── CorrectionCapture/       # 🎯 TRAINING DATA - Lightweight correction UI
│   ├── AnnotationPanel.tsx  # <60 second correction interface
│   ├── ProblemTypeSelector.tsx # Tag problem type (semantic, structural, etc.)
│   ├── SuggestedFix.tsx     # AI-suggested fix (user can accept/modify)
│   └── TrainingQueue.tsx    # Show corrections queued for model training
├── DocumentClassifier/      # 🎯 MVP FOCUS - Document type detection
│   ├── TypeBadge.tsx        # Show detected document type
│   ├── ConfidenceScore.tsx  # Classifier confidence
│   └── ManualOverride.tsx   # User can override classification
├── HierarchyTree/           # 🌟 CORE - Visual document structure
│   ├── TreeView.tsx         # Collapsible tree navigation
│   ├── TreeNode.tsx         # Individual section node
│   ├── TreeStats.tsx        # Structure statistics
│   └── ProblemAnnotations.tsx # Show diagnostic problems in tree context
├── ChunkPreview/            # 🌟 Live chunk preview panel
│   ├── ChunkCard.tsx        # Individual chunk display
│   ├── ChunkEditor.tsx      # Edit/split/merge chunks
│   ├── QualityBadge.tsx     # Show quality scores
│   └── DiagnosticBadge.tsx  # Show detected problems (NEW)
├── StrategySelector/        # 🌟 Choose chunking approach
│   ├── StrategyPicker.tsx   # Radio selector for strategies
│   ├── ParameterPanel.tsx   # Configure strategy parameters
│   └── PreviewButton.tsx    # "Preview chunks" button
├── QueryTester/             # 🌟 Test retrieval (integrated with diagnostics)
│   ├── QueryInput.tsx       # Enter test queries
│   ├── ResultsList.tsx      # Show retrieved chunks
│   └── ComparisonView.tsx   # Compare strategies side-by-side
├── ComparisonDashboard/     # 🌟 Strategy comparison UI
│   ├── SideBySide.tsx       # Show two strategies
│   ├── MetricsPanel.tsx     # Quality metrics
│   ├── DiagnosticMetrics.tsx # Problem detection rates (NEW)
│   └── RecommendationBox.tsx # Suggest best strategy
├── DocumentViewer/          # Document/chunk preview
├── Toolbar.tsx              # Top toolbar with actions
└── Sidebar.tsx              # Document list
```

## Development Commands

```bash
# Install Python dependencies
pip install -e .

# Optional: Install enhanced extraction (Docling - RECOMMENDED)
pip install chonk[enhanced]

# Run backend (required before UI)
cd src && uvicorn chonk.server:app --reload --port 8420

# Install UI dependencies
cd ui && npm install

# Run UI in development
cd ui && npm run dev

# Run Electron + UI together
cd ui && npm run electron:dev
```

## Key Concepts

### Document Model Hierarchy

1. **Document** - The source file (PDF, DOCX, etc.)
2. **Blocks** - Semantic units extracted from document
   - Heading blocks (with levels 1-6)
   - Text blocks (paragraphs)
   - Table blocks
   - List blocks
   - Code blocks
3. **Hierarchy Tree** - Structure of document
   - Built from heading blocks
   - Parent-child relationships
   - Section IDs and paths
4. **Chunks** - Groups of blocks for embedding
   - Can be flat (token-based) or hierarchical (section-based)
   - Include quality scores
   - Have hierarchy paths for context
5. **Test Queries** - Sample questions to validate retrieval
6. **Project** - Container for documents, chunks, test suites (.chonk files)

### Chunking Strategies

**Hierarchical (RECOMMENDED):**
- Respects document structure
- Creates one chunk per section
- Preserves semantic boundaries
- Includes hierarchy paths for context
- Example: "Section 2.1.1" becomes one chunk with path "Section 2 > 2.1 General > 2.1.1 Safety"

**Fixed (BASELINE):**
- Token-based sliding window
- Good for comparison only
- Destroys structure
- Use to show why hierarchical is better

**Semantic (ADVANCED):**
- Embedding-based similarity
- Groups semantically related blocks
- Expensive (requires embeddings)
- Use for unstructured documents

**Custom:**
- User-defined rules
- Advanced users only

### Hierarchy Tree Structure

The **HierarchyNode** is the core data structure:

```python
class HierarchyNode:
    section_id: str              # "2.1.1" or "FOREWORD"
    heading: str                 # Heading text (separated)
    heading_block_id: str        # Link to source block
    content: str                 # Body text (without heading)
    content_block_ids: list[str] # Links to source blocks
    level: int                   # Heading level (1-6)
    token_count: int             # Total tokens
    page_range: list[int]        # [start_page, end_page]
    parent: HierarchyNode | None # Parent section
    children: list[HierarchyNode] # Subsections
```

**Key Features:**
- Heading and content **separated**
- Full parent-child tree structure
- Traceability to source blocks
- Flexible for different chunking strategies

### Quality Scores

Chunks have multi-dimensional quality scores:

```python
class QualityScore:
    token_range: float           # 0-1, is size optimal?
    sentence_complete: float     # 0-1, proper boundaries?
    hierarchy_preserved: float   # 0-1, no orphan headings?
    table_integrity: float       # 0-1, tables not split?
    reference_complete: float    # 0-1, no orphan references?
    overall: float               # Weighted average
```

Use these to identify problem chunks before embedding.

## API Endpoints (port 8420)

### Diagnostics (MVP FOCUS - Ship First)
- `POST /api/diagnostics/analyze` - Analyze chunks for problems (all 4 categories)
- `GET /api/diagnostics/{doc_id}/problems` - Get detected problems for document
- `POST /api/diagnostics/trace-query` - 🌟 KILLER: Trace query to failure cause
- `POST /api/diagnostics/visualize` - Generate chunk boundary overlay on source PDF
- `GET /api/diagnostics/problems/{problem_id}` - Get detailed problem analysis
- `POST /api/diagnostics/suggest-fix` - Get AI-suggested fix for problem

### Classification (MVP FOCUS)
- `POST /api/classify/document` - Classify document type (linear, procedural, technical, etc.)
- `GET /api/classify/types` - List supported document types with complexity levels
- `PUT /api/classify/{doc_id}/override` - Manual classification override

### Corrections (Training Data Capture)
- `POST /api/corrections` - Log user correction (lightweight, <60s interface)
- `GET /api/corrections/{doc_id}` - Get corrections for document
- `POST /api/corrections/batch-export` - Export corrections for training
- `GET /api/training/dataset-stats` - Get training data statistics

### Hierarchy (Core Feature)
- `POST /api/hierarchy/build` - Build hierarchy tree from document
- `GET /api/hierarchy/{doc_id}` - Get hierarchy tree
- `GET /api/hierarchy/{doc_id}/stats` - Get structure statistics
- `POST /api/hierarchy/validate` - Check for structure issues

### Chunking (Future - Ship Second)
- `POST /api/chunk/preview` - Preview chunks with strategy (don't save)
- `POST /api/chunk/apply` - Apply chunking strategy and save
- `POST /api/chunk/compare` - Compare multiple strategies side-by-side
- `GET /api/chunk/strategies` - List available strategies

### Testing (Integrated with Diagnostics)
- `POST /api/test/search` - Search chunks with query
- `POST /api/test/compare-strategies` - Compare retrieval quality across strategies
- `POST /api/test-suites` - Create test suite
- `POST /api/test-suites/{id}/run` - Run test suite
- `GET /api/test/recommendations` - Get strategy recommendations based on tests

### Documents (Simplified)
- `POST /api/documents/upload` - Upload and extract blocks
- `GET /api/documents/{id}` - Get document with hierarchy
- `DELETE /api/documents/{id}` - Remove document

### Chunks (Manual Editing)
- `POST /api/chunks/merge` - Merge selected chunks
- `POST /api/chunks/split` - Split chunk at position
- `PUT /api/chunks/{id}` - Update chunk metadata (tags, notes, lock)

### Export
- `POST /api/export` - Export chunks (jsonl, json, csv, nested-json)
- `POST /api/export/diagnostics` - Export diagnostic report

## User Workflow

### Diagnostic-First Workflow (MVP - Ship This)

**Path A: Diagnose Existing Chunks (Primary Use Case)**

```
1. UPLOAD EXISTING CHUNKS + SOURCE PDF
   ↓
2. RUN DIAGNOSTICS (automatic, 4 problem categories)
   ↓
3. SEE PROBLEMS (visual overlay on PDF, severity scores)
   ↓
4. TRACE QUERY FAILURES 🌟 KILLER FEATURE
   - Enter query that returned bad results
   - See what SHOULD have been retrieved
   - See HOW chunking fragmented the content
   ↓
5. REVIEW SUGGESTED FIXES (AI-generated, one-click apply)
   ↓
6. ANNOTATE CORRECTIONS (lightweight UI, <60 seconds)
   - Tag problem type
   - Accept/modify suggested fix
   - Contributes to training data
   ↓
7. EXPORT DIAGNOSTIC REPORT + FIXED CHUNKS
```

**Path B: Create Better Chunks (Future Product - Ship Second)**

```
1. DROP PDF
   ↓
2. CLASSIFY DOCUMENT (automatic: linear, procedural, technical, etc.)
   ↓
3. EXTRACT BLOCKS (tier selected based on document type)
   ↓
4. BUILD HIERARCHY (automatic, visualize as tree)
   ↓
5. CHOOSE STRATEGY (recommended based on document type)
   ↓
6. PREVIEW CHUNKS (see what you'll get, run diagnostics)
   ↓
7. TEST QUERIES (does retrieval work? compare strategies)
   ↓
8. REFINE (merge/split/annotate problem chunks)
   ↓
9. EXPORT (JSONL, JSON, nested JSON) + RUN DIAGNOSTICS
```

### Key Insight: The "Aha Moment"

**Diagnostic-first creates immediate value:**
1. User has existing RAG system with bad retrieval
2. User uploads chunks + source PDF to CHONK
3. **Diagnostic shows exact problems** (semantic fragmentation, broken references, etc.)
4. User enters failed query → **CHONK traces failure to specific chunking problem**
5. User sees visual overlay of chunk boundaries destroying semantic units
6. **"Aha! THAT'S why retrieval fails!"**
7. User becomes paid customer to fix their chunks

**This is the wedge that makes users pay you while teaching your system.**

### The Diagnostic Flywheel

```
User diagnoses problem → User annotates fix → CHONK captures training data
                                  ↓
                        Fine-tune chunking engine
                                  ↓
                    Better chunking recommendations
                                  ↓
                        More users succeed
                                  ↓
                     More correction data
```

Every correction improves the product. You get paid to collect the data.

## Code Style

- **Python**: Black + Ruff formatting, type hints required, dataclasses for models
- **TypeScript**: Strict mode, functional components, Zustand for state
- **Theme**: Retro 8-bit pixel art aesthetic with custom Tailwind palette

## Design Principles

### 1. Diagnosis Before Creation
Show users what's wrong with existing chunks before offering to create better ones. The "aha moment" comes from seeing problems, not promises.

### 2. Visual Clarity
Users should **see** their problems, not read about them. Chunk boundary overlays, fragmentation visualization, color-coded severity.

### 3. Actionable Insights
Every detected problem must have a suggested fix. Never leave users with "you have a problem" - always provide "here's how to fix it."

### 4. Training Data as Product
Every user interaction is potential training data. Design workflows to capture corrections in <60 seconds. The flywheel is the moat.

### 5. Security by Default
Assume enterprise use from day one. Local-first, encrypted at rest, audit logging, SSO-ready. No bolting-on security later.

### 6. Progressive Disclosure
- Free tier: Show diagnostic value (10 reports/month)
- Starter tier: Unlimited diagnostics + clean exports
- Team tier: Collaboration + advanced features
- Enterprise tier: SSO + audit logs + SLA

Each tier is a natural upgrade when users hit limits, not arbitrary gates.

### 7. Fail Visually
Don't just say "chunking failed" - show WHERE it failed, WHY it failed, and HOW to fix it. Screenshots > error messages.

### 8. Type-Aware Intelligence
Different documents need different approaches. Linear prose ≠ procedural spec ≠ technical drawing. Classify first, then apply appropriate diagnostics.

## What Makes CHONK Different

### Not a Chunker - A Diagnostic System

**Commodity Tools (LangChain, LlamaIndex, unstructured.io):**
- "Here's 500 chunks, good luck!"
- Black box - retrieval fails, no idea why
- No visualization of problems
- No path to fixing issues
- Users abandon when retrieval doesn't work

**CHONK Diagnostic Tool (Ship First):**
- "Your query 'safety requirements' failed because chunks 47-52 fragmented the safety section"
- "Here's the visual overlay showing where chunking broke semantic boundaries"
- "Problem: Semantic Incompleteness (severity: high) - chunk contains partial idea with dangling reference"
- "Suggested fix: Merge chunks 47-52 into single semantic unit"
- **Users see WHY retrieval fails and HOW to fix it**

**CHONK Chunking Engine (Ship Second):**
- "Your document is classified as 'Structured Reference, Medium Complexity'"
- "Recommended strategy: Table-aware hierarchical chunking"
- "Running diagnostics on preview chunks... 3 problems detected, click to fix"
- **Users graduate to this after seeing diagnostic value**

### The Diagnostic Difference

**Other tools:** "Your chunks are ready!" → (retrieval fails) → (user churns)

**CHONK:** "Your chunks have 47 problems, here's how to fix them" → (user sees value) → (user pays) → (retrieval works) → (user stays)

### The Training Data Moat

**Every diagnostic session generates training data:**
- User identifies problem → Tags problem type → Accepts/modifies fix
- Correction captured in <60 seconds
- Fine-tune LayoutLMv3 on real-world corrections
- Chunking engine gets smarter with every user

**Competitors can't replicate this** because they don't have the diagnostic workflow that generates corrections at scale.

## Technical Foundation

### Fine-Tunable Models (Commercial Licensing)

**LayoutLMv3 (MIT License) - PRIMARY MODEL**
- Layout-aware document understanding
- Best for structure/hierarchy detection
- Handles multi-column layouts, tables, reading order
- Fine-tune on correction data from diagnostic tool
- GPU-friendly, good inference speed

**Donut (MIT License) - SECONDARY MODEL**
- End-to-end document understanding (no OCR dependency)
- Good for drawings/forms/templates
- Visual-only approach
- Fine-tune for document type classification

**PaddleOCR/PPStructure (Apache 2.0) - SUPPORTING LIBRARY**
- Layout analysis and table recognition
- Open-source, production-ready
- Good baseline for table integrity detection

**Training Data Provenance (Critical for Commercial Use):**
- Use **public domain documents** (MIL-STD, government specs, academic papers) for training
- Use **proprietary documents** only for validation
- Clear licensing ensures enterprise sales viability

### Diagnostic Detection Methods

Each problem category has specific detection heuristics:

**1. Semantic Incompleteness:**
- Sentence boundary detection (incomplete sentences at chunk edges)
- Dangling connectives ("however", "therefore", "additionally" at chunk start)
- Embedding shift analysis (add neighboring sentences, measure similarity change)
- Orphaned references ("see above", "as mentioned", "following" with no target)

**2. Semantic Contamination:**
- Sentence-level embedding clustering (multiple distinct clusters = contamination)
- Topic coherence scoring (LDA or BERTopic on chunk sentences)
- Abrupt topic transitions (cosine similarity drop >0.3 between adjacent sentences)

**3. Structural Breakage:**
- List/procedure pattern matching (numbered steps, bullet points split across chunks)
- Table detection (table rows/columns split)
- Orphaned headers (heading at chunk end with no content)
- Code block splits (syntax-aware detection)

**4. Reference Orphaning:**
- Regex patterns: "see above", "as follows", "in section X", "table Y"
- Forward/backward reference tracking (cross-chunk reference validation)
- Figure/table caption orphaning (caption without content or vice versa)

## Training Data Strategy

### Annotation-as-You-Go (Solo Dev Approach)

**Instead of:** "Annotate 10,000 documents before shipping"
**Do this:** "Log corrections during real diagnostic work"

**Implementation:**
1. **Lightweight Correction UI** (<60 seconds per correction)
   - User sees diagnostic problem
   - Clicks "Suggest Fix"
   - Reviews AI suggestion
   - Accepts or modifies in 2-3 clicks
   - Correction logged automatically

2. **Prioritize High-Value Documents**
   - Recurring structures (same document type multiple times)
   - Systematic failures (same problem type repeatedly)
   - User complaints (what problems do paying customers hit?)

3. **Synthetic Data Multiplication**
   - Generate variations from real corrections
   - Augment with layout transformations (font size, spacing, columns)
   - Target: 10x multiplier (1 manual correction → 10 training examples)

4. **Implicit Training Signals**
   - Track which suggested fixes users accept vs. modify
   - Log which diagnostic problems users ignore vs. fix
   - Measure retrieval improvement after fixes (reinforcement signal)

**Target:** 200-500 annotated pages for meaningful fine-tuning results on specific document types.

**Achievable Timeline:**
- Week 1-4: Ship diagnostic MVP, start logging corrections
- Week 5-12: Accumulate 50-100 corrections from early users
- Week 13-16: First fine-tuning experiment, measure improvement
- Week 17+: Continuous improvement loop

## Commercial Positioning

### Value Proposition (Refined)

**NOT THIS:** "Better chunking for all documents" (too broad, commoditizing)

**THIS:** "Understand why your RAG system fails and fix it" (immediate pain, clear value)

### Market Segments

| Segment | Price | Volume | Churn | Characteristics |
|---------|-------|--------|-------|-----------------|
| **Individual Devs** | $50/mo | High | High | Self-service, credit card, need immediate value |
| **Startup Teams** | $300/mo | Moderate | Moderate | Small team collaboration, Slack integration |
| **Enterprise** | $2k/mo | Lower | Low | SSO, audit logs, SLA, support |
| **Vertical-Specific** | $5-10k/mo | Smallest | Lowest | Defense, legal, medical - domain expertise |

### Pricing Strategy

**Freemium Model:**
- Free: 10 diagnostic reports/month, watermarked exports
- Starter ($50/mo): 100 reports/month, clean exports, basic support
- Team ($300/mo): Unlimited reports, collaboration, priority support
- Enterprise ($2k+/mo): SSO, audit logs, custom training, SLA

**Value Metric:** Diagnostic reports (not documents, not chunks)
- Aligns with value delivery (each report shows problems + fixes)
- Encourages usage (iterate until retrieval works)
- Natural upgrade path (need more reports = getting value)

### Realistic Business Outcomes

**NOT:** Venture-scale hypergrowth ($100M+ ARR)

**YES:** Capital-efficient sustainable business ($1-5M ARR)
- Solo dev or tiny team (2-3 people max)
- High margins (software, minimal infrastructure)
- Deep domain expertise becomes moat
- Profitable from year 1-2

## Extraction Tiers (Supporting Feature)

CHONK supports multiple extraction backends:

**Tier 1 (Fast):** PyMuPDF + pdfplumber
- Built-in, no extra deps
- Good for simple documents
- Basic heading detection

**Tier 2 (Enhanced):** IBM Docling - **RECOMMENDED**
- `pip install chonk[enhanced]`
- GPU-accelerated (CUDA)
- Excellent structure detection (section headers, tables, lists)
- Best hierarchy quality

**Tier 3 (AI):** LayoutParser
- `pip install chonk[ai]`
- Deep learning layout analysis
- For complex/scanned documents

**Key Point:** Extraction quality affects hierarchy quality. Docling is recommended for best results.

## Example: Why Hierarchy Matters

### Flat Chunking (Bad)
```
Chunk 1: "...end of section 1.2. 2.1 Safety Requirements Safety is critical..."
Chunk 2: "...procedures must follow. 2.2 Performance The system shall..."
```
❌ Sections mixed together
❌ No context about where chunks came from
❌ Retrieval returns partial, mixed content

### Hierarchical Chunking (Good)
```
Chunk 1:
  Section: "2.1 Safety Requirements"
  Path: "Section 2 Requirements > 2.1 Safety Requirements"
  Content: "Safety is critical. All procedures must..."

Chunk 2:
  Section: "2.2 Performance"
  Path: "Section 2 Requirements > 2.2 Performance"
  Content: "The system shall meet the following..."
```
✅ Complete sections
✅ Context preserved (hierarchy path)
✅ Retrieval returns exact, relevant sections

## MVP Roadmap (Diagnostic-First)

### Phase 1: Diagnostic MVP (Weeks 1-2)

**Goal:** Ship simple web UI that creates the "aha moment"

**Deliverables:**
- [ ] Upload PDF + chunk boundaries (JSON format: `[{text, start_pos, end_pos}]`)
- [ ] Visualize chunks overlaid on source PDF (highlight boundaries)
- [ ] Manual problem annotation UI (<60 seconds per problem)
  - Click chunk → Select problem type → Add note → Save
- [ ] Export annotated problems as JSON

**Success Criteria:**
- User can upload existing chunks and see them on the PDF
- User can manually tag 5-10 problems in under 10 minutes
- Visual overlay clearly shows chunk boundary issues

### Phase 2: Automated Detection (Weeks 3-4)

**Goal:** Automate the four problem categories

**Deliverables:**
- [ ] Semantic incompleteness detector
  - Sentence fragment detection (incomplete sentences at edges)
  - Dangling connective detection (regex patterns)
  - Orphaned reference detection ("see above", "as follows")
- [ ] Semantic contamination detector
  - Sentence embedding clustering (scikit-learn)
  - Abrupt topic transition detection (cosine similarity drops)
- [ ] Structural breakage detector
  - List/table pattern matching (regex)
  - Orphaned header detection
- [ ] Reference orphaning detector
  - Cross-reference pattern matching
  - Figure/table caption validation
- [ ] Problem severity scoring (high/medium/low)
- [ ] Automated diagnostic report generation

**Success Criteria:**
- System detects 70%+ of manually-tagged problems
- False positive rate <30%
- Diagnostic report shows all 4 problem categories

### Phase 3: Query-Aware Diagnostics (Weeks 5-6) 🌟 KILLER FEATURE

**Goal:** Trace failed queries to chunking problems

**Deliverables:**
- [ ] Query input interface
- [ ] Expected result identification (user highlights what should have been retrieved)
- [ ] Actual result comparison (show what was retrieved)
- [ ] Fragmentation visualization
  - Show which chunks contain parts of the expected result
  - Highlight how chunking broke the semantic unit
  - Color-code severity (red = critical fragmentation)
- [ ] Fix suggestions
  - "Merge chunks X, Y, Z to preserve semantic unit"
  - "Split chunk A at position P to separate topics"
  - One-click apply suggestions

**Success Criteria:**
- User enters failed query → System shows exact chunking problem
- Visual overlay makes fragmentation obvious
- Suggested fixes are actionable and correct 60%+ of the time

### Phase 4: Training Data Pipeline (Weeks 7-8)

**Goal:** Capture corrections for model fine-tuning

**Deliverables:**
- [ ] Correction logging system
  - Auto-save accepted/modified fixes
  - Tag problem type and severity
  - Store before/after chunk boundaries
- [ ] Training data export
  - JSON format for LayoutLMv3 fine-tuning
  - Synthetic data generation (10x multiplier)
- [ ] Dataset statistics dashboard
  - Show correction count by problem type
  - Show document type distribution
  - Show model improvement metrics (if trained)

**Success Criteria:**
- Each correction logged in <60 seconds
- Export format compatible with LayoutLMv3 training
- 50+ corrections accumulated in first 2 weeks of usage

### Phase 5: Document Classification (Weeks 9-10)

**Goal:** Route documents to appropriate diagnostic handlers

**Deliverables:**
- [ ] Document type classifier
  - Linear prose vs. structured reference vs. procedural vs. technical
  - Confidence scoring
- [ ] Type-specific diagnostic rules
  - Different detection thresholds per document type
  - Custom problem patterns (e.g., step sequence for procedural docs)
- [ ] Manual override UI
  - User can correct misclassification
  - Corrections logged as training data

**Success Criteria:**
- Classifier accuracy 70%+ on common document types
- Type-specific diagnostics improve detection rate by 15%+

## Current Status (Post-Update)

**Implemented (Chunking Engine - Ship Second):**
- ✅ Block extraction (Tier 1, 2, 3)
- ✅ Hierarchy tree building
- ✅ Hierarchical chunking
- ✅ Quality scoring
- ✅ Nested JSON export
- ✅ Basic retrieval testing

**MVP Focus (Diagnostic Tool - Ship First):**
- 🎯 Phase 1: Diagnostic MVP (Weeks 1-2) - **START HERE**
- 🎯 Phase 2: Automated Detection (Weeks 3-4)
- 🎯 Phase 3: Query-Aware Diagnostics (Weeks 5-6) - **KILLER FEATURE**
- 🎯 Phase 4: Training Data Pipeline (Weeks 7-8)
- 🎯 Phase 5: Document Classification (Weeks 9-10)

**Future (Chunking Engine):**
- 📋 Fine-tuned LayoutLMv3 for structure detection
- 📋 Type-aware chunking strategies
- 📋 Automated fix application
- 📋 Real-time diagnostic preview during chunking
- 📋 Continuous improvement loop (diagnostics → corrections → fine-tuning → better chunking)

## Security & Enterprise Readiness

### Security-First Architecture

CHONK is designed for enterprise deployment from day one. All security considerations are built-in, not bolted-on.

### 1. Data Privacy & Local-First

**Principle:** User documents never leave the user's machine unless explicitly authorized.

**Implementation:**
- **Local processing:** All extraction, chunking, and diagnostics run locally
- **No telemetry:** No automatic data transmission to external servers
- **Opt-in cloud features:** Embedding models, training data export require explicit consent
- **Data residency:** Users choose where corrections/annotations are stored (local filesystem by default)
- **Encryption at rest:** AES-256 encryption for all stored projects (.chonk files)

**Enterprise Features:**
- Self-hosted deployment option (Docker container)
- Air-gapped mode (no internet connectivity required)
- GDPR/HIPAA compliance documentation

### 2. Input Validation & File Security

**Threat Model:** Malicious PDFs, injection attacks, resource exhaustion

**Protections:**
- **File type validation:** Strict MIME type checking, magic number verification
- **Size limits:** Configurable max file size (default 100MB, enterprise can adjust)
- **Sandbox extraction:** PDF parsing runs in isolated process with resource limits
- **Memory limits:** Hard caps on extraction memory (configurable per document type)
- **Timeout protection:** Kill long-running extractions (30s default, configurable)
- **Path traversal prevention:** All file operations use safe path joining, no user-controlled paths
- **ZIP bomb protection:** Compressed file extraction size limits

**Input Sanitization:**
- All text content sanitized before rendering (prevent XSS in UI)
- Regex patterns validated before compilation (prevent ReDoS attacks)
- SQL parameterization for all database queries (if using DB backend)
- Command injection prevention (no shell commands with user input)

### 3. API Security

**Authentication:**
- **Free/Starter tier:** API key authentication (generated client-side, stored in keychain)
- **Team tier:** OAuth 2.0 with JWT tokens
- **Enterprise tier:** SSO (SAML 2.0, OIDC), Active Directory integration

**Authorization:**
- Role-based access control (RBAC) for team/enterprise tiers
  - Admin: Full access, manage users, view audit logs
  - Editor: Upload documents, run diagnostics, export results
  - Viewer: Read-only access to diagnostic reports
- Per-document access control (share specific documents with specific users)
- API rate limiting (prevent abuse)
  - Free: 10 requests/minute
  - Starter: 100 requests/minute
  - Team/Enterprise: Configurable

**Security Headers:**
- HTTPS-only (TLS 1.3 minimum)
- CORS restricted to allowed origins
- Content-Security-Policy header (prevent XSS)
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- Strict-Transport-Security header (HSTS)

### 4. Dependency Security

**Supply Chain Protection:**
- **Dependency pinning:** Exact versions in requirements.txt, package-lock.json
- **Automated scanning:** Dependabot, Snyk for vulnerability detection
- **Regular updates:** Monthly security patch cycle
- **Minimal dependencies:** Audit and remove unnecessary libraries
- **License compliance:** Automated license checking (only MIT, Apache 2.0, BSD)

**Python Dependencies (Core):**
- `fastapi` - API framework
- `pydantic` - Input validation
- `sentence-transformers` - Embeddings (local model, no API calls)
- `transformers` - LayoutLMv3 (local, no cloud)
- `scikit-learn` - Clustering for contamination detection
- `tiktoken` - Token counting

**Known Risk Dependencies (Monitored):**
- `PyMuPDF`, `pdfplumber` - PDF parsing (potential vulnerabilities in native code)
  - Mitigation: Sandbox process, resource limits, regular updates

### 5. Audit Logging (Enterprise Tier)

**What's Logged:**
- Document uploads (filename hash, timestamp, user)
- Diagnostic runs (document ID, problem count, duration)
- Query traces (query hash, results, user)
- Corrections (problem type, fix applied, user)
- Export events (format, destination, user)
- Authentication events (login, logout, failed attempts)
- API requests (endpoint, user, timestamp, response code)

**What's NOT Logged:**
- Actual document content (only hashes)
- Actual query text (only hashes)
- User annotations (unless explicitly exported for training)

**Log Storage:**
- Local filesystem (JSON lines format)
- Optional: Send to SIEM (Splunk, Datadog, etc.)
- Tamper-evident (append-only, cryptographic hashing)
- Retention policy: 90 days default, configurable

### 6. Secrets Management

**No Hardcoded Secrets:**
- All API keys, passwords stored in environment variables or secret manager
- `.env` files never committed to git (`.gitignore` enforced)
- Secrets rotation support (no downtime when rotating keys)

**Enterprise Secrets Management:**
- Integration with HashiCorp Vault, AWS Secrets Manager, Azure Key Vault
- Automatic secret rotation for API keys
- Audit trail for secret access

### 7. Secure Development Practices

**Code Review:**
- All code changes reviewed by at least one other developer
- Security-focused review checklist (input validation, auth, logging)
- Automated static analysis (Bandit for Python, ESLint security plugin for TypeScript)

**Testing:**
- Security test suite (OWASP Top 10 coverage)
- Fuzzing for file parsers (PDF, DOCX)
- Penetration testing before major releases (annual for enterprise tier)

**Deployment:**
- Signed releases (GPG signatures for downloads)
- Integrity checks (SHA-256 hashes published)
- Automated deployment pipeline with security gates

### 8. Incident Response

**Plan:**
1. **Detection:** Monitoring alerts, user reports, automated scans
2. **Assessment:** Severity classification (critical/high/medium/low)
3. **Containment:** Disable affected features, patch deployment
4. **Communication:** Security advisory to users (email, website banner)
5. **Recovery:** Verify fix, post-mortem, update security docs

**SLA (Enterprise Tier):**
- Critical vulnerabilities: Patch within 24 hours
- High vulnerabilities: Patch within 7 days
- Medium vulnerabilities: Patch within 30 days

### 9. Enterprise Feature Checklist

**Must-Have for Enterprise Sales:**
- [x] SSO (SAML 2.0, OIDC)
- [x] RBAC (Role-Based Access Control)
- [x] Audit logging (tamper-evident, exportable)
- [x] Data encryption at rest (AES-256)
- [x] Self-hosted deployment (Docker)
- [x] Air-gapped mode (no internet required)
- [x] GDPR compliance documentation
- [x] HIPAA compliance documentation (for medical document use case)
- [x] SOC 2 Type II certification (future - year 2)
- [x] Security questionnaire responses (standard vendor assessment)
- [x] Penetration test reports (annual, third-party)

**Nice-to-Have:**
- [ ] Multi-tenancy (isolated data per organization)
- [ ] Data residency controls (choose geographic region)
- [ ] DLP integration (prevent sensitive data export)
- [ ] Advanced threat protection (integrate with CrowdStrike, SentinelOne)

### 10. Training Data Security

**Special Considerations:** Training data capture is a unique security risk.

**Protections:**
- **Explicit consent:** Users opt-in to training data collection
- **PII redaction:** Automatic detection and redaction of sensitive data
  - Names, email addresses, phone numbers, SSNs, credit cards
  - Custom regex patterns per industry (medical record numbers, case numbers)
- **Anonymization:** Document metadata stripped (author, creation date, file path)
- **Watermarking:** Training data includes invisible watermarks (detect leaks)
- **Access control:** Only ML team has access to raw training data
- **Retention limits:** Training data deleted after model training (6 months max)
- **Opt-out:** Users can request deletion of contributed training data

**Compliance:**
- GDPR Article 6 (lawful basis for processing)
- GDPR Article 17 (right to erasure)
- CCPA (California Consumer Privacy Act)
- Clear privacy policy explaining training data use

## For Developers

When working on CHONK, remember:

**Priority 1:** Diagnostic workflow (query tracing, problem detection, visual overlay)
**Priority 2:** Training data capture (corrections, annotations, export pipeline)
**Priority 3:** Security and enterprise features (SSO, audit logs, encryption)
**Priority 4:** Chunking engine (fine-tuned models, type-aware strategies)
**Priority 5:** Everything else

**Ship order:** Diagnostic tool first (create "aha moment"), chunking engine second (graduation path).

**Security mindset:** Assume enterprise use from day one. No bolting-on security later.

## Strategic Summary

### The Big Picture

**Problem:** RAG retrieval failures are invisible. Users can't diagnose why chunks fail.

**Solution:** CHONK makes chunking failures visible and fixable.

**Wedge Product:** Diagnostic tool (ship weeks 1-6)
- Upload existing chunks + source PDF
- See exact problems (4 categories, auto-detected)
- Trace failed queries to chunking issues (killer feature)
- Visual overlay makes problems obvious
- Creates "aha moment" → user pays

**Graduation Product:** Chunking engine (ship later)
- Structure-aware document processing
- Type-specific chunking strategies
- Real-time diagnostic preview
- Trained on correction data from diagnostic tool

**The Flywheel:**
```
User diagnoses problem → User fixes problem → Correction captured
                                  ↓
                        Training data accumulates
                                  ↓
                       Fine-tune LayoutLMv3
                                  ↓
                    Chunking engine improves
                                  ↓
                Better automatic suggestions
                                  ↓
            More users succeed faster
                                  ↓
        More users = more corrections
```

**The Moat:** Every diagnostic session generates training data. Competitors can't replicate this because they don't have the workflow.

**The Business:** Capital-efficient, sustainable, $1-5M ARR
- Solo dev or tiny team (2-3 people)
- Freemium SaaS ($50-$10k/month tiers)
- High margins (software, minimal infrastructure)
- Enterprise-ready from day one (security, SSO, audit logs)

**The Timeline:**
- Weeks 1-2: Diagnostic MVP (manual annotation)
- Weeks 3-4: Automated detection (4 problem categories)
- Weeks 5-6: Query tracing (killer feature)
- Weeks 7-8: Training data pipeline
- Weeks 9-10: Document classification
- Week 11+: First fine-tuning experiments

**The Key Insight:** Diagnosis creates immediate value AND generates training data for long-term value. You get paid to improve the product.

### Why This Wins

1. **Immediate Value:** Diagnostics work on existing chunks (no re-chunking required)
2. **Visual Proof:** Users SEE their problems (not just told about them)
3. **Actionable:** Every problem has a suggested fix
4. **Training Data:** Every fix improves the product
5. **Moat:** Workflow generates data competitors can't get
6. **Enterprise-Ready:** Security, SSO, audit logs from day one
7. **Capital-Efficient:** Solo dev can ship and operate
8. **Sustainable:** $1-5M ARR without venture pressure

This is not a "better chunker." This is a **diagnostic system that makes RAG failures visible and fixable**, with a chunking engine as the graduation path.
