# Kiln Architecture

**Version:** MVP (Sprints 1-11)
**Last Updated:** 2026-02-15

---

## Overview

Kiln is a unified platform comprising four integrated tools that form a closed loop from raw documents to working domain-specific AI system and back.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Quarry: Document Processing                                    │
│  PDFs → Structural Analysis → Hierarchy → Metadata → Chunks     │
│                                                                 │
│  Export: Retrieval-ready JSON chunks                            │
│                                      ↓                           │
│                          ┌───────────────────────┐              │
│                          │  Quarry Knowledge     │              │
│                          │  Base (Vector DB)     │              │
│                          └───────────────────────┘              │
│                                      ↓                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Forge: Curriculum Builder                              │   │
│  │  Discovery → Competency Mapping → Example Elicitation   │   │
│  │                                                           │   │
│  │  Export: Training/test JSONL (human-validated)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                      ↓                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Foundry: Training & Evaluation                          │   │
│  │  Base Model + LoRA Training → Competency Evaluation      │   │
│  │                                                           │   │
│  │  Output: Trained LoRA adapter (10-100MB)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                      ↓                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Hearth: Interaction Layer                               │   │
│  │  Query → LoRA + RAG Retrieval → Response + Citations     │   │
│  │                                                           │   │
│  │  Feedback Capture: Signals → Route to Quarry/Forge       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                    ↓                         ↓                   │
│           Retrieval Issues          Response Quality Issues      │
│                    ↓                         ↓                   │
│            Improve Quarry            Add Examples to Forge       │
│                    └─────────┬───────────────┘                   │
│                              ↓                                   │
│                      Continuous Improvement                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tool 1: Quarry

### Purpose
Transform complex domain documents into high-quality, metadata-enriched, retrieval-ready chunks optimized for RAG systems.

### Architecture: Three-Tier Pipeline

#### Tier 1: Structural Fingerprinting and Classification

**Input:** Raw PDF file (with embedded text layer)
**Process:**
1. **Statistical Analysis** — Extract structural features without parsing content:
   - Byte patterns and file structure
   - Formatting marker distributions
   - Whitespace rhythm and patterns
   - Character frequency profiles
   - Repetition patterns in markup
   - Structural element frequency and spacing

2. **ML Classification** — Traditional ML classifier (random forest or gradient boost):
   - Input: Statistical feature vector
   - Output: Document type + structural profile
   - Trained on: Open-source datasets + manual labels + validated documents
   - Inference time: < 1 second
   - Inspectable: Feature importances available

3. **Manual Fallback** — For novel document types:
   - User interface for document type selection
   - Structural conventions form
   - Captures new training example for classifier
   - Graceful degradation without blocking pipeline

**Output:** Structural profile (JSON):
```json
{
  "document_type": "military_technical_manual",
  "hierarchy_scheme": "decimal_numbering",
  "section_markers": ["\\d+\\.\\d+", "\\d+\\.\\d+\\.\\d+"],
  "expected_elements": ["procedure", "table", "warning", "figure"],
  "formatting_conventions": { ... }
}
```

**Why ML Instead of LLM:**
- Format detection is classification, not generation
- ML faster (milliseconds vs. seconds)
- Deterministic and inspectable (feature importances)
- No GPU required
- Content-agnostic (only structure matters)

#### Tier 2: Content Extraction

**Input:** PDF + structural profile from Tier 1
**Process:** Docling-powered layout-aware extraction
- PDF parsing with reading order detection
- Table extraction and structure preservation
- Multi-column layout handling
- Figure/caption association
- Structural profile guides extraction heuristics

**Output:** Extracted blocks with layout metadata:
```json
{
  "blocks": [
    {
      "id": "block_001",
      "type": "heading",
      "level": 1,
      "text": "3.1 Safety Requirements",
      "page": 12,
      "bbox": [100, 200, 400, 220]
    },
    {
      "id": "block_002",
      "type": "paragraph",
      "text": "All maintenance procedures...",
      "page": 12,
      "bbox": [100, 230, 400, 300]
    }
  ]
}
```

#### Tier 3: Hierarchy Construction and QA

**Input:** Extracted blocks from Tier 2
**Process:**

1. **Hierarchy Construction**
   - Build tree from heading blocks
   - Detect section numbering schemes
   - Establish parent-child relationships
   - Associate content blocks with sections

2. **Classification and Filtering**
   - Identify zero-value content:
     - Tables of contents, indices
     - Distribution statements, copyright boilerplate
     - Page headers/footers
     - Repetitive administrative text
   - Remove or flag for review

3. **Cleaning and Normalization**
   - Strip repetitive headers
   - Normalize whitespace
   - Consolidate continuation entries
   - Remove formatting artifacts (LaTeX remnants, etc.)

4. **Metadata Enrichment**
   - Extract metadata from formatting cues:
     - Section hierarchy (breadcrumb path)
     - Equipment system identifiers
     - Maintenance level indicators
     - Procedure type classification
     - Reference numbers
   - Domain-specific metadata derived from structural patterns

**Output:** Structured chunks with separated body + metadata:
```json
{
  "chunks": [
    {
      "id": "chunk_001",
      "body": "All maintenance procedures must follow lockout/tagout protocols...",
      "metadata": {
        "section_path": "3.1 > 3.1.2 > Safety Procedures",
        "section_id": "3.1.2",
        "page_range": [12, 13],
        "equipment_system": "hydraulic",
        "maintenance_level": "organizational",
        "procedure_type": "safety",
        "references": ["TM 9-1234", "OSHA 1910.147"]
      },
      "hierarchy": {
        "parent_id": "chunk_000",
        "children_ids": [],
        "depth": 3
      }
    }
  ]
}
```

### Metadata-Filtered Retrieval Pipeline

**Three-stage retrieval for efficiency and precision:**

#### Stage 1: Structural Pre-Filter (Deterministic)
```python
# Example: Query for hydraulic system safety procedures
metadata_filter = {
    "equipment_system": "hydraulic",
    "procedure_type": "safety",
    "maintenance_level": "organizational"
}
# Result: 90% search space reduction, zero computational cost
```

#### Stage 2: Semantic Search (Embedding Similarity)
```python
# Embed query, search only within pre-filtered chunks
query_embedding = embed("How do I safely service hydraulic lines?")
results = vector_db.search(
    query_embedding,
    filter=metadata_filter,  # Only search pre-filtered set
    top_k=10
)
# Result: High precision, low false positives
```

#### Stage 3: Validation Pass
```python
# Check retrieved chunks against expected patterns
for chunk in results:
    if "safety" not in chunk["metadata"]["section_path"].lower():
        chunk["confidence"] *= 0.5  # Downrank unexpected results
    if chunk["metadata"]["references"]:
        chunk["confidence"] *= 1.2  # Boost chunks with citations
```

### Export Format

**Portable JSON with clear separation:**
```json
{
  "document_id": "TM-9-1234",
  "document_metadata": { ... },
  "chunks": [
    {
      "chunk_id": "unique_id",
      "body": "content for embedding",
      "metadata": {
        "filterable_attributes": { ... }
      }
    }
  ],
  "export_version": "1.0",
  "quarry_version": "0.7.0"
}
```

**Vector Database Mapping:**
- ChromaDB: `metadata` → `metadata` field
- Qdrant: `metadata` → `payload` field
- Weaviate: `metadata` → properties
- Pinecone: `metadata` → `metadata` field

All major vector databases support attribute filtering. Documentation provides exact mapping examples.

---

## Tool 2: Forge

### Purpose
Guide domain experts through creating human-validated training data for fine-tuning small language models using a discipline-level curriculum methodology.

### Architecture: Four-Step Guided Process

#### Step 1: Discipline Discovery (Framework-Only Interview)

**Duration:** 45-60 minutes
**Interface:** Structured questionnaire (no LLM in MVP)
**Process:**

1. **Domain Characteristics**
   - Document types in discipline
   - Typical user roles and expertise levels
   - Information organization patterns
   - Specialized vocabulary and terminology

2. **Core Competencies**
   - Essential skills practitioners must demonstrate
   - Common question categories
   - Typical reasoning patterns
   - Quality standards for responses

3. **Characteristic Mistakes**
   - Where do novices struggle?
   - What confusions are common?
   - What safety-critical errors occur?

**Output:** Discipline Model (JSON):
```json
{
  "discipline_id": "military_maintenance",
  "characteristics": {
    "document_types": ["technical_manual", "parts_catalog", "safety_bulletin"],
    "user_roles": ["technician", "mechanic", "supervisor"],
    "reasoning_patterns": ["procedural", "diagnostic", "safety_critical"]
  },
  "vocabulary": {
    "abbreviations": ["PMCS", "TM", "WP", "NSN"],
    "technical_terms": ["torque_specification", "clearance_fit", "lockout_tagout"]
  }
}
```

#### Step 2: Competency Mapping

**Duration:** 15-20 minutes
**Interface:** Visual competency map builder
**Process:**

1. **Generate Initial Map** from discipline model
2. **Expert Validation** — Review, refine, add competencies
3. **Hierarchy** — Parent/child competencies if needed
4. **Coverage Targets** — How many examples per competency area

**Output:** Competency Map:
```json
{
  "discipline_id": "military_maintenance",
  "competencies": [
    {
      "competency_id": "procedural_comprehension",
      "name": "Procedural Comprehension",
      "description": "Understand and follow technical procedures",
      "target_examples": 50,
      "current_examples": 0,
      "children": []
    },
    {
      "competency_id": "fault_isolation",
      "name": "Fault Isolation Reasoning",
      "description": "Diagnose problems using systematic troubleshooting",
      "target_examples": 40,
      "current_examples": 0,
      "children": ["symptom_analysis", "test_procedure_selection"]
    }
  ]
}
```

#### Step 3: Example Elicitation

**Duration:** 2-4 hours across multiple sessions
**Interface:** Guided example creation form
**Process:**

1. **Competency Area Selection** — Forge recommends based on coverage gaps
2. **Question Entry** — Expert writes realistic question
3. **Ideal Answer** — Expert writes correct response
4. **Tricky Variants** — Questions where inexperience leads to errors
5. **Explanations** — Why correct answer is correct, why wrong answers are wrong
6. **Metadata Tagging** — Competency, reasoning pattern, equipment, procedure type

**Coverage Dashboard:**
```
Procedural Comprehension:     ████████████░░░░░░░░  12/50 (24%)
Fault Isolation:              ███████░░░░░░░░░░░░░   7/40 (18%)
Safety Awareness:             ████████████████████  45/45 (100%) ✓
Parts Interpretation:         ██░░░░░░░░░░░░░░░░░░   2/30 (7%)  ← ADD MORE

Forge recommends: Focus on "Parts Interpretation" next
```

**Output:** Example Database (stored, not yet exported):
```json
{
  "example_id": "ex_001",
  "competency": "fault_isolation",
  "question": "Hydraulic system pressure reads 1800 PSI instead of specified 3000 PSI. What are the first three diagnostic steps?",
  "ideal_answer": "1. Check fluid level in reservoir (low fluid = low pressure)\n2. Inspect for visible leaks in lines and fittings\n3. Test pump output pressure with gauge at pump outlet\n\nRefer to TM 9-1234, WP 0015 for detailed procedure.",
  "tricky_variant": {
    "question": "Same scenario. Can I just replace the pump?",
    "wrong_answer": "Yes, low pressure usually means pump failure",
    "correct_answer": "No, always diagnose systematically first. Low pressure has 6 common causes, pump failure is only one. Replacing pump without diagnosis wastes time and parts.",
    "explanation": "Premature part replacement is a common novice mistake"
  },
  "metadata": {
    "equipment": "hydraulic_system",
    "reasoning_pattern": "diagnostic",
    "reference_tm": "TM-9-1234",
    "difficulty": "intermediate"
  },
  "contributor": "SME_001",
  "created": "2026-02-20T10:30:00Z"
}
```

#### Step 4: Quality Scaffolding

**Continuous process during Step 3**
**Automated checks:**

1. **Consistency Checking**
   - Response length consistency
   - Terminology usage consistency
   - Citation format consistency
   - Flag conflicting examples

2. **Coverage Analysis**
   - Real-time competency coverage percentages
   - Identify narrow/repetitive examples
   - Recommend underrepresented areas

3. **Multi-Contributor Conflict Detection**
   - Same question, different answers from different experts
   - Inconsistent explanations for same concept
   - Flagged for discipline lead review

4. **Held-Out Test Set Reservation**
   - 15-20% of examples per competency automatically reserved
   - Stratified sampling ensures representative test set
   - Challenge examples explicitly marked for evaluation

**Output:** Curriculum Export (JSONL for Foundry):
```jsonl
{"type": "train", "competency": "procedural_comprehension", "question": "...", "answer": "...", "metadata": {...}}
{"type": "train", "competency": "fault_isolation", "question": "...", "answer": "...", "metadata": {...}}
{"type": "test", "competency": "procedural_comprehension", "question": "...", "answer": "...", "metadata": {...}}
```

### Multi-Contributor Architecture

```
┌─────────────────────────────────────────┐
│  Discipline Lead                        │
│  - Owns overall discipline curriculum   │
│  - Reviews all contributions            │
│  - Resolves conflicts                   │
│  - Approves for export                  │
└────────────┬────────────────────────────┘
             │
     ┌───────┴────────┐
     ↓                ↓
┌─────────┐      ┌─────────┐
│ SME 1   │      │ SME 2   │
│ Owns:   │      │ Owns:   │
│ - Proc. │      │ - Fault │
│ - Safety│      │ - Parts │
└─────────┘      └─────────┘
```

### The Facilitator Model (Post-MVP)

**Bootstrap Strategy:**
1. MVP: Framework-only (templates, forms, dashboards) — deterministic
2. Post-MVP: Capture 5-8 discipline sessions with real experts
3. Train LoRA on human interaction data (facilitation patterns, not discipline content)
4. Integrate facilitator model as advisory (expert always confirms)

**Facilitator learns:**
- Effective questions based on domain characteristics
- Coverage gap identification
- Productive pivots when expert stalls
- Response-to-competency mapping heuristics

**Facilitator does NOT:**
- Generate training data
- Make authoritative decisions
- Bypass human validation

---

## Tool 3: Foundry

### Purpose
Manage the complete lifecycle of discipline-specific LoRA adapters: training, evaluation, versioning, regression testing, and optional model merging.

### Architecture: Training + Evaluation Pipeline

#### Training Pipeline

**Input:**
- Forge curriculum JSONL (training examples)
- Selected base model (Phi, Llama, Mistral, Qwen 3-20B)

**Process:**
```python
# Simplified user interface
foundry.train(
    curriculum="military_maintenance_v1.jsonl",
    base_model="Qwen2.5-7B-Instruct",
    # Advanced settings optional, defaults are tuned:
    # rank=16, alpha=32, learning_rate=2e-4, epochs=3
)
```

**Under the hood (Unsloth or Axolotl):**
1. Load base model
2. Configure LoRA adapter (rank, alpha, target modules)
3. Prepare training data from JSONL
4. Train with automatic mixed precision
5. Save adapter weights (10-100MB)

**Output:**
- Trained LoRA adapter (`military_maintenance_lora.safetensors`)
- Training log with loss curves
- Estimated completion time and resource usage

#### Automated Evaluation: Three Layers

**Layer 1: Competency Testing**

Run held-out test set through trained LoRA. Report per-competency accuracy in plain language.

```
=== EVALUATION RESULTS ===
Military Maintenance Discipline

Procedural Comprehension:     9/10 correct  (90%) ✓
Fault Isolation Reasoning:    7/10 correct  (70%) ⚠
Safety Awareness:            10/10 correct (100%) ✓
Parts Interpretation:         4/10 correct  (40%) ✗ WEAK

Overall: 30/40 (75%)

Recommendation: Add more examples for "Fault Isolation" and "Parts Interpretation"
```

**Layer 2: Comparative Evaluation**

Same test queries run through base model (without LoRA). Side-by-side comparison shows training effectiveness.

```
=== COMPARISON: LoRA vs Base Model ===

                        LoRA    Base
Procedural:             90%     60%   +30% improvement ✓
Fault Isolation:        70%     55%   +15% improvement ✓
Safety:                100%     80%   +20% improvement ✓
Parts:                  40%     35%    +5% improvement ⚠

Training effective overall, but "Parts" needs more examples
```

**Layer 3: RAG-Integrated Evaluation**

End-to-end queries requiring both discipline understanding + document retrieval.

```
=== RAG EVALUATION (30 realistic queries) ===

Correct with citations:     24/30 (80%) ✓
Partially correct:           4/30 (13%)
Incorrect:                   2/30 (7%)

Failed queries:
1. "What torque for bolt AN960-416?" → Retrieved correct TM section but LoRA hallucinated value
2. "Grounding procedure for test stand?" → Failed to retrieve relevant chunk (Quarry issue)

Query 1 = training gap (add more parts specification examples to Forge)
Query 2 = retrieval gap (improve Quarry metadata for test equipment procedures)
```

#### Regression Testing

**Version Management:**
```
military_maintenance/
├── v1.0_baseline/
│   ├── lora.safetensors
│   ├── evaluation_results.json
│   └── timestamp: 2026-02-15
├── v1.1_added_parts_examples/
│   ├── lora.safetensors
│   ├── evaluation_results.json
│   ├── regression_report.json  ← Comparison to v1.0
│   └── timestamp: 2026-02-22
```

**Auto-triggered on:**
- Retraining with new examples
- Base model swap
- LoRA merging
- Quarry knowledge base reprocessing

**Regression Report:**
```
=== REGRESSION CHECK: v1.1 vs v1.0 ===

Procedural:        90% → 92%   +2%  ✓ Improved
Fault Isolation:   70% → 68%   -2%  ⚠ Slight regression
Safety:           100% → 100%   0%  ✓ Maintained
Parts:             40% → 72%  +32%  ✓ Major improvement

Overall: 75% → 83% (+8%)

Verdict: APPROVED (minor regression in one area, major gains overall)
Action: Monitor "Fault Isolation" in next iteration
```

**Rollback capability:** If regression unacceptable, revert to previous version with one click.

#### Failure Detection

**Training Issue Detection:**
```
⚠ TRAINING ISSUE DETECTED

Loss not converging after epoch 2. Possible causes:
1. Learning rate too high (try reducing to 1e-4)
2. Insufficient training examples (current: 150, recommended: 300+)
3. Examples too similar (check curriculum diversity in Forge)

Suggestion: Add more diverse examples to weak competency areas in Forge,
then retrain with adjusted learning rate.
```

#### Model Merging (Optional)

**Use case:** Organization has LoRAs for multiple disciplines (maintenance, logistics, safety).

**Process:**
```python
foundry.merge(
    loras=["maintenance_lora.safetensors", "logistics_lora.safetensors"],
    method="ties",  # or "linear"
    output="merged_lora.safetensors"
)
```

**Auto-evaluation of merged model:**
```
=== MERGED MODEL EVALUATION ===

Maintenance competencies:  85% (vs 92% individual) -7%
Logistics competencies:    78% (vs 89% individual) -11%

Tradeoff: Slight accuracy drop for single-model convenience.

Options:
1. Use merged model for everyday convenience (85%/78% still good)
2. Use dual-query mode in Hearth when precision matters (92%/89%)
3. Keep individual models, switch as needed

All options supported. Your choice.
```

---

## Tool 4: Hearth

### Purpose
Built-in interaction layer where practitioners use trained models with Quarry knowledge bases. Feedback loop closes back to Quarry and Forge for continuous improvement.

### Architecture: Chat + Feedback

#### Interaction Layer

**Interface:** Chat panel within Kiln (not standalone deployment)

**Flow:**
```
User Query
    ↓
Model Selection (LoRA + base model)
    ↓
Metadata Pre-Filter (from query analysis)
    ↓
Quarry Retrieval Pipeline (3-stage)
    ↓
Retrieved Chunks (with metadata)
    ↓
LoRA Inference (generate response using retrieved context)
    ↓
Response + Citations
    ↓
User Feedback (accept/flag/follow-up/rephrase)
```

**Example Session:**
```
User: "How do I service hydraulic pump on M1234?"

[Metadata extraction from query]
equipment: "hydraulic_pump"
platform: "M1234"
procedure_type: "maintenance"

[Quarry retrieval with pre-filter]
→ Stage 1 filter: equipment=hydraulic_pump, platform=M1234
→ Stage 2 search: semantic similarity to "service procedure"
→ Stage 3 validate: ensure procedure chunks, not parts lists

[LoRA inference with retrieved chunks]
Model: maintenance_lora
Context: 3 chunks from TM-1234 Section 5.2

[Response]
"To service the hydraulic pump on M1234:

1. Perform lockout/tagout per WP 0020 (CRITICAL: system must be depressurized)
2. Drain hydraulic fluid into approved container (capacity: 15 gallons)
3. Remove mounting bolts (torque spec on reinstall: 45 ft-lbs)
4. Disconnect inlet/outlet lines (cap immediately to prevent contamination)
5. Inspect O-rings and replace if worn (NSN 5330-01-234-5678)

Detailed procedure: TM-1234, WP 0052, pages 5-12 to 5-18

⚠ CAUTION: Do not service pump while system is pressurized. Risk of injury from high-pressure fluid."

[Citations]
📄 TM-1234, Section 5.2.3: Hydraulic Pump Removal (pages 5-12 to 5-18)
📄 TM-1234, WP 0020: Lockout/Tagout Procedures
📄 Parts Catalog, Item 47: O-ring Set (NSN 5330-01-234-5678)

[User feedback options]
✓ Helpful  |  ✗ Incorrect  |  ⚠ Partially correct  |  🔄 Need more detail
```

#### Model Switching

**Interface:** Dropdown or quick-switch buttons

**Options:**
- Select discipline LoRA (if multiple trained)
- Select base model variant (quantized vs. full precision)
- Use merged model (if available)

**Switch time:** < 10 seconds (LoRA load time)

#### Multi-Discipline Query Mode (Dual-Query)

**Use case:** Query spans multiple disciplines (e.g., maintenance + logistics)

**Process:**
```
User: "What parts do I need to service hydraulic pump, and what's the lead time?"

[Dual query mode activated]

Query 1 → Maintenance LoRA + Quarry(maintenance docs)
Query 2 → Logistics LoRA + Quarry(supply chain docs)

[Results]

Maintenance perspective:
"Parts required: O-ring set (NSN 5330-01-234-5678), filter element (NSN 4330-01-111-2222)..."

Logistics perspective:
"NSN 5330-01-234-5678: In stock, 2-day delivery. NSN 4330-01-111-2222: Backordered, 3-week lead time..."

[Comparison summary]
Both answers agree on required parts. Logistics identifies supply constraint (filter element backordered).

Actionable: Order filter element now if service is scheduled <3 weeks out.
```

**Tradeoff:** Dual queries take 2x time (sequential inference on local hardware). User makes informed choice.

#### Feedback Capture and Routing

**Signals collected:**

| Signal | Meaning | Routing |
|--------|---------|---------|
| ✓ Accepted answer | Implicit positive | Log for analysis |
| ✗ Flagged incorrect | Strong negative | Immediate review queue |
| ⚠ Partially correct | Mixed signal | Analyze which part failed |
| 🔄 Follow-up question | Coverage gap | Potential missing competency |
| Rephrased query | Retrieval failure | Quarry improvement candidate |

**Routing Logic:**
```python
def route_feedback(query, response, chunks_retrieved, user_signal):
    if user_signal == "incorrect":
        if chunks_retrieved and chunks_relevant:
            # Good retrieval, bad generation → Forge issue
            return ForgeImprovementQueue(
                issue="model_generated_incorrect_response",
                competency=identify_competency(query),
                suggested_action="add_challenge_example"
            )
        else:
            # Bad retrieval → Quarry issue
            return QuarryImprovementQueue(
                issue="retrieval_failure",
                query_metadata=extract_metadata(query),
                suggested_action="improve_metadata_extraction_or_add_structural_profile"
            )
    elif user_signal == "follow_up":
        # Potential missing competency
        return ForgeImprovementQueue(
            issue="coverage_gap",
            original_query=query,
            follow_up=user_message,
            suggested_action="review_if_new_competency_needed"
        )
```

**Discipline Owner Dashboard:**
```
=== FEEDBACK SUMMARY (Last 7 days) ===

Queries handled: 147
Accepted: 118 (80%)
Flagged incorrect: 12 (8%)
Partially correct: 17 (12%)

Weak areas:
- Torque specifications: 5 incorrect responses (Forge: add more parts examples)
- Test equipment procedures: 7 retrieval failures (Quarry: improve test equipment metadata)

Top-performing areas:
- Safety procedures: 100% correct (45/45)
- Fault isolation: 92% correct (33/36)

Actionable:
1. Add 10-15 torque specification examples to Forge curriculum
2. Review Quarry metadata extraction for test equipment procedures
```

**Critical constraint:** NO automated training data generation. Signals surface opportunities. Humans decide actions.

---

## Data Flow Between Tools

### Forward Flow (Document → Working System)

```
1. Quarry Processing
   PDF → [Tier 1: Classification] → [Tier 2: Extraction] → [Tier 3: Hierarchy + Metadata]
   → Retrieval-ready chunks (JSON)

2. Quarry → Forge Integration
   Chunks → [Forge scaffolding] → Candidate examples
   → Expert reviews/edits → Validated examples

3. Forge → Foundry Export
   Validated examples → Training/test split → JSONL curriculum
   → Foundry training pipeline

4. Foundry Training
   Curriculum + Base model → LoRA training → Trained adapter
   → Evaluation (3 layers) → Version stored

5. Foundry → Hearth Deployment
   Trained LoRA → [Hearth model loader]
   Quarry chunks → [Vector DB] → [Hearth retrieval integration]
   → Working RAG system

6. Hearth Interaction
   User query → [Retrieval + Inference] → Response + Citations
   → User feedback captured
```

### Feedback Flow (Improvement Loop)

```
Hearth Signals
    ↓
  Analyze
    ↓
┌───┴────┐
│        │
↓        ↓
Retrieval  Response
Failure    Quality
Issue      Issue
│          │
↓          ↓
Quarry     Forge
Improvement Improvement
Queue       Queue
│          │
↓          ↓
Human      Human
Review     Review
│          │
↓          ↓
Fix        Add
Metadata   Examples
Extraction to Curriculum
│          │
└────┬─────┘
     ↓
  Retrigger
  Evaluation
     ↓
  Regression
  Test
     ↓
  Approve or
  Rollback
```

---

## Technical Decisions

### Why Small Models (3-20B) Over Frontier Models?

**Rationale:**
- Local deployment on modest hardware (laptop/workstation)
- Zero ongoing inference costs
- No cloud dependency or data custody concerns
- Small models with focused training + precise retrieval match or exceed frontier model domain performance
- Latency acceptable (< 3 seconds end-to-end)

**Tradeoffs accepted:**
- Less capable on out-of-domain queries (acceptable, this is domain-specific tool)
- Longer training times vs. API calls (one-time cost, acceptable)

### Why Human-Validated Training Data Over Synthetic?

**Rationale:**
- Model collapse mathematically inevitable with purely synthetic data
- "Knowledge collapse" produces "confidently wrong" phase (dangerous in safety-critical domains)
- Human-validated data has known provenance, verified accuracy, full auditability
- Research shows 1,000 curated human examples >> 10,000 synthetic examples
- Regulatory compliance may require human-in-the-loop for training data

**Tradeoffs accepted:**
- More SME time required (2-4 hours per discipline)
- Smaller training sets (300-500 vs. thousands synthetic)
- Slower curriculum growth

**Mitigation:**
- Forge makes SME time efficient (structured workflow, real-time guidance)
- Multi-contributor support distributes burden
- 300-500 discipline-level examples sufficient (research-backed)

### Why Metadata-Filtered Retrieval Over Pure Semantic Search?

**Rationale:**
- Deterministic pre-filter reduces search space 80-90% at zero computational cost
- Critical for resource-constrained local deployment
- Dramatically reduces false positives (precision over recall)
- Metadata extraction from formatting is reliable for well-formatted documents
- Target market (military, legal, healthcare) produces standardized docs

**Tradeoffs accepted:**
- Requires well-formatted source documents (graceful degradation on poor formatting)
- Metadata extraction rules need maintenance as document types expand

**Mitigation:**
- Tier 1 classifier adapts to new document types
- Validation pass (Stage 3) catches metadata mismatches
- Manual fallback always available

### Why Traditional ML Over LLM for Document Classification?

**Rationale:**
- Format detection is classification with well-defined features (ML's strength)
- ML faster (milliseconds), deterministic, inspectable (feature importances)
- No GPU required (runs on any hardware)
- Content-agnostic (structure only)
- Smaller model (KB vs. GB)

**Tradeoffs accepted:**
- Requires manual feature engineering (one-time cost)
- Classifier needs retraining as new document types added

**Mitigation:**
- Manual classification fallback captures new training examples automatically
- Classifier improves with every validated document

---

## Security Architecture

### Defense in Depth

**Layer 1: Input Validation**
- File type validation (MIME + magic numbers)
- Size limits (default 100MB, configurable)
- Path traversal prevention
- No user-controlled paths in file operations

**Layer 2: Sandboxing**
- PDF extraction (Tier 2) runs in isolated process
- Resource limits: memory cap, timeout (30s default)
- No shell commands with user input

**Layer 3: Data Privacy**
- All processing local (no cloud transmission)
- Training data access controls (contributor attribution tracked)
- PII redaction before export (regex + NER)
- Curriculum versioning and audit trail

**Layer 4: Secrets Management**
- No hardcoded secrets (environment variables only)
- .env never committed (gitignore enforced)
- Hooks prevent accidental secret commits (H-04 secrets firewall)

**Layer 5: Dependency Security**
- Automated CVE scanning (H-09 dependency auditor)
- Pinned versions in requirements.txt / package-lock.json
- Regular security updates (monthly cycle)

---

## Performance Optimization

### Quarry Optimizations

- **Tier 1 caching:** Fingerprints cached (don't recompute for same doc)
- **Tier 2 GPU acceleration:** Docling uses CUDA if available
- **Tier 3 batch processing:** Process multiple chunks in parallel
- **Metadata indexing:** Pre-built indices for common filter queries

### Foundry Optimizations

- **Mixed precision training:** Faster, less memory (automatic with Unsloth)
- **Gradient checkpointing:** Reduce memory footprint
- **LoRA rank tuning:** Lower rank = faster, smaller adapter (defaults tuned)
- **Evaluation caching:** Don't re-run unchanged test cases

### Hearth Optimizations

- **Model quantization:** 4-bit/8-bit for faster inference (optional)
- **KV cache:** Cache key-value tensors across turns (conversation mode)
- **Batch retrieval:** Retrieve multiple chunks in one vector DB query
- **Prefetching:** Load common models/chunks into memory

---

## Deployment Architecture (MVP)

```
┌─────────────────────────────────────────────────────────┐
│  Kiln Desktop Application (Electron)                    │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐                 │
│  │  Frontend UI   │  │  Python Backend│                 │
│  │  (React/TS)    │←→│  (FastAPI)     │                 │
│  │                │  │                │                 │
│  │  - Quarry UI   │  │  - Quarry API  │                 │
│  │  - Forge UI    │  │  - Forge API   │                 │
│  │  - Foundry UI  │  │  - Foundry API │                 │
│  │  - Hearth UI   │  │  - Hearth API  │                 │
│  └────────────────┘  └────────────────┘                 │
│         ↑                     ↑                          │
│         │                     │                          │
│  ┌──────┴─────────────────────┴──────┐                  │
│  │  Local Filesystem                 │                  │
│  │  - Documents (PDFs)                │                  │
│  │  - Processed chunks (JSON)         │                  │
│  │  - Curricula (JSONL)               │                  │
│  │  - Trained LoRAs (.safetensors)    │                  │
│  │  - Vector DB (ChromaDB/Qdrant)     │                  │
│  └────────────────────────────────────┘                  │
│                                                          │
│  No cloud connectivity required after initial setup     │
│  All inference local (privacy preserved)                │
└─────────────────────────────────────────────────────────┘
```

**System Requirements (MVP):**
- **Minimum:** Laptop with 16GB RAM, 50GB storage (quantized 7B model)
- **Recommended:** Workstation with 32GB RAM, NVIDIA GPU (8GB VRAM), 100GB storage (full 20B model)

---

## Future Architecture (Post-MVP)

Not in scope for MVP, but designed for:

- **OCR support:** Scanned PDF routing through Docling OCR before Tier 1
- **Multi-discipline deployment:** Simultaneous serving of multiple LoRAs
- **Cloud deployment option:** Docker containers, Kubernetes orchestration
- **Collaboration features:** Shared curricula, distributed SME contributions
- **API exposure:** Allow external tools to integrate Kiln components

---

## Versioning Strategy

**Semantic versioning:** MAJOR.MINOR.PATCH

- **MAJOR:** Breaking API changes, data format incompatibilities
- **MINOR:** New features, backward-compatible
- **PATCH:** Bug fixes, performance improvements

**Component versioning:**
- Quarry: v0.7.x (70% complete at MVP start)
- Forge: v0.1.x (MVP first version)
- Foundry: v0.1.x (MVP first version)
- Hearth: v0.1.x (MVP first version)

**Export format versioning:** All exports include `version` field for forward compatibility

---

## Conclusion

Kiln's architecture is designed for:
- **Trustworthiness:** Human validation at critical points
- **Local ownership:** No cloud dependency, zero ongoing costs
- **Compounding improvement:** Every use generates data for future enhancement
- **Enterprise readiness:** Security, auditability, regulatory compliance built-in

The four tools form a closed loop from raw documents to working AI system and back, with feedback driving continuous improvement while maintaining human authority over all critical decisions.

