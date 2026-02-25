# Kiln MVP Demonstration

## Overview

Kiln is a complete pipeline for building trustworthy domain-specific AI systems.
This demonstration walks through all four tools with military maintenance as the
example discipline, proving the end-to-end pipeline works.

**What MVP proves:** An organization can take domain documents, build training
curricula with subject-matter experts, train a local model, and query it with
citations -- all without cloud connectivity or ML expertise.

## Prerequisites

- Python 3.10+
- Dependencies installed: `pip install -e ".[dev]"`
- No GPU required (MVP uses dry-run training backend)

## Running the Demo

```bash
# From the project root
PYTHONPATH=. python scripts/demo_mvp.py

# Or via pytest (recommended, handles path setup)
python -m pytest tests/integration/test_mvp_demo.py -v -s
```

## Demo Flow

### Step 1: Quarry -- Document Processing

Quarry processes raw PDF documents into retrieval-ready chunks. The demo:

1. **Creates a synthetic military technical manual** (TM 9-2320-280-20)
   containing maintenance procedures, tools lists, and parts appendices.
2. **Fingerprints the document structure** using statistical analysis of
   fonts, layout, whitespace, and byte-level features -- no content parsing.
3. **Converts the fingerprint to a 49-dimension feature vector** for the
   ML classifier.
4. **Classifies the document type** using a GradientBoostingClassifier
   trained on synthetic profiles. Classification runs in under 1 second.

Key architectural decision: Quarry uses traditional ML (not LLM) for
classification because format detection is a classification problem with
well-defined features. ML is faster, deterministic, and inspectable.

### Step 2: Forge -- Curriculum Building

Forge guides domain experts through creating human-validated training data.
The demo:

1. **Creates a contributor** (SSG Rodriguez, a maintenance SME).
2. **Creates a discipline** (Military Vehicle Maintenance) with domain
   vocabulary and document types.
3. **Maps four competency areas:**
   - Procedural Comprehension
   - Parts Identification
   - Fault Isolation
   - Safety Awareness
4. **Adds 8 human-validated training examples** across all competencies,
   with 3 held out as a test set.
5. **Exports curriculum to JSONL** in Alpaca format for Foundry consumption.

Key architectural decision: Forge trains at the discipline level, not
document level. Specific factual content comes from RAG at inference time.
300-500 examples cover full competency range for a discipline.

### Step 3: Foundry -- Training and Evaluation

Foundry handles LoRA fine-tuning and competency-based evaluation. The demo:

1. **Auto-configures hyperparameters** based on curriculum size (5 examples
   triggers small-dataset configuration: more epochs, lower learning rate).
2. **Runs training pipeline** using the dry-run backend (simulates training
   metrics without requiring a GPU or real model).
3. **Evaluates the model** against held-out test cases using keyword
   similarity scoring.
4. **Reports results in SME-friendly language:**
   - "Procedural Comprehension: 0/1 correct" (not "F1: 0.87")
   - "Parts Identification: needs improvement" (not "perplexity: 8.3")

Key architectural decision: Foundry reports results in competency language
that domain experts understand. No ML jargon.

### Step 4: Hearth -- Interaction Layer

Hearth provides the query interface connecting trained models with document
retrieval. The demo:

1. **Sets up RAG pipeline** with mock inference and mock retrieval (pre-
   configured with military TM chunks).
2. **Queries:** "How do I replace a hydraulic filter?"
3. **Shows response with citations:**
   - Answer references TM 9-2320-280-20, paragraph 3-1
   - Three source chunks cited with document title, section, and page
4. **Captures feedback signal** (thumbs up/down) and routes to Forge
   review queue for human review.

Key architectural decision: Feedback surfaces opportunities for human review.
It does NOT auto-generate training data from model outputs. Human authority
is maintained throughout.

## Key Metrics

| Metric | Result |
|--------|--------|
| Document fingerprinting | ~0.02s per document |
| Classification (with training) | ~3s (training) + <0.001s (inference) |
| Feature vector dimensions | 49 |
| Curriculum examples created | 8 (5 train, 3 test) |
| Competency areas mapped | 4 |
| Training pipeline (dry-run) | <0.001s |
| RAG query response | <1ms (mock inference) |
| Citations per response | 3 |
| Total automated tests | 1,608+ |

## What the Demo Does NOT Show

- Real GPU training (uses dry-run backend)
- OCR for scanned documents (out of scope)
- Multi-discipline deployment
- Production UI (backend-only MVP)
- Real model inference (uses MockInference)

These are deliberate MVP scope boundaries. See `docs/VALIDATION_CRITERIA.md`
for the full list of what MVP proves and does not prove.

## Troubleshooting

**ModuleNotFoundError:** Ensure you run from the project root with
`PYTHONPATH=.` set, or use `python -m pytest` which handles this.

**Import errors for fitz:** Install PyMuPDF: `pip install pymupdf`

**scikit-learn errors:** Install ML dependencies: `pip install scikit-learn joblib`
