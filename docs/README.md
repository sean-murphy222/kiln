# CHONK

**Visual Document Chunking Studio for RAG Pipelines**

See your document's structure. Build chunks that actually work.

---

## What is CHONK?

CHONK is **not** just another PDF chunker. It's a **visual chunk organization studio** that makes RAG actually work.

### The Problem

Everyone can extract text from PDFs. But creating **meaningful chunks** for RAG is hard:

- ❌ Most tools give you arbitrary 512-token fragments
- ❌ Document structure gets destroyed
- ❌ You pay for embeddings, then discover retrieval doesn't work
- ❌ No way to visualize or refine chunks before committing

### The CHONK Solution

1. **VISUALIZE** - See your document as a tree, not flat text
2. **ORGANIZE** - Chunks respect sections, preserve context
3. **TEST** - Try queries BEFORE paying for embeddings
4. **REFINE** - Merge/split/annotate visually
5. **EXPORT** - With confidence

## Key Features

### 🌳 Visual Hierarchy Explorer

```
Document Tree View                  Chunk Preview
├─ FOREWORD                        ┌────────────────────┐
├─ Section 1: Introduction         │ FOREWORD           │
│  ├─ 1.1 Purpose                  │                    │
│  └─ 1.2 Scope                    │ This standard is...│
├─ Section 2: Requirements         │                    │
   ├─ 2.1 General                  │ [290 tokens]       │
   └─ 2.2 Technical                └────────────────────┘
```

### 🎯 Test-Before-Embed Workflow

Don't guess if your chunks will work - **test them first**:

```
Query: "What are the safety requirements?"

Flat Chunking:     ❌ Mixed content, partial sections
Hierarchical:      ✅ Exact section "2.1 Safety Requirements"
```

### 📊 Strategy Comparison

Try multiple chunking strategies side-by-side:

- **Hierarchical** (section-based) - RECOMMENDED
- **Fixed** (token-based) - baseline
- **Semantic** (embedding-based) - advanced

See concrete metrics before choosing.

### 🎨 Visual Refinement

- Click to merge sections
- Split oversized chunks
- Lock perfect chunks
- Add notes for your team

## Installation

```bash
# Basic installation
pip install chonk

# With enhanced extraction (Docling - RECOMMENDED)
pip install chonk[enhanced]

# With AI-powered extraction (LayoutParser)
pip install chonk[ai]
```

## Quick Start

### GUI Mode (Recommended)

```bash
chonk
```

Then open http://localhost:8420

### CLI Mode

```python
from chonk import CHONK

# Load document and build hierarchy
chonk = CHONK("document.pdf")
tree = chonk.build_hierarchy()

# Preview chunks with different strategies
hierarchical_chunks = chonk.preview_chunks(strategy="hierarchical")
fixed_chunks = chonk.preview_chunks(strategy="fixed")

# Compare strategies
comparison = chonk.compare_strategies(
    ["hierarchical", "fixed"],
    test_queries=["What are the requirements?", "How do I install?"]
)

print(comparison.recommendation)
# "✅ RECOMMENDED: HIERARCHICAL strategy"
# "  Reasons: preserves document structure, high quality chunks"

# Export
chonk.export(strategy="hierarchical", format="jsonl")
```

## How It Works

### 1. Extract Blocks

CHONK supports multiple extraction backends:

- **Tier 1 (Fast)**: PyMuPDF + pdfplumber - built-in, basic
- **Tier 2 (Enhanced)**: IBM Docling - GPU-accelerated, excellent structure detection ⭐
- **Tier 3 (AI)**: LayoutParser - deep learning, for complex documents

**Tip**: Use Docling (Tier 2) for best hierarchy quality.

### 2. Build Hierarchy

CHONK analyzes extracted blocks and builds a **tree structure**:

```python
{
  "section_id": "2.1",
  "heading": "Safety Requirements",        # ← Separated
  "content": "All procedures must...",     # ← Clean content
  "children": [
    {
      "section_id": "2.1.1",
      "heading": "General Safety",
      "content": "...",
      "children": []
    }
  ]
}
```

### 3. Choose Strategy

Different documents need different strategies:

**Hierarchical** (Best for structured docs):
- Respects section boundaries
- Preserves context with hierarchy paths
- Example: Technical manuals, standards, research papers

**Fixed** (Baseline):
- Token-based sliding window
- Good for comparison only
- Use to show why hierarchical is better

**Semantic** (Advanced):
- Embedding-based similarity
- Good for unstructured documents
- More expensive (requires embeddings)

### 4. Test Queries

The killer feature: **test retrieval BEFORE embedding**

```python
# Define test queries
queries = [
    "What are the safety requirements?",
    "How do I perform maintenance?",
    "What tools are required?"
]

# Test each strategy
results = chonk.test_queries(
    strategies=["hierarchical", "fixed"],
    queries=queries
)

# See which strategy retrieves better results
print(results.best_strategy)
# "hierarchical" - found exact sections vs mixed content
```

### 5. Refine & Export

- Review quality scores
- Merge/split problem chunks
- Export in your preferred format:
  - JSONL (LangChain/LlamaIndex compatible)
  - JSON (full metadata)
  - Nested JSON (hierarchy preserved)
  - CSV

## Example Output

### Hierarchical Chunks (Good)

```json
{
  "id": "chunk_abc123",
  "heading": "2.1 Safety Requirements",
  "content": "All procedures must follow safety protocols...",
  "hierarchy_path": "Section 2 Requirements > 2.1 Safety",
  "token_count": 290,
  "quality_score": 1.0,
  "page_range": [12, 13]
}
```

✅ Complete section
✅ Context preserved (hierarchy path)
✅ Clean heading/content separation

### Flat Chunks (Bad)

```json
{
  "id": "chunk_xyz789",
  "content": "...end of section 1.2. 2.1 Safety Requirements Safety is critical...",
  "token_count": 512
}
```

❌ Sections mixed together
❌ No context
❌ Arbitrary boundaries

## What Makes CHONK Different?

### vs. LangChain/LlamaIndex

**They give you:**
- "Here's 500 chunks, good luck!"
- No visualization
- No testing
- No refinement

**CHONK gives you:**
- "Here's your document as a tree with 2,700 sections"
- Visual hierarchy explorer
- Test queries before embedding
- Interactive refinement

### vs. unstructured.io

**They focus on:** Extraction (blocks from PDFs)
**CHONK focuses on:** Organization (blocks → intelligent chunks)

Extraction is commodity. Organization is value.

## Architecture

### Backend (Python + FastAPI)

```
src/chonk/
├── hierarchy/          # 🌟 CORE - Document structure
├── chunking/           # 🌟 CORE - Multiple strategies
├── comparison/         # 🌟 CORE - Strategy comparison
├── testing/            # 🌟 KILLER FEATURE - Test retrieval
├── extraction/         # 📦 Commodity - Get blocks
└── exporters/          # Export formats
```

### Frontend (Electron + React + TypeScript)

```
ui/src/components/
├── HierarchyTree/      # 🌟 Visual document structure
├── ChunkPreview/       # 🌟 Live preview
├── QueryTester/        # 🌟 Test before embed
├── ComparisonDashboard/# 🌟 Compare strategies
└── ...
```

## Use Cases

### Technical Documentation

- Military standards (MIL-STD)
- API documentation
- User manuals

**Why CHONK**: Hierarchical structure is critical. Sections must stay intact.

### Research Papers

- Academic papers
- White papers
- Reports

**Why CHONK**: Section-based chunking preserves argument flow and citations.

### Legal Documents

- Contracts
- Policies
- Regulations

**Why CHONK**: Legal sections have meaning. Don't mix them.

### Unstructured Content

- Blogs
- Articles
- Books

**Why CHONK**: Try semantic chunking, compare with hierarchical, test queries.

## Configuration

### Chunking Parameters

```python
from chonk.chunkers import HierarchyChunker, ChunkerConfig

config = ChunkerConfig(
    max_tokens=512,              # Max tokens per chunk
    overlap_tokens=50,           # Overlap between chunks
    preserve_tables=True,        # Keep tables intact
    group_under_headings=True,   # Respect sections
)

chunker = HierarchyChunker(config=config)
```

### Quality Scores

Chunks are automatically scored on:

- **token_range** - Is size optimal?
- **sentence_complete** - Proper boundaries?
- **hierarchy_preserved** - No orphan headings?
- **table_integrity** - Tables not split?
- **reference_complete** - No orphan references?

## Roadmap

- ✅ Hierarchy tree building
- ✅ Multiple chunking strategies
- ✅ Quality scoring
- ✅ Strategy comparison
- ✅ Nested JSON export
- 🚧 Visual hierarchy tree UI
- 🚧 Interactive chunk refinement
- 📋 Real-time chunk preview
- 📋 Recommendation engine
- 📋 Export format customization
- 📋 Batch processing
- 📋 Cloud deployment option

## Contributing

We welcome contributions! CHONK is focused on **chunk organization**, not just extraction.

**Priority areas:**
1. Hierarchy visualization improvements
2. New chunking strategies
3. Better quality metrics
4. UI/UX enhancements

## License

MIT License - See LICENSE file

## Support

- 📖 [Documentation](https://github.com/yourusername/chonk/docs)
- 🐛 [Issues](https://github.com/yourusername/chonk/issues)
- 💬 [Discussions](https://github.com/yourusername/chonk/discussions)

---

**The Figma of RAG Chunking**

CHONK makes chunking visual, testable, and actually work.
