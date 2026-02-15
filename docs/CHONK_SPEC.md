# CHONK - Visual Document Chunking Studio

## Project Overview

CHONK is a local-first, visual document preparation tool for semantic embedding and RAG pipelines. It provides a GUI for refining AI-generated chunk boundaries and testing retrieval before committing to expensive embedding operations.

**Tagline**: "Know your chunks work before you embed them."

**Core Value Proposition**:
- Visual feedback loop BEFORE embedding (no one else does this)
- Test retrieval BEFORE export (the killer feature)
- Local-first (documents never leave your machine)
- Multi-format support (PDF, DOCX, HTML, Markdown, TXT)
- One-time license (no per-page API fees)

**The Core Loop**:
```
DROP FILE → SEE CHUNKS → TEST QUERIES → REFINE → EXPORT JSON
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CHONK                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
│  │   LOADERS   │ → │   PARSERS   │ → │    CHUNKERS     │    │
│  │             │   │             │   │                 │    │
│  │ pdf_loader  │   │ layout      │   │ fixed_size      │    │
│  │ docx_loader │   │ semantic    │   │ recursive       │    │
│  │ pptx_loader │   │ table       │   │ semantic        │    │
│  │ html_loader │   │ ocr         │   │ hierarchy       │    │
│  │ md_loader   │   │             │   │ schema_aware    │    │
│  └─────────────┘   └─────────────┘   └─────────────────┘    │
│         │                │                   │               │
│         └────────────────┼───────────────────┘               │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            UNIFIED DOCUMENT MODEL (UDM)              │   │
│  │                                                      │   │
│  │  {                                                   │   │
│  │    "source": "document.pdf",                         │   │
│  │    "blocks": [                                       │   │
│  │      {                                               │   │
│  │        "id": "block_001",                            │   │
│  │        "type": "text|table|image|code|heading",      │   │
│  │        "content": "...",                             │   │
│  │        "bbox": [x1, y1, x2, y2],                     │   │
│  │        "page": 1,                                    │   │
│  │        "parent_id": null,                            │   │
│  │        "metadata": {}                                │   │
│  │      }                                               │   │
│  │    ],                                                │   │
│  │    "chunks": [                                       │   │
│  │      {                                               │   │
│  │        "id": "chunk_001",                            │   │
│  │        "block_ids": ["block_001", "block_002"],      │   │
│  │        "content": "...",                             │   │
│  │        "token_count": 256,                           │   │
│  │        "quality_score": 0.85,                        │   │
│  │        "hierarchy_path": "Section 1 > Subsection A" │   │
│  │      }                                               │   │
│  │    ]                                                 │   │
│  │  }                                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 VISUAL EDITOR UI                     │   │
│  │                                                      │   │
│  │  ┌────────────────────┬─────────────────────────┐   │   │
│  │  │                    │                         │   │   │
│  │  │   DOCUMENT VIEW    │    CHUNK PANEL          │   │   │
│  │  │                    │                         │   │   │
│  │  │  [Rendered doc     │  • Chunk list           │   │   │
│  │  │   with overlay     │  • Token counts         │   │   │
│  │  │   showing chunk    │  • Quality scores       │   │   │
│  │  │   boundaries]      │  • Hierarchy view       │   │   │
│  │  │                    │  • Merge/Split buttons  │   │   │
│  │  │  Click to select   │                         │   │   │
│  │  │  Drag to resize    │  [EXPORT]               │   │   │
│  │  │                    │                         │   │   │
│  │  └────────────────────┴─────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 RETRIEVAL TESTER                     │   │
│  │                                                      │   │
│  │  • Local embedding (all-MiniLM-L6-v2 default)        │   │
│  │  • Optional: OpenAI text-embedding-3-small           │   │
│  │  • In-memory vector search (numpy cosine sim)        │   │
│  │  • Query → top-k chunks with similarity scores       │   │
│  │  • "Know your chunks work before you embed them"     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    EXPORTERS                         │   │
│  │                                                      │   │
│  │  • JSONL (LangChain/LlamaIndex compatible)           │   │
│  │  • JSON with metadata                                │   │
│  │  • CSV (simple)                                      │   │
│  │  • Direct to Chroma/Pinecone (future)                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend (Python)
```
chonk/
├── __init__.py
├── cli.py                 # CLI entry point
├── server.py              # FastAPI server for GUI
├── core/
│   ├── __init__.py
│   ├── document.py        # Unified Document Model
│   ├── loader.py          # Base loader class
│   ├── parser.py          # Base parser class
│   ├── chunker.py         # Base chunker class
│   └── exporter.py        # Base exporter class
├── loaders/
│   ├── __init__.py
│   ├── pdf.py             # PyMuPDF + pdfplumber
│   ├── docx.py            # python-docx
│   ├── pptx.py            # python-pptx
│   ├── html.py            # BeautifulSoup
│   └── markdown.py        # markdown-it
├── parsers/
│   ├── __init__.py
│   ├── layout.py          # Layout detection
│   ├── table.py           # Table extraction
│   ├── semantic.py        # Sentence/paragraph detection
│   └── ocr.py             # Tesseract/EasyOCR wrapper
├── chunkers/
│   ├── __init__.py
│   ├── fixed.py           # Fixed size with overlap
│   ├── recursive.py       # Recursive character
│   ├── semantic.py        # Embedding-based semantic
│   └── hierarchy.py       # Heading-aware hierarchical
├── exporters/
│   ├── __init__.py
│   ├── jsonl.py
│   ├── json_export.py
│   └── csv_export.py
├── testing/
│   ├── __init__.py
│   ├── embedder.py         # Local embedding models
│   ├── searcher.py         # In-memory vector search
│   └── test_suite.py       # Saved test queries (Phase 2)
└── utils/
    ├── __init__.py
    ├── tokens.py          # Token counting (tiktoken)
    └── quality.py         # Chunk quality scoring
```

### Frontend (Electron + React)
```
chonk-ui/
├── package.json
├── electron/
│   ├── main.js            # Electron main process
│   └── preload.js
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── DocumentViewer.tsx    # PDF.js / doc renderer
│   │   ├── ChunkOverlay.tsx      # SVG overlay for boundaries
│   │   ├── ChunkPanel.tsx        # Right sidebar
│   │   ├── ChunkCard.tsx         # Individual chunk display
│   │   ├── QualityBadge.tsx      # Quality score indicator
│   │   ├── RetrievalTester.tsx   # Query testing panel
│   │   ├── SearchResult.tsx      # Individual search result
│   │   └── ExportModal.tsx
│   ├── hooks/
│   │   ├── useDocument.ts
│   │   ├── useChunks.ts
│   │   └── useExport.ts
│   └── api/
│       └── chonk.ts              # API calls to Python backend
└── public/
    └── index.html
```

---

## Unified Document Model (UDM) Schema

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class BlockType(Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    LIST = "list"
    FOOTER = "footer"
    HEADER = "header"

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    page: int

@dataclass
class Block:
    id: str
    type: BlockType
    content: str
    bbox: Optional[BoundingBox] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Parser confidence score

@dataclass
class Chunk:
    id: str
    block_ids: List[str]
    content: str
    token_count: int
    quality_score: float  # 0-1, computed by quality analyzer
    hierarchy_path: str   # "Section 1 > Subsection A"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For visual editor
    is_modified: bool = False
    is_locked: bool = False  # User locked this chunk

@dataclass
class ChonkDocument:
    source: str
    source_type: str  # pdf, docx, pptx, etc.
    blocks: List[Block]
    chunks: List[Chunk]
    page_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    loader_used: str
    parser_used: str
    chunker_used: str
    chunker_config: Dict[str, Any] = field(default_factory=dict)
```

---

## Quality Scoring System

Each chunk gets a quality score (0-1) based on:

```python
def compute_quality_score(chunk: Chunk, document: ChonkDocument) -> float:
    scores = []
    
    # 1. Token count in optimal range (200-500 tokens ideal)
    token_score = 1.0
    if chunk.token_count < 100:
        token_score = chunk.token_count / 100
    elif chunk.token_count > 600:
        token_score = max(0.3, 1 - (chunk.token_count - 600) / 1000)
    scores.append(("token_range", token_score, 0.25))
    
    # 2. Sentence completeness (starts with capital, ends with period)
    content = chunk.content.strip()
    sentence_score = 1.0
    if content and not content[0].isupper():
        sentence_score -= 0.3
    if content and content[-1] not in '.!?:':
        sentence_score -= 0.3
    scores.append(("sentence_complete", max(0, sentence_score), 0.20))
    
    # 3. Hierarchy preservation (heading followed by content)
    hierarchy_score = 1.0
    blocks = [b for b in document.blocks if b.id in chunk.block_ids]
    has_orphan_heading = any(
        b.type == BlockType.HEADING and 
        b.id == chunk.block_ids[-1]  # Heading at end of chunk
        for b in blocks
    )
    if has_orphan_heading:
        hierarchy_score = 0.4
    scores.append(("hierarchy", hierarchy_score, 0.25))
    
    # 4. Table integrity (tables not split)
    table_score = 1.0
    # Check if any table block is partially included
    # (implementation depends on table detection)
    scores.append(("table_integrity", table_score, 0.15))
    
    # 5. Reference completeness (no orphaned references)
    ref_score = 1.0
    # Check for patterns like "see above" or "as shown in" without context
    scores.append(("references", ref_score, 0.15))
    
    # Weighted average
    total = sum(score * weight for _, score, weight in scores)
    return round(total, 3)
```

---

## Retrieval Tester ("Test Before You Embed")

The killer feature that closes the feedback loop. Users can test queries against their chunks BEFORE exporting and embedding elsewhere.

### Why This Matters

| Without Testing | With Testing |
|-----------------|--------------|
| "I think these chunks look right" | "I know these chunks retrieve correctly" |
| Export, embed, build RAG, test, fail, restart | Test before export, fix issues immediately |
| Debugging happens in production | Debugging happens in CHONK |
| Hours wasted on bad chunks | Minutes to validate |

### UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Document View]              │  [Chunk Panel]              │
│                               │                             │
│   (rendered document)         │   (chunk list)              │
│                               │                             │
├───────────────────────────────┴─────────────────────────────┤
│  🔍 TEST RETRIEVAL                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ "What are the safety requirements for maintenance?"    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  TOP MATCHES:                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🟢 0.89  Chunk 7 (p.3)                               │   │
│  │ "Safety requirements include lockout/tagout..."      │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🟡 0.72  Chunk 12 (p.5)                              │   │
│  │ "Maintenance personnel shall wear protective..."     │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🔴 0.61  Chunk 3 (p.1)                               │   │
│  │ "This manual covers system maintenance..."           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  💡 Chunk 15 "Safety warnings for hydraulic systems"       │
│     scored 0.58 - consider merging with Chunk 7?           │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

class RetrievalTester:
    """
    Local embedding + search for testing chunk quality.
    Runs entirely on-device, no API keys required.
    """
    
    # Model options (user can choose in settings)
    MODELS = {
        "fast": "all-MiniLM-L6-v2",      # 80MB, fastest
        "balanced": "all-mpnet-base-v2",  # 420MB, better quality
        "accurate": "bge-small-en-v1.5",  # 130MB, modern
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.chunks: Optional[List[Chunk]] = None
        self._is_indexed = False
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """
        Embed all chunks. Called once after chunking/re-chunking.
        Typically takes 1-5 seconds for 100 chunks.
        """
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.chunk_embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        self._is_indexed = True
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Find most similar chunks to query using cosine similarity.
        Returns top_k results sorted by relevance.
        """
        if not self._is_indexed:
            raise ValueError("Must call index_chunks() first")
        
        # Embed the query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )[0]
        
        # Cosine similarity against all chunks
        similarities = np.dot(self.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.chunk_embeddings, axis=1) * 
            np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(similarities[i]),
                rank=rank + 1
            )
            for rank, i in enumerate(top_indices)
        ]
    
    def reindex_if_needed(self, chunks: List[Chunk]) -> None:
        """Re-index only if chunks have changed."""
        if self.chunks is None or len(chunks) != len(self.chunks):
            self.index_chunks(chunks)
            return
        
        # Check if any content changed
        for old, new in zip(self.chunks, chunks):
            if old.content != new.content:
                self.index_chunks(chunks)
                return


# API integration
class ChunkTesterAPI:
    """FastAPI endpoints for retrieval testing."""
    
    def __init__(self):
        self.tester = RetrievalTester()
    
    async def index(self, document_id: str, chunks: List[Chunk]):
        """POST /api/test/index"""
        self.tester.index_chunks(chunks)
        return {"status": "indexed", "chunk_count": len(chunks)}
    
    async def search(self, query: str, top_k: int = 5):
        """POST /api/test/search"""
        results = self.tester.search(query, top_k)
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": r.chunk.id,
                    "score": r.score,
                    "rank": r.rank,
                    "preview": r.chunk.content[:200],
                    "page": r.chunk.metadata.get("page"),
                }
                for r in results
            ]
        }
```

### Model Options

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 80MB | ~50ms/query | Good | Default, fast iteration |
| `all-mpnet-base-v2` | 420MB | ~150ms/query | Better | When quality matters |
| `bge-small-en-v1.5` | 130MB | ~80ms/query | Better | Modern alternative |
| OpenAI API | N/A | ~200ms/query | Best | Phase 2, optional |

**Default:** `all-MiniLM-L6-v2` - good enough for testing, fast enough for live preview.

### User Flow

1. User chunks document
2. Chunks are auto-indexed in background (1-5 sec)
3. User types test query in search box
4. Results appear instantly with similarity scores
5. User clicks result → chunk highlights in document view
6. If wrong chunks retrieved → user refines chunking
7. Repeat until satisfied
8. Export with confidence

### Score Interpretation (UI Badges)

| Score | Badge | Meaning |
|-------|-------|---------|
| ≥ 0.80 | 🟢 | Strong match - this chunk is highly relevant |
| 0.65 - 0.79 | 🟡 | Moderate match - probably relevant |
| 0.50 - 0.64 | 🟠 | Weak match - might be relevant |
| < 0.50 | 🔴 | Poor match - likely not relevant |

### Future Enhancements (Phase 2+)

**Test Suites:**
```python
@dataclass
class TestCase:
    query: str
    expected_chunk_ids: List[str]  # Should be in top-k
    excluded_chunk_ids: List[str]  # Should NOT be in top-k

@dataclass  
class TestSuite:
    name: str
    test_cases: List[TestCase]
    
    def run(self, tester: RetrievalTester) -> TestReport:
        """Run all test cases, return pass/fail report."""
        pass
```

**Coverage Analysis:**
```python
def analyze_coverage(chunks: List[Chunk], test_suite: TestSuite) -> CoverageReport:
    """
    Identify chunks that are never retrieved by any test query.
    Helps find dead content or gaps in test coverage.
    """
    pass
```

**Similarity Heatmap:**
```python
def compute_similarity_matrix(chunks: List[Chunk]) -> np.ndarray:
    """
    NxN matrix of chunk-to-chunk similarity.
    Helps identify redundant chunks or merge candidates.
    """
    pass
```

---

## MVP Feature Set

### Phase 1: Core MVP (6-8 weeks solo dev)

**Must Have:**
- [ ] PDF loader (PyMuPDF + pdfplumber hybrid)
- [ ] DOCX loader (python-docx)
- [ ] HTML loader (BeautifulSoup + readability)
- [ ] Markdown loader
- [ ] Plain text loader
- [ ] Basic layout parser (text blocks, headings, tables)
- [ ] Fixed-size chunker with overlap
- [ ] Recursive character chunker
- [ ] Hierarchy-aware chunker (respects headings)
- [ ] Unified Document Model
- [ ] Quality scoring (basic)
- [ ] Electron app shell
- [ ] Document viewer (PDF.js for PDF, HTML render for others)
- [ ] Chunk boundary overlay (colored rectangles)
- [ ] Chunk list panel with token counts
- [ ] Click to select chunk
- [ ] Merge two adjacent chunks (button)
- [ ] Split chunk at cursor position
- [ ] **Retrieval tester (local embedding + search)**
- [ ] **Test query input with live results**
- [ ] **Click search result → highlight chunk in doc**
- [ ] JSONL export
- [ ] JSON export with metadata

**Nice to Have (Phase 1.5):**
- [ ] PPTX loader
- [ ] Semantic chunking (requires embedding model)
- [ ] Drag to resize chunk boundaries
- [ ] Undo/redo
- [ ] Dark mode
- [ ] OpenAI embeddings option for testing

### Phase 2: Differentiation (4-6 weeks)

- [ ] PPTX loader
- [ ] Semantic chunking with embedding preview
- [ ] **Saved test suites (save queries, expected results)**
- [ ] **Coverage analysis (which chunks never retrieved)**
- [ ] **Chunk similarity heatmap**
- [ ] Template system ("remember my settings for this doc type")
- [ ] Batch processing (folder of docs)
- [ ] Chunk comparison view (before/after)
- [ ] Export directly to Chroma/Pinecone
- [ ] Schema-aware chunking (load DTD/XSD)
- [ ] Multi-engine comparison (run pdfplumber AND PyMuPDF, compare)
- [ ] OCR integration for scanned PDFs

### Phase 3: Enterprise (ongoing)

- [ ] Confluence connector
- [ ] SharePoint connector
- [ ] Notion connector
- [ ] S3/GCS batch import
- [ ] Team features (shared templates)
- [ ] API for headless operation
- [ ] Custom chunker plugins

---

## CLI Interface

```bash
# Basic usage
chonk process document.pdf --output chunks.jsonl

# With options
chonk process document.pdf \
  --chunker recursive \
  --chunk-size 400 \
  --overlap 50 \
  --output chunks.jsonl

# Launch GUI
chonk gui

# Launch GUI with file
chonk gui document.pdf

# Batch process
chonk batch ./documents/ --output ./chunks/

# Compare chunking strategies
chonk compare document.pdf --chunkers fixed,recursive,semantic
```

---

## API Endpoints (FastAPI)

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="CHONK API")

@app.post("/api/load")
async def load_document(file: UploadFile = File(...)):
    """Load and parse a document, return UDM."""
    pass

@app.post("/api/chunk")
async def chunk_document(document_id: str, config: ChunkConfig):
    """Apply chunking strategy to loaded document."""
    pass

@app.post("/api/chunks/{chunk_id}/split")
async def split_chunk(chunk_id: str, split_position: int):
    """Split a chunk at the given character position."""
    pass

@app.post("/api/chunks/merge")
async def merge_chunks(chunk_ids: List[str]):
    """Merge multiple adjacent chunks."""
    pass

@app.put("/api/chunks/{chunk_id}")
async def update_chunk(chunk_id: str, updates: ChunkUpdate):
    """Update chunk content or metadata."""
    pass

@app.post("/api/export")
async def export_chunks(document_id: str, format: str):
    """Export chunks in specified format."""
    pass

@app.get("/api/quality/{document_id}")
async def get_quality_report(document_id: str):
    """Get quality analysis for all chunks."""
    pass

# Retrieval Testing Endpoints
@app.post("/api/test/index")
async def index_for_testing(document_id: str):
    """Index all chunks for retrieval testing. Called after chunking."""
    pass

@app.post("/api/test/search")
async def test_search(query: str, top_k: int = 5):
    """Search chunks with a test query, return ranked results with scores."""
    pass

@app.get("/api/test/status")
async def get_index_status(document_id: str):
    """Check if chunks are indexed and ready for testing."""
    pass
```

---

## Dependencies

### Python (requirements.txt)
```
# Core
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0

# Document loading
pymupdf>=1.23.0
pdfplumber>=0.10.0
python-docx>=1.0.0
python-pptx>=0.6.21
beautifulsoup4>=4.12.0
readability-lxml>=0.8.1
markdown-it-py>=3.0.0

# Parsing & NLP
spacy>=3.7.0
tiktoken>=0.5.0

# Retrieval testing (local embeddings)
sentence-transformers>=2.2.0
numpy>=1.24.0

# Optional: OCR
# pytesseract>=0.3.10
# easyocr>=1.7.0

# Optional: OpenAI embeddings for testing
# openai>=1.0.0

# Dev
pytest>=7.4.0
black>=23.0.0
ruff>=0.1.0
```

### Node (package.json)
```json
{
  "name": "chonk-ui",
  "version": "0.1.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "pdfjs-dist": "^3.11.0",
    "@radix-ui/react-*": "latest",
    "tailwindcss": "^3.4.0",
    "zustand": "^4.4.0"
  },
  "devDependencies": {
    "electron": "^28.0.0",
    "electron-builder": "^24.0.0",
    "vite": "^5.0.0",
    "typescript": "^5.3.0"
  }
}
```

---

## File Structure for Claude CLI

When starting development, create this structure:

```
chonk/
├── README.md
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md              # Instructions for Claude CLI
├── src/
│   └── chonk/
│       ├── __init__.py
│       ├── cli.py
│       ├── server.py
│       ├── core/
│       ├── loaders/
│       ├── parsers/
│       ├── chunkers/
│       ├── exporters/
│       └── utils/
├── tests/
│   ├── test_loaders.py
│   ├── test_chunkers.py
│   └── fixtures/
│       ├── sample.pdf
│       ├── sample.docx
│       └── sample.md
└── ui/
    ├── package.json
    ├── electron/
    └── src/
```

---

## CLAUDE.md (for Claude CLI)

```markdown
# CHONK Development Guide

## Project Overview
CHONK is a visual document chunking tool for RAG pipelines. It loads documents,
parses them into blocks, chunks the blocks, provides a GUI for refinement,
and lets users TEST retrieval before exporting.

**The Core Loop:**
Drop file → Auto-chunk → Live preview → Test queries → Refine → Export

## Key Concepts
- **Block**: A semantic unit from the source document (paragraph, heading, table)
- **Chunk**: A group of blocks that will become one embedding
- **UDM**: Unified Document Model - the internal representation
- **Retrieval Tester**: Local embedding + search to test chunks before export

## Architecture
- Python backend (FastAPI) handles document processing + retrieval testing
- Electron + React frontend provides the visual editor
- sentence-transformers for local embeddings (no API key needed)
- Communication via REST API on localhost

## Development Commands
```bash
# Backend
cd src && uvicorn chonk.server:app --reload

# Frontend
cd ui && npm run dev

# Run Electron
cd ui && npm run electron:dev
```

## Current Focus
MVP Phase 1: 
- PDF/DOCX/HTML/MD/TXT loading
- Fixed + recursive + hierarchy chunking
- Visual editor with merge/split
- Retrieval testing (query → see which chunks match)
- JSONL/JSON export

## Code Style
- Python: Black + Ruff, type hints required
- TypeScript: Strict mode, functional components
- Test all loaders, chunkers, and retrieval tester

## Key Files
- src/chonk/core/document.py - UDM dataclasses
- src/chonk/server.py - API endpoints
- src/chonk/testing/searcher.py - Retrieval tester
- ui/src/components/DocumentViewer.tsx - Main viewer
- ui/src/components/ChunkOverlay.tsx - Visual chunk boundaries
- ui/src/components/RetrievalTester.tsx - Query testing panel
```

---

## Sample Test Fixtures

Include these test documents in tests/fixtures/:

1. **sample.pdf** - 3-page document with:
   - Headings (H1, H2, H3)
   - Body paragraphs
   - One table
   - One image
   - Page numbers/headers

2. **sample.docx** - Similar structure to PDF

3. **sample.md** - Markdown with:
   - Multiple heading levels
   - Code blocks
   - Lists
   - Links

4. **complex.pdf** - Edge cases:
   - Multi-column layout
   - Tables spanning pages
   - Footnotes
   - Scanned page (for OCR testing)

---

## Getting Started Prompt for Claude CLI

When you start working with Claude CLI, use this prompt:

```
I'm building CHONK, a visual document chunking tool for RAG. Read CLAUDE.md for context.

The core value prop: "Drop docs → See chunks → Test queries → Export JSON"

Current task: [describe what you want to build]

Start by:
1. Setting up the Python package structure
2. Implementing the UDM dataclasses in core/document.py
3. Creating the base loader class
4. Implementing the PDF loader
5. Implementing the RetrievalTester with sentence-transformers

Use pytest for testing. Create fixtures as needed.
```
