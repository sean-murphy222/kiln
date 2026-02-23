"""
Pytest configuration and fixtures for CHONK tests.
"""

import tempfile
from pathlib import Path

import pytest

from chonk.core.document import (
    Block,
    BlockType,
    Chunk,
    ChonkDocument,
    ChonkProject,
    DocumentMetadata,
)


@pytest.fixture
def sample_blocks() -> list[Block]:
    """Create sample blocks for testing."""
    return [
        Block(
            id="block_1",
            type=BlockType.HEADING,
            content="Introduction",
            page=1,
            heading_level=1,
        ),
        Block(
            id="block_2",
            type=BlockType.TEXT,
            content="This is the first paragraph of content. It contains important information about the topic being discussed.",
            page=1,
        ),
        Block(
            id="block_3",
            type=BlockType.TEXT,
            content="Here is another paragraph that continues the discussion with more details and examples.",
            page=1,
        ),
        Block(
            id="block_4",
            type=BlockType.HEADING,
            content="Methods",
            page=2,
            heading_level=1,
        ),
        Block(
            id="block_5",
            type=BlockType.TEXT,
            content="The methods section describes the approach used in this study. We employed several techniques.",
            page=2,
        ),
        Block(
            id="block_6",
            type=BlockType.CODE,
            content="def example():\n    return 'Hello, World!'",
            page=2,
        ),
        Block(
            id="block_7",
            type=BlockType.HEADING,
            content="Results",
            page=3,
            heading_level=1,
        ),
        Block(
            id="block_8",
            type=BlockType.TEXT,
            content="The results show significant improvements across all metrics measured in this experiment.",
            page=3,
        ),
    ]


@pytest.fixture
def sample_document(sample_blocks: list[Block]) -> ChonkDocument:
    """Create a sample document for testing."""
    doc = ChonkDocument(
        id="doc_1",
        source_path=Path("test_document.pdf"),
        source_type="pdf",
        blocks=sample_blocks,
        chunks=[],
        metadata=DocumentMetadata(
            title="Test Document",
            author="Test Author",
            page_count=3,
            word_count=150,
            file_size_bytes=1024,
        ),
    )
    return doc


@pytest.fixture
def sample_project(sample_document: ChonkDocument) -> ChonkProject:
    """Create a sample project for testing."""
    project = ChonkProject(
        id="project_1",
        name="Test Project",
        documents=[sample_document],
    )
    return project


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Sample PDF bytes for testing (minimal valid PDF)."""
    # This is a minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
312
%%EOF"""


@pytest.fixture
def sample_markdown_content() -> str:
    """Sample Markdown content for testing."""
    return """# Document Title

This is an introduction paragraph with some text.

## Section One

Content for section one. It has multiple sentences. Here is another one.

### Subsection 1.1

More detailed content in the subsection.

## Section Two

Another section with different content.

```python
def hello():
    print("Hello, World!")
```

## Conclusion

Final thoughts on the topic.
"""


@pytest.fixture
def sample_text_content() -> str:
    """Sample plain text content for testing."""
    return """This is a sample document for testing purposes.

It contains multiple paragraphs of text that can be used to test the chunking functionality.

Each paragraph should be separated by blank lines, which helps the chunker identify natural boundaries.

The document also includes some longer sentences that might need to be split if they exceed the token limit for a single chunk.

Finally, this is the last paragraph of the test document. It provides additional content for testing various edge cases in the chunking algorithm.
"""
