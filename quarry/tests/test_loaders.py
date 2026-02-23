"""
Tests for CHONK document loaders.
"""

from pathlib import Path

import pytest

from chonk.core.document import BlockType
from chonk.loaders import LoaderRegistry
from chonk.loaders.base import BaseLoader, LoaderError
from chonk.loaders.markdown import MarkdownLoader
from chonk.loaders.text import TextLoader


class TestLoaderRegistry:
    """Tests for LoaderRegistry."""

    def test_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = LoaderRegistry.supported_extensions()

        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".md" in extensions
        assert ".txt" in extensions

    def test_get_loader_for_pdf(self):
        """Test getting loader for PDF."""
        loader = LoaderRegistry.get_loader(Path("test.pdf"))
        assert loader is not None

    def test_get_loader_for_markdown(self):
        """Test getting loader for markdown."""
        loader = LoaderRegistry.get_loader(Path("test.md"))
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_for_text(self):
        """Test getting loader for text."""
        loader = LoaderRegistry.get_loader(Path("test.txt"))
        assert isinstance(loader, TextLoader)

    def test_get_loader_for_unknown(self):
        """Test getting loader for unknown extension."""
        loader = LoaderRegistry.get_loader(Path("test.xyz"))
        assert loader is None


class TestTextLoader:
    """Tests for TextLoader."""

    def test_text_loader_creation(self):
        """Test text loader instantiation."""
        loader = TextLoader()
        assert ".txt" in loader.SUPPORTED_EXTENSIONS

    def test_text_loader_load(self, temp_dir, sample_text_content):
        """Test loading a text file."""
        file_path = temp_dir / "test.txt"
        file_path.write_text(sample_text_content)

        loader = TextLoader()
        doc = loader.load_document(file_path)

        assert doc.source_type == "txt"
        assert len(doc.blocks) > 0
        assert doc.metadata.word_count > 0

    def test_text_loader_paragraphs(self, temp_dir):
        """Test that text loader creates blocks for paragraphs."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        file_path = temp_dir / "paragraphs.txt"
        file_path.write_text(content)

        loader = TextLoader()
        doc = loader.load_document(file_path)

        # Should have blocks for each paragraph
        text_blocks = [b for b in doc.blocks if b.type == BlockType.TEXT]
        assert len(text_blocks) == 3

    def test_text_loader_missing_file(self, temp_dir):
        """Test loading missing file raises error."""
        loader = TextLoader()

        with pytest.raises(LoaderError):
            loader.load(temp_dir / "nonexistent.txt")


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_markdown_loader_creation(self):
        """Test markdown loader instantiation."""
        loader = MarkdownLoader()
        assert ".md" in loader.SUPPORTED_EXTENSIONS

    def test_markdown_loader_load(self, temp_dir, sample_markdown_content):
        """Test loading a markdown file."""
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_markdown_content)

        loader = MarkdownLoader()
        doc = loader.load_document(file_path)

        assert doc.source_type == "md"
        assert len(doc.blocks) > 0

    def test_markdown_loader_headings(self, temp_dir, sample_markdown_content):
        """Test that markdown loader extracts headings."""
        file_path = temp_dir / "headings.md"
        file_path.write_text(sample_markdown_content)

        loader = MarkdownLoader()
        doc = loader.load_document(file_path)

        heading_blocks = [b for b in doc.blocks if b.type == BlockType.HEADING]
        assert len(heading_blocks) > 0

        # Check heading levels are set
        h1_blocks = [b for b in heading_blocks if b.heading_level == 1]
        assert len(h1_blocks) >= 1

    def test_markdown_loader_code_blocks(self, temp_dir):
        """Test that markdown loader extracts code blocks."""
        content = """# Code Example

Here is some code:

```python
def hello():
    print("Hello!")
```

And more text after.
"""
        file_path = temp_dir / "code.md"
        file_path.write_text(content)

        loader = MarkdownLoader()
        doc = loader.load_document(file_path)

        code_blocks = [b for b in doc.blocks if b.type == BlockType.CODE]
        assert len(code_blocks) == 1
        assert "def hello" in code_blocks[0].content


class TestLoaderIntegration:
    """Integration tests for document loading."""

    def test_load_and_access_content(self, temp_dir, sample_markdown_content):
        """Test loading document and accessing content."""
        file_path = temp_dir / "doc.md"
        file_path.write_text(sample_markdown_content)

        doc = LoaderRegistry.load_document(file_path)

        # Should be able to access all content
        all_content = " ".join(b.content for b in doc.blocks)
        assert "Document Title" in all_content
        assert "Section One" in all_content

    def test_load_sets_metadata(self, temp_dir, sample_text_content):
        """Test that loading sets document metadata."""
        file_path = temp_dir / "meta.txt"
        file_path.write_text(sample_text_content)

        doc = LoaderRegistry.load_document(file_path)

        assert doc.metadata.file_size_bytes > 0
        assert doc.metadata.word_count > 0
        assert doc.loader_used == "text"
