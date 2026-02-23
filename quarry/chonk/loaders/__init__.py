"""Document loaders for CHONK."""

from chonk.loaders.base import BaseLoader, LoaderError, LoaderRegistry
from chonk.loaders.pdf import PDFLoader
from chonk.loaders.docx import DocxLoader
from chonk.loaders.markdown import MarkdownLoader
from chonk.loaders.text import TextLoader

__all__ = [
    "BaseLoader",
    "LoaderError",
    "LoaderRegistry",
    "PDFLoader",
    "DocxLoader",
    "MarkdownLoader",
    "TextLoader",
]
