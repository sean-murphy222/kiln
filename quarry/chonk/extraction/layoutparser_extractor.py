"""
LayoutParser extractor (Tier 3).

Uses LayoutParser with deep learning models for complex document layouts.
Best for academic papers, multi-column documents, and scanned PDFs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chonk.core.document import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
)
from chonk.extraction.strategy import ExtractionResult, ExtractionTier


# Mapping from LayoutParser labels to CHONK BlockTypes
LAYOUTPARSER_TYPE_MAP = {
    "text": BlockType.TEXT,
    "title": BlockType.HEADING,
    "heading": BlockType.HEADING,
    "section": BlockType.HEADING,
    "list": BlockType.LIST,
    "table": BlockType.TABLE,
    "figure": BlockType.IMAGE,
    "image": BlockType.IMAGE,
    "caption": BlockType.TEXT,
    "code": BlockType.CODE,
    "formula": BlockType.CODE,
    "equation": BlockType.CODE,
    "header": BlockType.TEXT,
    "footer": BlockType.TEXT,
    "page-number": BlockType.TEXT,
    "footnote": BlockType.TEXT,
}


class LayoutParserExtractor:
    """
    Tier 3: AI-powered extraction using LayoutParser.

    Uses deep learning models (Detectron2) to detect document layout.
    Best for:
    - Academic papers with complex layouts
    - Multi-column documents
    - Scanned PDFs (with OCR)
    - Documents with figures and tables

    Requires: pip install chonk[ai]
    Note: May require GPU for reasonable performance.
    """

    # Available pre-trained models
    MODELS = {
        "publaynet": "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        "publaynet_mask": "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
        "prima": "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
        "newspaper": "lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config",
        "tablebank": "lp://TableBank/faster_rcnn_R_50_FPN_3x/config",
        "hjdataset": "lp://HJDataset/faster_rcnn_R_50_FPN_3x/config",
    }

    def __init__(self, model: str = "publaynet") -> None:
        """
        Initialize LayoutParser extractor.

        Args:
            model: Model to use. Options:
                - "publaynet": General documents (default)
                - "publaynet_mask": Higher accuracy, slower
                - "prima": Historical documents
                - "newspaper": Newspaper layouts
                - "tablebank": Table-focused
                - "hjdataset": Japanese documents
        """
        self._model_name = model
        self._model = None
        self._warnings: list[str] = []
        self._layoutparser_available = self._check_available()

    @property
    def tier(self) -> ExtractionTier:
        return ExtractionTier.AI

    def is_available(self) -> bool:
        """Check if LayoutParser and dependencies are installed."""
        return self._layoutparser_available

    def _check_available(self) -> bool:
        """Check if LayoutParser can be imported."""
        try:
            import layoutparser as lp
            return True
        except ImportError:
            return False

    def _load_model(self) -> Any:
        """Load the LayoutParser model (lazy loading)."""
        if self._model is not None:
            return self._model

        if not self._layoutparser_available:
            raise RuntimeError(
                "LayoutParser is not installed. Install with: pip install chonk[ai]"
            )

        try:
            import layoutparser as lp

            model_config = self.MODELS.get(self._model_name)
            if not model_config:
                self._warnings.append(
                    f"Unknown model '{self._model_name}', using 'publaynet'"
                )
                model_config = self.MODELS["publaynet"]

            # Load Detectron2 model
            self._model = lp.Detectron2LayoutModel(
                model_config,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={
                    0: "text",
                    1: "title",
                    2: "list",
                    3: "table",
                    4: "figure",
                },
            )

            return self._model

        except Exception as e:
            raise RuntimeError(f"Failed to load LayoutParser model: {e}")

    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract content using LayoutParser (Tier 3).

        Args:
            path: Path to the document

        Returns:
            ExtractionResult with blocks and metadata
        """
        self._warnings = []

        if not self._layoutparser_available:
            raise RuntimeError(
                "LayoutParser is not installed. Install with: pip install chonk[ai]"
            )

        try:
            import layoutparser as lp
            import pdf2image
            import pytesseract
        except ImportError as e:
            self._warnings.append(f"Missing dependency: {e}")
            # Fall back to enhanced or fast
            try:
                from chonk.extraction.docling_extractor import DoclingExtractor
                extractor = DoclingExtractor()
                if extractor.is_available():
                    return extractor.extract(path)
            except Exception:
                pass
            from chonk.extraction.fast_extractor import FastExtractor
            return FastExtractor().extract(path)

        # Load model
        try:
            model = self._load_model()
        except Exception as e:
            self._warnings.append(f"Model loading failed: {e}")
            from chonk.extraction.fast_extractor import FastExtractor
            return FastExtractor().extract(path)

        # Convert PDF to images
        try:
            images = pdf2image.convert_from_path(path, dpi=150)
        except Exception as e:
            self._warnings.append(f"PDF to image conversion failed: {e}")
            from chonk.extraction.fast_extractor import FastExtractor
            return FastExtractor().extract(path)

        # Process each page
        all_blocks: list[Block] = []
        block_id = 0

        for page_num, image in enumerate(images, start=1):
            try:
                # Detect layout
                layout = model.detect(image)

                # Sort by reading order (top to bottom, left to right)
                layout = lp.Layout([
                    b for b in sorted(
                        layout,
                        key=lambda x: (x.block.y_1, x.block.x_1)
                    )
                ])

                # Extract text from each detected region
                for element in layout:
                    block = self._process_layout_element(
                        element, image, page_num, block_id, pytesseract
                    )
                    if block:
                        all_blocks.append(block)
                        block_id += 1

            except Exception as e:
                self._warnings.append(f"Page {page_num} processing error: {e}")
                continue

        # Extract metadata
        metadata = self._extract_metadata(path, len(images), all_blocks)

        return ExtractionResult(
            blocks=all_blocks,
            metadata=metadata,
            tier_used=ExtractionTier.AI,
            extraction_info={
                "extractor": "layoutparser",
                "model": self._model_name,
                "page_count": len(images),
                "element_count": len(all_blocks),
            },
            warnings=self._warnings,
        )

    def _process_layout_element(
        self,
        element: Any,
        image: Any,
        page_num: int,
        block_id: int,
        pytesseract: Any,
    ) -> Block | None:
        """Process a single layout element into a Block."""
        try:
            # Get element type
            element_type = element.type.lower() if element.type else "text"
            block_type = LAYOUTPARSER_TYPE_MAP.get(element_type, BlockType.TEXT)

            # Get bounding box
            bbox = element.block
            x1, y1, x2, y2 = bbox.x_1, bbox.y_1, bbox.x_2, bbox.y_2

            # Crop region from image
            import numpy as np
            from PIL import Image

            img_array = np.array(image)
            cropped = img_array[int(y1):int(y2), int(x1):int(x2)]
            cropped_img = Image.fromarray(cropped)

            # Extract text using OCR
            content = ""
            if block_type != BlockType.IMAGE:
                try:
                    content = pytesseract.image_to_string(
                        cropped_img,
                        config="--psm 6"  # Assume uniform block of text
                    ).strip()
                except Exception as e:
                    self._warnings.append(f"OCR error on block {block_id}: {e}")

            # For images/figures, use placeholder
            if block_type == BlockType.IMAGE or not content:
                if block_type == BlockType.IMAGE:
                    content = f"[Figure on page {page_num}]"
                elif block_type == BlockType.TABLE:
                    content = f"[Table on page {page_num}]"
                else:
                    return None  # Skip empty text blocks

            # Determine heading level for titles
            heading_level = None
            if block_type == BlockType.HEADING:
                # Estimate level based on text size (larger box = higher level)
                area = (x2 - x1) * (y2 - y1)
                if area > 50000:
                    heading_level = 1
                elif area > 20000:
                    heading_level = 2
                else:
                    heading_level = 3

            return Block(
                id=f"lp_blk_{block_id}",
                type=block_type,
                content=content,
                bbox=BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    page=page_num,
                ),
                page=page_num,
                heading_level=heading_level,
                confidence=element.score if hasattr(element, "score") else 0.0,
                metadata={
                    "source": "layoutparser",
                    "detected_type": element_type,
                    "confidence": element.score if hasattr(element, "score") else 0.0,
                },
            )

        except Exception as e:
            self._warnings.append(f"Element processing error: {e}")
            return None

    def _extract_metadata(
        self, path: Path, page_count: int, blocks: list[Block]
    ) -> DocumentMetadata:
        """Extract metadata from processed document."""
        # Count words from all text blocks
        word_count = sum(
            len(b.content.split())
            for b in blocks
            if b.type in (BlockType.TEXT, BlockType.HEADING)
        )

        return DocumentMetadata(
            page_count=page_count,
            word_count=word_count,
            file_size_bytes=path.stat().st_size,
            custom={
                "extractor": "layoutparser",
                "model": self._model_name,
                "tier": "ai",
            },
        )
