"""
Build fully nested hierarchical structure with separated heading/content.

Output structure:
{
  "section_id": "E.5.3.5",
  "heading": "Maintenance work packages...",
  "heading_block_id": "docling_blk_3783",
  "content": "Body text without heading...",
  "content_block_ids": ["docling_blk_3784", "docling_blk_3785"],
  "token_count": 290,
  "quality_score": 1.0,
  "page_range": [344, 344],
  "children": [...]
}
"""
import json
import re
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chonk.core.document import Block, BlockType

class HierarchyNode:
    """A node in the document hierarchy tree."""

    def __init__(
        self,
        section_id: str | None = None,
        heading: str | None = None,
        heading_block: Block | None = None,
        level: int = 0,
    ):
        self.section_id = section_id
        self.heading = heading
        self.heading_block = heading_block
        self.level = level
        self.content_blocks: list[Block] = []
        self.children: list[HierarchyNode] = []
        self.parent: HierarchyNode | None = None

    def add_child(self, child: "HierarchyNode") -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def add_content(self, block: Block) -> None:
        """Add a content block to this section."""
        self.content_blocks.append(block)

    @property
    def total_token_count(self) -> int:
        """Calculate total tokens including heading and content."""
        from chonk.utils.tokens import TokenCounter
        counter = TokenCounter()

        total = 0
        if self.heading_block:
            total += counter.count(self.heading_block.content)
        for block in self.content_blocks:
            total += counter.count(block.content)
        return total

    @property
    def content_text(self) -> str:
        """Get combined content text (without heading)."""
        return "\n\n".join(block.content for block in self.content_blocks)

    @property
    def page_range(self) -> list[int]:
        """Get page range for this section."""
        pages = set()
        if self.heading_block and self.heading_block.page:
            pages.add(self.heading_block.page)
        for block in self.content_blocks:
            if block.page:
                pages.add(block.page)
        if pages:
            return [min(pages), max(pages)]
        return [0, 0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "heading_block_id": self.heading_block.id if self.heading_block else None,
            "heading_level": self.level,
            "content": self.content_text,
            "content_block_ids": [b.id for b in self.content_blocks],
            "token_count": self.total_token_count,
            "page_range": self.page_range,
            "block_count": len(self.content_blocks) + (1 if self.heading_block else 0),
            "children": [child.to_dict() for child in self.children],
        }


def extract_section_id(heading_text: str) -> str | None:
    """
    Extract section ID from heading text.

    Examples:
    "E.5.3.5 Maintenance work packages..." -> "E.5.3.5"
    "3.66 Graphic(s)." -> "3.66"
    "FOREWORD" -> "FOREWORD"
    """
    # Try numbered section (e.g., "E.5.3.5" or "3.66")
    match = re.match(r'^([A-Z]?\d+(?:\.\d+)*)', heading_text.strip())
    if match:
        return match.group(1)

    # Otherwise use first 50 chars as ID
    return heading_text.strip()[:50]


def build_hierarchy_tree(blocks: list[Block]) -> HierarchyNode:
    """
    Build a hierarchical tree from blocks.

    Strategy:
    1. Create root node
    2. For each heading block, create a node
    3. Attach content blocks to the current heading
    4. Build parent-child relationships based on heading levels
    """
    root = HierarchyNode(section_id="root", heading="Document Root", level=0)
    current_node = root
    node_stack = [root]  # Stack to track hierarchy

    for block in blocks:
        if block.type == BlockType.HEADING:
            # Create new section node
            section_id = extract_section_id(block.content)
            heading_level = block.heading_level or 1

            node = HierarchyNode(
                section_id=section_id,
                heading=block.content,
                heading_block=block,
                level=heading_level,
            )

            # Find the right parent based on level
            # Pop stack until we find a parent with lower level
            while len(node_stack) > 1 and node_stack[-1].level >= heading_level:
                node_stack.pop()

            parent = node_stack[-1]
            parent.add_child(node)
            node_stack.append(node)
            current_node = node

        else:
            # Add content to current section
            current_node.add_content(block)

    return root


def count_nodes(node: HierarchyNode) -> int:
    """Count total nodes in tree."""
    return 1 + sum(count_nodes(child) for child in node.children)


def count_leaf_nodes(node: HierarchyNode) -> int:
    """Count leaf nodes (sections with no children)."""
    if not node.children:
        return 1
    return sum(count_leaf_nodes(child) for child in node.children)


def print_tree(node: HierarchyNode, indent: int = 0, max_depth: int = 3) -> None:
    """Print tree structure for visualization."""
    if indent > max_depth:
        return

    prefix = "  " * indent
    heading_preview = (node.heading or "ROOT")[:60]
    content_preview = node.content_text[:50].replace('\n', ' ')

    if node.heading_block:
        print(f"{prefix}[{node.section_id}] {heading_preview}")
        print(f"{prefix}  Blocks: {len(node.content_blocks)}, Tokens: {node.total_token_count}, Pages: {node.page_range}")
        if content_preview:
            print(f"{prefix}  Content: {content_preview}...")

    for child in node.children:
        print_tree(child, indent + 1, max_depth)


def main():
    blocks_file = Path("MIL-STD-extraction-blocks-HIERARCHY.json")

    if not blocks_file.exists():
        print(f"ERROR: {blocks_file} not found!")
        return

    print("=" * 70)
    print("BUILDING NESTED HIERARCHY")
    print("=" * 70)
    print(f"[LOADING] {blocks_file.name}...")

    with open(blocks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct blocks
    from chonk.core.document import BoundingBox

    blocks = []
    for b in data["blocks"]:
        bbox = None
        if b["bbox"]:
            bbox = BoundingBox(
                x1=b["bbox"]["x1"],
                y1=b["bbox"]["y1"],
                x2=b["bbox"]["x2"],
                y2=b["bbox"]["y2"],
                page=b["bbox"]["page"],
            )

        block = Block(
            id=b["id"],
            type=BlockType(b["type"]),
            content=b["content"],
            bbox=bbox,
            page=b["page"],
            heading_level=b.get("heading_level"),
            metadata=b.get("metadata", {}),
        )
        blocks.append(block)

    headings = [b for b in blocks if b.type == BlockType.HEADING]
    print(f"[LOADED] {len(blocks)} blocks ({len(headings)} headings)")
    print()

    # Build hierarchy tree
    print("[BUILDING] Constructing hierarchy tree...")
    root = build_hierarchy_tree(blocks)

    total_nodes = count_nodes(root) - 1  # Exclude root
    leaf_nodes = count_leaf_nodes(root) - 1  # Exclude root

    print(f"[COMPLETE] Built hierarchy tree")
    print(f"  Total sections: {total_nodes}")
    print(f"  Leaf sections: {leaf_nodes}")
    print(f"  Max depth: {max(n.level for n in _get_all_nodes(root))}")
    print()

    # Show tree preview
    print("[TREE PREVIEW] First 3 levels:")
    print("-" * 70)
    print_tree(root, max_depth=3)
    print()

    # Convert to JSON
    print("[EXPORTING] Converting to JSON...")

    # Export full tree
    output_data = {
        "document": data["document"],
        "hierarchy": {
            "total_sections": total_nodes,
            "leaf_sections": leaf_nodes,
            "max_depth": max(n.level for n in _get_all_nodes(root)),
        },
        "sections": [child.to_dict() for child in root.children],  # Don't include root
    }

    output_file = Path("MIL-STD-extraction-NESTED-HIERARCHY.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Nested hierarchy saved to: {output_file}")
    print(f"        File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Show example section
    print("[EXAMPLE] Sample nested section:")
    print("-" * 70)
    # Find a section with children
    example = None
    for node in _get_all_nodes(root):
        if node.children and node.content_blocks and node != root:
            example = node
            break

    if example:
        example_dict = example.to_dict()
        print(json.dumps(example_dict, indent=2)[:1000] + "...")
    print()

    # Statistics
    print("=" * 70)
    print("[STATISTICS]")
    print("=" * 70)

    all_nodes = [n for n in _get_all_nodes(root) if n != root]
    nodes_with_content = [n for n in all_nodes if n.content_blocks]
    nodes_with_children = [n for n in all_nodes if n.children]

    print(f"Total sections: {len(all_nodes)}")
    print(f"Sections with content: {len(nodes_with_content)} ({100*len(nodes_with_content)/len(all_nodes):.1f}%)")
    print(f"Sections with subsections: {len(nodes_with_children)} ({100*len(nodes_with_children)/len(all_nodes):.1f}%)")
    print()

    # Token distribution
    token_counts = [n.total_token_count for n in nodes_with_content]
    if token_counts:
        print("Token distribution:")
        print(f"  Avg: {sum(token_counts) / len(token_counts):.1f}")
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
    print()

    print("=" * 70)
    print("[SUCCESS] Nested hierarchy complete!")
    print(f"  - {len(all_nodes)} sections with parent-child relationships")
    print(f"  - Heading and content separated in each section")
    print(f"  - Ready for hierarchical RAG or display")
    print("=" * 70)


def _get_all_nodes(node: HierarchyNode) -> list[HierarchyNode]:
    """Get all nodes in tree (DFS)."""
    nodes = [node]
    for child in node.children:
        nodes.extend(_get_all_nodes(child))
    return nodes


if __name__ == "__main__":
    main()
